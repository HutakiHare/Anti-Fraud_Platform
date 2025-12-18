import os
import io
import base64
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from PIL import Image
from pydantic import BaseModel
from dotenv import load_dotenv

# 引入 Google GenAI SDK
from google import genai
from google.genai import types

# 載入 .env 檔案中的環境變數
load_dotenv()

# --- 配置區塊 ---

# 獲取 API Key。如果環境變數不存在，則會報錯或使用 None
# 💡 如果您堅持要寫死，可以將 os.environ.get("GEMINI_API_KEY") 替換為 "您的_實際_Gemini_API_Key"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # 這裡會因為無法初始化 client 而在運行時報錯
    logging.error("GEMINI_API_KEY 環境變數未設定。請在 .env 檔案中設置您的金鑰。")
    # 為了能讓程式碼跑起來，我們還是初始化 Client，但後面會檢查 Key 是否存在
    
try:
    # 初始化 Gemini Client
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    # 處理初始化錯誤，例如 API Key 格式不正確
    logging.error(f"初始化 Gemini Client 失敗: {e}")
    client = None # 如果初始化失敗，將 client 設為 None

# Gemini 模型設定
MODEL_NAME = "gemini-2.5-flash"

# 自定義 Prompt
# 這是您最主要的控制中心，用來指導模型如何進行詐騙查核和報告輸出。
SYSTEM_PROMPT = """
你是一個專門的反詐騙偵測系統。你的任務是分析使用者提供的內容（文字、語音轉文字、圖片/截圖內容），判斷其潛在的詐騙風險，並以清晰、專業的 Markdown 格式輸出查核報告。

## 輸出格式要求:
1. **必須** 使用 Markdown 格式。
2. 報告開頭必須包含風險等級和一個總結性的標題。
3. 報告中必須包含「**內容分析摘要**」、「**判斷依據**」和「**您的行動 (防詐三步驟)**」這三個主要段落。
4. **判斷依據** 應涵蓋內容中的關鍵詞句、語氣、手法模式或圖片中的可疑元素。
5. **風險等級** 只能是以下三種之一: **嚴重風險 (🚨)**, **中等風險 (⚠️)**, **低風險 (✅)**。
6. 對於高度風險的詞彙，請使用 Markdown 的 **粗體** 標記。

## 風險判斷原則:
- **嚴重風險 (🚨):** 包含立即匯款、點擊不明連結、威脅性語氣、要求提供密碼/OTP、號稱高額回饋且具備極度時間急迫性等。
- **中等風險 (⚠️):** 語氣可疑、涉及金錢但缺乏細節、可能是釣魚訊息但無明顯惡意連結、非官方渠道要求驗證。
- **低風險 (✅):** 正常交易通知、純粹的產品推廣、無法判斷風險的簡短或模糊內容 (但應提醒用戶保持警惕)。

---
"""

# FastAPI 應用初始化
app = FastAPI(
    title="Gemini 多模態防詐騙查核 API",
    description="使用 Google Gemini API 處理文字、音檔和圖片，進行潛在詐騙內容的風險評估。",
    version="1.0.0"
)

# 設置 CORS 中間件 (讓前端可以跨域呼叫)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 允許所有來源，您也可以設定為特定的前端 URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 輔助函數 ---

def get_mime_type(filename: str) -> str:
    """根據副檔名判斷 MIME 類型"""
    ext = filename.split('.')[-1].lower()
    if ext in ['jpg', 'jpeg']:
        return "image/jpeg"
    elif ext == 'png':
        return "image/png"
    elif ext in ['wav', 'mp3', 'flac', 'm4a', 'webm']:
        return "audio/wav" # 暫時使用通用 audio/wav，實際會依據 GenAI 支援度而定
    return "application/octet-stream"

async def check_key_and_client():
    """檢查 API Key 和 Client 是否成功初始化"""
    if not client:
         raise HTTPException(
            status_code=500,
            detail="Gemini 服務未啟動。請檢查您的 GEMINI_API_KEY 是否有效或已設定。"
        )

# --- FastAPI 端點定義 ---

class PromptUpdate(BaseModel):
    """用於更新 Prompt 的資料模型"""
    new_prompt: str

@app.post("/update_prompt")
async def update_prompt(update: PromptUpdate):
    """
    更新系統 Prompt 的 API 接口。
    """
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = update.new_prompt
    return {"message": "系統 Prompt 更新成功", "new_prompt_length": len(SYSTEM_PROMPT)}

@app.get("/get_prompt")
def get_prompt():
    """獲取當前系統 Prompt 的 API 接口。"""
    return {"current_prompt": SYSTEM_PROMPT}


@app.post("/check_scam_report")
async def check_scam_report(
    text: Optional[str] = Form(None), # 文字內容 (可選)
    file: Optional[UploadFile] = File(None), # 音檔或圖片 (可選)
    # 前端上傳的音檔和錄製的音檔都使用 'file' 欄位，圖片也用 'file'。
    # 如果同時有多個檔案，前端可能需要多次呼叫 API 或調整上傳欄位名稱。
    # 這裡我們為了簡化，假設 'file' 是單一媒體文件（音檔或圖片）。
):
    """
    多模態詐騙查核 API 接口。

    - 接受文字 (text)
    - 接受檔案 (file): 音檔 (.mp3, .wav, .webm) 或圖片 (.png, .jpg)
    """
    await check_key_and_client()

    if not text and not file:
        raise HTTPException(status_code=400, detail="請提供至少一項文字、音檔或圖片內容進行查核。")

    # 1. 準備 Gemini API 的內容列表
    content_parts = []
    user_prompt_text = "請根據以下內容生成詐騙查核報告:\n\n"

    # 2. 處理文字輸入
    if text:
        content_parts.append(text)
        user_prompt_text += f"**[文字內容]**: {text}\n"

    # 3. 處理檔案輸入 (音檔或圖片)
    uploaded_file = None
    if file:
        file_bytes = await file.read()
        mime_type = get_mime_type(file.filename)
        
        # 檢查檔案大小限制 (以 MB 為單位)
        file_size_mb = len(file_bytes) / (1024 * 1024)
        if file_size_mb > 50: # 設置一個合理的限制，例如 50MB
            raise HTTPException(status_code=413, detail=f"檔案大小超過限制 ({file_size_mb:.2f}MB)。")
            
        
        try:
            # 檔案處理 (音檔或圖片)
            if mime_type.startswith("image/"):
                # 圖片處理
                img = Image.open(io.BytesIO(file_bytes))
                uploaded_file = client.files.upload(file=img)
                content_parts.append(uploaded_file)
                user_prompt_text += f"**[圖片檔案]** 已上傳並請分析圖片內容。\n"

            elif mime_type.startswith("audio/"):
                # 音檔處理
                # 由於音檔需要先上傳到 Google 服務器，我們直接使用 client.files.upload
                # 注意：GenAI 支援的音檔格式較多，這裡我們保持通用
                # ⚠️ 前端音檔格式如果是 webm，請確認後端是否支援處理。
                uploaded_file = client.files.upload(file=io.BytesIO(file_bytes), mime_type=mime_type)
                content_parts.append(uploaded_file)
                user_prompt_text += f"**[語音檔案]** 已上傳，請先進行語音轉文字 (STT)，然後再根據語音內容進行分析。\n"
                
            else:
                raise HTTPException(status_code=400, detail=f"不支援的檔案類型: {mime_type}")
                
        except Exception as e:
            # 處理檔案上傳或轉換錯誤
            logging.error(f"檔案處理失敗: {e}")
            raise HTTPException(status_code=500, detail=f"檔案處理失敗: {str(e)}")


    # 4. 組合最終給模型的 Prompt
    
    # 這裡我們使用一個簡單的邏輯：如果同時有文字和檔案，則將文字放在前面。
    # 模型的內容列表 (parts) 可以包含多種數據類型 (text, File object, etc.)
    final_content = content_parts + [user_prompt_text]


    # 5. 呼叫 Gemini API
    try:
        logging.info(f"開始呼叫 Gemini API，模型: {MODEL_NAME}")
        
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2, # 較低的 temperature 以獲得更穩定的查核結果
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=final_content,
            config=config,
        )
        
        report = response.text.strip()
        
        # 6. 清理上傳的檔案 (重要!)
        if uploaded_file:
            client.files.delete(name=uploaded_file.name)
            logging.info(f"已刪除暫存檔案: {uploaded_file.name}")
            
        return {"report": report}

    except genai.errors.APIError as e:
        # 處理 GenAI API 錯誤 (例如 Key 錯誤、模型錯誤等)
        logging.error(f"Gemini API 呼叫失敗: {e}")
        # 清理可能尚未清理的檔案
        if uploaded_file:
            client.files.delete(name=uploaded_file.name)
        raise HTTPException(status_code=500, detail=f"Gemini API 呼叫失敗: {str(e)}")
        
    except Exception as e:
        # 處理其他未知錯誤
        logging.error(f"伺服器處理錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"伺服器處理錯誤: {str(e)}")


# --- 運行後端服務 ---
# 您可以使用以下命令運行服務:
# uvicorn main:app --reload --host 0.0.0.0 --port 9001 
# 注意: 確保將 9001 替換成前端代碼中使用的端口 (例如 140.123.105.233:9001)