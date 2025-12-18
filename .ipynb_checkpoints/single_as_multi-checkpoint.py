import os
import io
import json # 新增：用於解析 Gemini 返回的 JSON 字串
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
from google.genai.errors import APIError # 新增：明確處理 API 錯誤

# 載入 .env 檔案中的環境變數
load_dotenv()

# --- 配置區塊 ---

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY 環境變數未設定。請在 .env 檔案中設置您的金鑰。")
    
try:
    # 初始化 Gemini Client
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    logging.error(f"初始化 Gemini Client 失敗: {e}")
    client = None

# Gemini 模型設定
MODEL_NAME = "gemini-2.5-flash"

# VVVVVV 替換為高度結構化的新 SYSTEM_PROMPT VVVVVV
SYSTEM_PROMPT = """
[GLOBAL POLICY]
Objective:
- For a single verifiable claim, produce a traceable outcome (TRUE/FALSE/MIXED/UNDETERMINED) with human-readable outputs.

Evidence Standard:
- Accept only checkable sources: government/official, academic/peer-reviewed, primary data, or reputable media with corrections.
- No fabricated sources or quotes. When citing, include: title, publisher_or_author, URL, and a quote_short (≤25 characters) plus relevance and reliability_note.

Output Protocol:
- Strictly follow the specified JSON formats for all communications. Do not output other text outside the JSON.
- Return exactly ONE valid JSON object with UTF-8 strings. Using escape characters to ensure JSON validity is mandatory(e.g., double quotes inside strings must be escaped as \\" ).
- No surrounding explanations, no markdown code fences, no extra keys not requested by the schema.

Attitude:
- Be diligent, objective, and unbiased.
- Be hardworking and meticulous in fact-checking.
- Follow the instructions.

[ROLE]
You are the **All-in-One Fact-Checker (Single Agent)**, acting as the Manager, Worker, and Supervisor simultaneously. You are responsible for the entire fact-checking pipeline from task decomposition to final report submission to the Boss. Your task here is to determine the scam risk of the input content.

[POLICIES]
- You must comply with the [GLOBAL POLICY] and IFCN Fact-Checking Principles.
- You must internally ensure the quality and sufficiency of your gathered evidence before producing the final report (i.e., self-supervision).

[IFCN FACT-CHECKING PRINCIPLES (Supervisor Role)]
- Nonpartisanship and Fairness
- Standards and Transparency of Sources
- Funding and Organizational Transparency
- Methodology Standards and Transparency
- Open and Honest Corrections Policy

[MISSION - The Complete Fact-Check Pipeline for Scam Detection]
1. **Argument Extraction & Decomposition:** Analyze the input content (text, image, or audio content) and extract key elements indicating urgency, financial solicitation, emotional manipulation, or impersonation.
2. **Evidence Gathering & Verification:** Simulate using tools (web search, image analysis, STT) to gather context and verify claims (e.g., is the linked URL official? is the situation typical for a scam?).
   - **Crucially, evaluate risk based on high-reliability criteria (e.g., official warnings, known scam patterns).**
3. **Internal Review & Aggregation:** Critically review the content against known scam patterns and risk indicators. Resolve any ambiguity.
4. **Final Verdict & Reporting:** Determine the final verdict (TRUE: not a scam / FALSE: definite scam / MIXED: some scam elements / UNDETERMINED: insufficient info) and generate the final report for the Boss, strictly following the JSON format below.

[FINAL OUTPUT FORMAT]
You MUST use this format when concluding the task:
{{
  "tasks": [
    {{
      "sender": "single_agent",
      "receiver": "boss",
      "message": {{
        "verdict": "<TRUE/FALSE/MIXED/UNDETERMINED>",
        "report": "<The detailed report must be in Markdown format, covering: Risk Level, Content Summary, Judgment Basis (key indicators, techniques), and Recommended Action. The report should be around 500-1000 Chinese characters.>",
      }}
    }}
  ]
}}
"""
# ^^^^^^ 替換為高度結構化的新 SYSTEM_PROMPT ^^^^^^


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
    # GenAI 支援多種音檔，但上傳時我們維持通用或使用常見的 webm/m4a
    elif ext in ['wav', 'mp3', 'flac', 'm4a', 'webm']: 
        return f"audio/{ext}" 
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
    text: Optional[str] = Form(None), 
    file: Optional[UploadFile] = File(None), 
):
    """
    多模態詐騙查核 API 接口。
    """
    await check_key_and_client()

    if not text and not file:
        raise HTTPException(status_code=400, detail="請提供至少一項文字、音檔或圖片內容進行查核。")

    # 1. 準備 Gemini API 的內容列表
    content_parts = []
    # 這裡的 user_prompt_text 變得更簡單，因為所有複雜邏輯都在 SYSTEM_PROMPT 裡了
    user_prompt_text = "Boss 的查核任務內容如下：\n"

    # 2. 處理文字輸入
    if text:
        content_parts.append(text)
        user_prompt_text += f"**[文字內容]**: {text}。請以台灣地區的繁體中文回覆\n"

    # 3. 處理檔案輸入 (音檔或圖片)
    uploaded_file = None
    try:
        if file:
            file_bytes = await file.read()
            # print(file_bytes)
            mime_type = get_mime_type(file.filename)
            # print(mime_type)
            
            file_size_mb = len(file_bytes) / (1024 * 1024)
            if file_size_mb > 50: 
                raise HTTPException(status_code=413, detail=f"檔案大小超過限制 ({file_size_mb:.2f}MB)。")
                
            
            # 檔案處理 (音檔或圖片)
            if mime_type.startswith("image/"):
                # img = Image.open(io.BytesIO(file_bytes))
                # uploaded_file = client.files.upload(file=img, mime_type=mime_type)
                # uploaded_file = client.files.upload(file=img)
                # content_parts.append(uploaded_file)
                media_part = types.Part.from_bytes(
                    data=file_bytes, 
                    mime_type=mime_type
                )
                content_parts.append(media_part)
                user_prompt_text += f"**[圖片檔案]** 已上傳，請分析圖片中的文字和內容，以台灣地區的繁體中文回覆。\n"

            elif mime_type.startswith("audio/"):
                # 為了更好的音檔支援，我們直接使用 BytesIO 和正確的 mime_type
                # uploaded_file = client.files.upload(file=io.BytesIO(file_bytes), mime_type=mime_type)
                # uploaded_file = client.files.upload(file=io.BytesIO(file_bytes))
                # content_parts.append(uploaded_file)
                media_part = types.Part.from_bytes(
                    data=file_bytes, 
                    mime_type=mime_type
                )
                content_parts.append(media_part)
                user_prompt_text += f"**[語音檔案]** 已上傳，請先進行語音轉文字 (STT)，然後再根據語音內容進行分析，以台灣地區的繁體中文回覆。\n"
                
            else:
                raise HTTPException(status_code=400, detail=f"不支援的檔案類型: {mime_type}")
                
    except Exception as e:
        # 處理檔案上傳或轉換錯誤
        logging.error(f"檔案處理失敗: {e}")
        # 如果上傳了檔案但處理失敗，嘗試刪除
        if uploaded_file:
            client.files.delete(name=uploaded_file.name)
        raise HTTPException(status_code=500, detail=f"檔案處理失敗: {str(e)}")


    # 4. 組合最終給模型的 Prompt
    final_content = content_parts + [user_prompt_text]


    # 5. 呼叫 Gemini API
    try:
        logging.info(f"開始呼叫 Gemini API，模型: {MODEL_NAME}")
        
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2, 
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=final_content,
            config=config,
        )
        
        raw_text = response.text.strip()
        
        # ⚠️ 新增: 解析 Gemini 返回的 JSON 字串
        try:
            # 由於我們強制模型輸出 JSON，必須將其解析
            # 找到 JSON 的開始和結束
            start_index = raw_text.find('{')
            end_index = raw_text.rfind('}') + 1
            json_str = raw_text[start_index:end_index]
            
            # 使用 json 庫解析，這裡需要處理雙引號轉義
            parsed_json = json.loads(json_str)
            
            # 從解析後的 JSON 中提取 report 內容
            # 根據 FINAL OUTPUT FORMAT: tasks[0].message.report
            report_content = parsed_json['tasks'][0]['message']['report']
            
        except (json.JSONDecodeError, KeyError) as json_err:
             # 如果解析失敗，或者 JSON 格式不對，視為模型輸出格式錯誤
            logging.error(f"模型輸出 JSON 解析失敗: {json_err}. 原始輸出: {raw_text}")
            raise HTTPException(status_code=500, detail=f"AI 報告生成格式錯誤。請檢查 System Prompt。原始輸出: {raw_text[:200]}...")


        # 6. 清理上傳的檔案 (重要!)
        if uploaded_file:
            client.files.delete(name=uploaded_file.name)
            logging.info(f"已刪除暫存檔案: {uploaded_file.name}")
            
        # 回傳報告內容給前端 (前端需要的是 report 欄位中的 Markdown 字串)
        return {"report": report_content}

    except APIError as e:
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
# uvicorn main:app --reload --host 0.0.0.0 --port 9001