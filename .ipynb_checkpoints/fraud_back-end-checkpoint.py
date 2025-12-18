import os
import shutil
import uvicorn
import mimetypes
import requests 
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional, Dict, Any
import json
from loguru import logger

# 引入您提供的 Agent 設定檔
from get_prompt import get_manager_agent_prompts, describe_media

# --- API 端點設定 ---
# ⚠️ 請將這裡替換為您 Agent 系統（運行 Manager Agent 的服務）的 IP 和端口
AGENT_SERVICE_URL = "http://140.123.105.233:4050/process_agent_task" 

# --- FastAPI 與環境設定 ---
app = FastAPI(title="防詐 Agent 協調中心", version="1.0")

# 設置暫存目錄
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- 1. 核心 API 呼叫函式 (只保留 Agent 系統呼叫) ---

def call_agent_system(final_prompt: str) -> str:
    """
    直接呼叫 Agent 系統（例如，運行 Gemini 的服務）並獲取最終的 Markdown 報告。
    這個呼叫取代了所有 Port 8080/4050 的邏輯。
    """
    logger.info(f"呼叫 Agent 系統服務: {AGENT_SERVICE_URL}")
    manager_system_prompt, _, _ = get_manager_agent_prompts()
    
    try:
        headers = {"Content-Type": "application/json"}
        
        # 將 Agent 的 System Prompt 和 User Prompt 包裝成服務可接受的格式
        # 這是 Manager Agent 的輸入
        payload = {
            "system_prompt": manager_system_prompt,
            "user_prompt": final_prompt
        }
        
        # 使用 requests 發送 POST 請求
        response = requests.post(AGENT_SERVICE_URL, data=json.dumps(payload), headers=headers, timeout=120)
        response.raise_for_status() # 對 HTTP 錯誤碼 (4xx, 5xx) 拋出異常
        
        agent_result = response.json()
        
        # 假設 Agent 服務回傳格式為 {"report": "Markdown 報告..."}
        if "report" in agent_result:
            return agent_result["report"]
            
        return "Agent 服務返回成功，但缺少 'report' 欄位:\n" + json.dumps(agent_result, indent=2)

    except requests.exceptions.RequestException as e:
        logger.error(f"Agent 系統連線失敗: {e}")
        raise HTTPException(status_code=503, detail=f"Agent 系統服務連線失敗或超時: {e}")
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Agent 系統回傳格式錯誤或報告欄位遺失: {e}")
        raise HTTPException(status_code=500, detail="Agent 系統回傳格式錯誤")


# --- 2. API 端點定義 ---

@app.post("/check_scam_report")
async def check_scam_report(
    # 'text' 欄位現在可能包含: 原始文字、或 STT 轉換後的文字
    text: Optional[str] = Form(None, description="原始文字、或已處理的音檔轉文字結果"),
    # 音檔/圖片檔案仍需上傳，以便我們執行 describe_media 或傳遞給 Agent
    media_file: Optional[UploadFile] = File(None, description="上傳圖片或音檔 (用於描述或 Agent 處理)")
) -> Dict[str, str]:
    """
    統一接收文字或圖片，生成 Manager Agent Prompt，並呼叫 Agent 系統獲取報告。
    """
    
    if not (text or media_file):
        raise HTTPException(
            status_code=400,
            detail="請至少提供文字內容或上傳一個媒體檔案進行查核。"
        )

    # 1. 初始化變數
    uploaded_path = None
    media_description = ""
    claim_text = text if text else "使用者未提供純文字主張。"
    
    try:
        # --- 處理媒體檔案 (圖片或音檔) ---
        if media_file:
            uploaded_file = media_file
            
            # 確定檔案副檔名
            file_extension = mimetypes.guess_extension(uploaded_file.content_type) or ".bin"
            if uploaded_file.filename and '.' in uploaded_file.filename:
                file_extension = '.' + uploaded_file.filename.split('.')[-1]
            
            # 儲存檔案到本地暫存區 (供 describe_media 讀取)
            uploaded_path = os.path.join(UPLOAD_DIR, f"{os.urandom(8).hex()}{file_extension}")
            with open(uploaded_path, "wb") as buffer:
                shutil.copyfileobj(uploaded_file.file, buffer)
            
            # 呼叫 get_prompt.py 中的描述函式 (依賴 genai_hub_digissl)
            media_description = describe_media(uploaded_path)
        
        # --- 2. 準備 Manager Agent 的 Prompt ---
        _, user_prompt_from_boss_template, _ = get_manager_agent_prompts()
        
        # 組合輸入訊息
        claim_section = f"[CLAIM FROM BOSS]\n{claim_text}\n"
        media_section = f"[SUPPLEMENTAL MEDIA DATA]{media_description}\n"
        
        final_input_task = f"{claim_section}\n{media_section}"
        
        final_prompt = user_prompt_from_boss_template + final_input_task
        
        # 3. 呼叫 Agent 服務並返回最終報告
        markdown_report = call_agent_system(final_prompt)
        
        return {"report": markdown_report, "status": "success"}

    except Exception as e:
        logger.error(f"主要查核流程失敗: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"服務內部錯誤: {str(e)}")
        
    finally:
        # 4. 清理暫存檔案 (無論成功或失敗都執行)
        if uploaded_path and os.path.exists(uploaded_path):
            os.remove(uploaded_path)


# --- 3. 運行伺服器 (開發模式) ---
if __name__ == "__main__":
    # 服務將在 9001 端口運行
    uvicorn.run(app, host="0.0.0.0", port=9001)