from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI

app = FastAPI()

# 提供 static 資料夾內容給 / 路徑（即首頁）
app.mount("/", StaticFiles(directory="static", html=True), name="static")
