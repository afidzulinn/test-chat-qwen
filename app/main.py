from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional, List
import uvicorn
import os

from app.routes.chat import router as chat_router

app = FastAPI(title="Qwen2.5 Chatbot API", description="API for chatbot using Qwen2.5 model")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api", tags=["chat"])

# Mount static files
os.makedirs("data/uploaded_files", exist_ok=True)
app.mount("/files", StaticFiles(directory="data/uploaded_files"), name="files")

@app.get("/")
async def root():
    return {"message": "Welcome to Qwen2.5 Chatbot API"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", port=8000, reload=True)