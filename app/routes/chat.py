from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from typing import List, Optional, Dict, Any
import os
import json
from uuid import uuid4

from app.schemas.chat import ChatRequest, DocumentChatRequest, ChatResponse, ChatMessage
from app.models.qwen_model import qwen_model
from app.utils.file_processing import file_processor
from app.utils.chroma_db import chroma_db

router = APIRouter()

def format_history(history: Optional[List[ChatMessage]]) -> List[Dict[str, str]]:
    return [{"role": msg.role, "content": msg.content} for msg in history] if history else []

def update_chat_history(history: Optional[List[ChatMessage]], user_message: str, assistant_response: str) -> List[ChatMessage]:
    updated_history = history.copy() if history else []
    updated_history.append(ChatMessage(role="user", content=user_message))
    updated_history.append(ChatMessage(role="assistant", content=assistant_response))
    return updated_history

@router.post("/chat", response_model=ChatResponse)
async def chat(
    message: str = Form(...),
    history: Optional[str] = Form(None)
):
    """
    Endpoint for regular chat interaction without additional context, using Form data.
    """
    try:
        parsed_history = []
        if history:
            try:
                parsed_history = json.loads(history)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid history format. Must be a valid JSON list.")
        
        formatted_history = format_history(parsed_history)
        
        # Hardcoded system role
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": message}
        ]
        
        response = qwen_model.generate_response(message, formatted_history)
        updated_history = update_chat_history(parsed_history, message, response)

        return ChatResponse(
            response=response,
            history=updated_history,
            context_used=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@router.post("/upload-document")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Endpoint to upload and process a document
    """
    try:
        file_path = await file_processor.save_upload_file(file)
        texts, metadatas = file_processor.extract_text_from_file(file_path)
        document_id = str(uuid4())
        for metadata in metadatas:
            metadata["document_id"] = document_id
        ids = [f"{document_id}_{i}" for i in range(len(texts))]
        chroma_db.add_documents(texts, metadatas, ids)

        return {
            "document_id": document_id,
            "filename": file.filename,
            "chunks": len(texts),
            "message": "Document processed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.post("/document-chat", response_model=ChatResponse)
async def document_chat(request: DocumentChatRequest):
    """
    Endpoint for chatting with context from an uploaded document
    """
    try:
        if not request.document_id:
            raise HTTPException(status_code=400, detail="Document ID is required")
        
        formatted_history = format_history(request.history)
        search_results = chroma_db.search(
            request.message, 
            n_results=3,
            where={"document_id": request.document_id} if request.document_id else None
        )
        
        if not search_results:
            return ChatResponse(
                response="I couldn't find relevant information in the document to answer your question.",
                history=update_chat_history(request.history, request.message, "I couldn't find relevant information in the document to answer your question."),
                context_used=None
            )
        
        context = "\n\n".join([result["document"] for result in search_results])
        response = qwen_model.generate_response_with_context(request.message, context, formatted_history)
        updated_history = update_chat_history(request.history, request.message, response)
        
        context_info = {
            "document_id": request.document_id,
            "sources": [
                {
                    "metadata": result["metadata"],
                    "relevance_score": 1 - result["distance"]
                }
                for result in search_results
            ]
        }
        
        return ChatResponse(
            response=response,
            history=updated_history,
            context_used=context_info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
