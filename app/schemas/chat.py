from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    history: Optional[List[ChatMessage]] = Field(default=None, description="Chat history")

class DocumentChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    history: Optional[List[ChatMessage]] = Field(default=None, description="Chat history")
    document_id: Optional[str] = Field(default=None, description="ID of previously uploaded document")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant response")
    history: List[ChatMessage] = Field(..., description="Updated chat history")
    context_used: Optional[Dict[str, Any]] = Field(default=None, description="Information about context used (if any)")