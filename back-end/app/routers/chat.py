from fastapi import APIRouter, Request
from app.services.openai_client import get_embedding
from app.services.rag import query_qdrant_and_get_text
from app.core.config import qdrant_client, openai_client

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("query", "")
    text = query_qdrant_and_get_text(openai_client, qdrant_client, user_message)

    return {"response": f"You said: {text}"}
