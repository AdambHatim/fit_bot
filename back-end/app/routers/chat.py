from fastapi import APIRouter, Request

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("query", "")
    print("Received from frontend:", user_message)
    return {"response": f"You said: {user_message}"}
