from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat, health

app = FastAPI(title="Fitness RAG API")

# Enable frontend access (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://ragbotfit.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach routes
app.include_router(health.router)
app.include_router(chat.router)

@app.get("/")
def root():
    return {"message": "Welcome to my Fitness RAG API 🚀"}