import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.chat import router as chat_router
from backend.api.routes.products import router as products_router


app = FastAPI(
    title="PS-12 Conversational Shopping Assistant",
    description="LangGraph-powered multi-agent product discovery system",
    version="1.0.0",
)
logging.basicConfig(level=logging.INFO)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat_router)
app.include_router(products_router)


@app.get("/")
def root():
    return {"message": "PS-12 Shopping Assistant API is running", "docs": "/docs"}


@app.get("/health")
def health():
    from backend.services.ai import gemini_state
    return {"status": "healthy", "gemini": gemini_state}