from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.chat import router as chat_router
from backend.api.routes.products import router as products_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload reranker singleton at startup so subsequent requests have zero init latency
    try:
        from backend.agents.ranking import get_cross_encoder
        get_cross_encoder()
    except Exception as exc:
        logging.getLogger("startup").warning("CrossEncoder warmup skipped: %s", exc)
    yield


app = FastAPI(
    title="PS-12 Conversational Shopping Assistant",
    description="LangGraph-powered multi-agent product discovery system",
    version="1.0.0",
    lifespan=lifespan,
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
    from backend.services.ai import gemini_state, llm_state
    return {"status": "healthy", "llm": llm_state, "gemini": gemini_state}