"""
TransactionGuard FastAPI Application
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.startup import load_all
from backend.router import score, transactions, graph, metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML assets on startup."""
    load_all()
    yield
    print("[shutdown] API shutting down.")


app = FastAPI(
    title="TransactionGuard API",
    description=(
        "Real-time fraud detection with graph-based ring detection and SHAP explainability. "
        "Built as a portfolio project demonstrating XGBoost + NetworkX + SHAP."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the Vite dev server (localhost:5173) and any origin for demo purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(score.router)
app.include_router(transactions.router)
app.include_router(graph.router)
app.include_router(metrics.router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "TransactionGuard",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
