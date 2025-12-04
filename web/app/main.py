"""LEAPS Ranker Web Application - FastAPI Backend."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.routes.leaps import router as leaps_router
from app.routes.credit_spreads import router as credit_spreads_router
from app.routes.iron_condors import router as iron_condors_router
from app.routes.ai_score import router as ai_score_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title="LEAPS Ranker",
    description="LEAPS Option Ranking and ROI Simulator",
    version="1.0.0",
    # Disable docs in production for security
    docs_url="/docs" if os.getenv("ENV", "production") == "development" else None,
    redoc_url=None,
)

# Add rate limiter to app state and exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Get allowed origins from environment or use defaults
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://leaps-ranker-*.run.app,http://localhost:8080,http://127.0.0.1:8080"
).split(",")

# Configure CORS - restrictive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # No credentials needed for read-only app
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    # Only add CSP for HTML responses
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self'; "
            "connect-src 'self'"
        )
    return response

# Mount static files
static_path = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Setup templates
templates_path = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_path))

# Include API routes
app.include_router(leaps_router)
app.include_router(credit_spreads_router)
app.include_router(iron_condors_router)
app.include_router(ai_score_router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main page (LEAPS Ranker)."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/credit-spreads", response_class=HTMLResponse)
async def credit_spreads_page(request: Request):
    """Serve the credit spreads screener page."""
    return templates.TemplateResponse("credit_spreads.html", {"request": request})


@app.get("/iron-condors", response_class=HTMLResponse)
async def iron_condors_page(request: Request):
    """Serve the Iron Condor screener page."""
    return templates.TemplateResponse("iron_condors.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run."""
    return {"status": "healthy", "service": "leaps-ranker"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)
