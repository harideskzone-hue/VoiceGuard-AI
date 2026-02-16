from fastapi import FastAPI, HTTPException, Security, status, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
import logging

from app.models import VoiceDetectionRequest, VoiceDetectionResponse, ErrorResponse
from app.config import API_KEY
from app.api_keys import generate_api_key, validate_key, list_keys, revoke_key, get_stats

# Graceful import for ML module to prevent startup crashes on Vercel
try:
    from ml.inference import predict_voice_authenticity
except ImportError as e:
    logging.error(f"Failed to import ML module: {e}")
    async def predict_voice_authenticity(*args, **kwargs):
        raise HTTPException(status_code=503, detail="ML Service Unavailable")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice_api")

app = FastAPI(
    title="AI Voice Detection API",
    description="Detects AI-generated vs human voice across 5 languages",
    version="1.0.0"
)

# Startup Event to Preload Model
@app.on_event("startup")
async def startup_event():
    """Preload SOTA model on startup to avoid first-request latency"""
    try:
        logger.info("Initializing SOTA Deepfake Detector...")
        from ml.sota_model import get_detector
        get_detector()
        logger.info("SOTA Model preloaded successfully!")
    except Exception as e:
        logger.warning(f"Could not preload SOTA model: {e}")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Security
API_KEY_HEADER = APIKeyHeader(name="x-api-key", auto_error=False)

# Custom Exception Handlers for strict JSON format
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_msg = "; ".join([f"{e['loc'][-1]}: {e['msg']}" for e in exc.errors()])
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"status": "error", "message": f"Validation error: {error_msg}"}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "status" in exc.detail:
         return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": str(exc.detail)}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status": "error", "message": f"Internal server error: {str(exc)}"}
    )

# Authentication Dependency — supports both master key AND generated keys
async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key is None or not validate_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"status": "error", "message": "Invalid API key provided"}
        )
    return api_key

# Health Check Endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "voice-detection-api"}

# Main Detection Endpoint
@app.post(
    "/api/voice-detection",
    response_model=VoiceDetectionResponse,
    responses={
        200: {"model": VoiceDetectionResponse},
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def detect_voice(
    request: VoiceDetectionRequest,
    api_key: str = Security(verify_api_key)
):
    """
    Detect if voice sample is AI-generated or human
    
    - Requires valid x-api-key header
    - Accepts Base64-encoded MP3 audio
    - Returns classification with confidence and explanation
    """
    try:
        logger.info(f"Processing request for language: {request.language}")
        
        result = await predict_voice_authenticity(
            audio_base64=request.audioBase64,
            language=request.language
        )
        
        logger.info(f"Classification: {result['classification']}, Confidence: {result['confidenceScore']}")
        
        return result
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"status": "error", "message": str(e)}
        )
    
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "error", "message": f"Audio processing error: {str(e)}"}
        )


# ============================================================
# API Key Management Endpoints
# ============================================================

class CreateKeyRequest(BaseModel):
    name: str

@app.post("/api/keys/generate")
async def create_api_key(request: CreateKeyRequest):
    """Generate a new API key for external users."""
    try:
        key_record = generate_api_key(name=request.name)
        return {
            "status": "success",
            "message": f"API key '{request.name}' created successfully",
            "key": key_record
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/keys")
async def get_api_keys():
    """List all generated API keys."""
    keys = list_keys()
    stats = get_stats()
    return {
        "status": "success",
        "keys": keys,
        "stats": stats
    }


@app.delete("/api/keys/{key_id}")
async def delete_api_key(key_id: str):
    """Revoke an API key."""
    success = revoke_key(key_id)
    if success:
        return {"status": "success", "message": f"Key {key_id} revoked"}
    raise HTTPException(status_code=404, detail=f"Key {key_id} not found")


@app.get("/api/keys/stats")
async def get_key_stats():
    """Get API key usage statistics."""
    return {"status": "success", "stats": get_stats()}


# Root endpoint  
# Mount Static Files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("app/static/index.html")
