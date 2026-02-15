from fastapi import FastAPI, HTTPException, Security, status, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
import logging

from app.models import VoiceDetectionRequest, VoiceDetectionResponse, ErrorResponse
from app.config import API_KEY
# Graceful import for ML module to prevent startup crashes on Vercel
try:
    from ml.inference import predict_voice_authenticity
except ImportError as e:
    logging.error(f"Failed to import ML module: {e}")
    # Define a dummy function preventing NameError
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
        # Import inside function to avoid top-level overhead if imports fail
        from ml.sota_model import get_detector
        get_detector() # Triggers model loading
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
    # Flatten errors to a single message
    error_msg = "; ".join([f"{e['loc'][-1]}: {e['msg']}" for e in exc.errors()])
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"status": "error", "message": f"Validation error: {error_msg}"}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Handle structured error response
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

# Authentication Dependency
async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key is None or api_key != API_KEY:
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
        
        # Run ML inference
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

# Root endpoint
# Mount Static Files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("app/static/index.html")
