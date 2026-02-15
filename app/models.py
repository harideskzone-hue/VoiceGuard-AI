"""
Pydantic Request/Response Models
================================
Strict JSON schema matching the GUVI hackathon API specification.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional

SUPPORTED_LANGUAGES = {"Tamil", "English", "Hindi", "Malayalam", "Telugu"}
SUPPORTED_FORMATS = {"mp3", "wav", "webm", "ogg", "flac", "m4a", "aac", "mpeg"}


class VoiceDetectionRequest(BaseModel):
    """
    Request body for voice detection.
    
    Matches spec:
    {
        "language": "Tamil | English | Hindi | Malayalam | Telugu",
        "audioFormat": "mp3",
        "audioBase64": "<Base64 MP3 string>"
    }
    """
    language: str = Field(..., description="One of: Tamil, English, Hindi, Malayalam, Telugu")
    audioFormat: str = Field(..., description="Audio format: mp3, wav, webm, ogg, flac, m4a, aac")
    audioBase64: str = Field(..., description="Base64-encoded audio data")
    
    @validator('language')
    def validate_language(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Invalid language. Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}")
        return v
    
    @validator('audioFormat')
    def validate_format(cls, v):
        if v.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Invalid audio format: '{v}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}")
        return v.lower()


class VoiceDetectionResponse(BaseModel):
    """
    Success response — matches problem statement spec exactly.
    
    {
        "status": "success",
        "language": "Tamil",
        "classification": "AI_GENERATED | HUMAN",
        "confidenceScore": 0.0-1.0,
        "explanation": "Short, human-readable reason"
    }
    
    NO extra fields (details, method, etc.) — strict spec compliance.
    """
    status: str = Field(..., description="Always 'success' for valid responses")
    language: str = Field(..., description="Input language echoed back")
    classification: str = Field(..., description="AI_GENERATED or HUMAN")
    confidenceScore: float = Field(..., ge=0.0, le=1.0, description="Detection confidence 0.0-1.0")
    explanation: str = Field(..., description="Human-readable explanation of detection")


class ErrorResponse(BaseModel):
    """
    Error response — matches problem statement spec exactly.
    
    {
        "status": "error",
        "message": "Clear error reason"
    }
    """
    status: str = Field(default="error")
    message: str = Field(..., description="Clear error description")
