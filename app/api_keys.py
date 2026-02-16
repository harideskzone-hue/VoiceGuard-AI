"""
API Key Management Module
=========================
Provides API key generation, validation, and listing for external users.
Keys are stored in-memory + persisted to a JSON file for hackathon demo.
"""

import os
import json
import uuid
import hashlib
import secrets
import logging
from datetime import datetime
from typing import Optional, Dict, List

logger = logging.getLogger("api_keys")

# Storage file
KEYS_FILE = os.path.join(os.path.dirname(__file__), "..", "api_keys.json")

# In-memory store
_api_keys: Dict[str, dict] = {}

# Default master key from .env
MASTER_KEY = os.getenv("API_KEY", "default_secret_key_for_testing_only")


def _load_keys():
    """Load keys from JSON file on startup."""
    global _api_keys
    try:
        if os.path.exists(KEYS_FILE):
            with open(KEYS_FILE, 'r') as f:
                _api_keys = json.load(f)
            logger.info(f"Loaded {len(_api_keys)} API keys from storage")
    except Exception as e:
        logger.error(f"Failed to load keys: {e}")
        _api_keys = {}


def _save_keys():
    """Persist keys to JSON file."""
    try:
        with open(KEYS_FILE, 'w') as f:
            json.dump(_api_keys, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save keys: {e}")


def generate_api_key(name: str, created_by: str = "dashboard") -> dict:
    """
    Generate a new API key for an external user.
    
    Returns dict with key details including the raw key (shown only once).
    """
    # Generate a secure random key
    raw_key = f"vg_{secrets.token_hex(16)}"
    key_id = str(uuid.uuid4())[:8]
    
    key_record = {
        "id": key_id,
        "name": name,
        "key": raw_key,
        "key_preview": raw_key[:7] + "..." + raw_key[-4:],
        "created_at": datetime.now().isoformat(),
        "created_by": created_by,
        "active": True,
        "usage_count": 0
    }
    
    _api_keys[raw_key] = key_record
    _save_keys()
    
    logger.info(f"Generated API key '{name}' (id: {key_id})")
    return key_record


def validate_key(api_key: str) -> bool:
    """Check if an API key is valid (either master key or generated key)."""
    if api_key == MASTER_KEY:
        return True
    
    if api_key in _api_keys and _api_keys[api_key].get("active", False):
        # Increment usage counter
        _api_keys[api_key]["usage_count"] = _api_keys[api_key].get("usage_count", 0) + 1
        _save_keys()
        return True
    
    return False


def list_keys() -> List[dict]:
    """List all generated API keys (without exposing full key values)."""
    result = []
    for key, record in _api_keys.items():
        result.append({
            "id": record.get("id"),
            "name": record.get("name"),
            "key_preview": record.get("key_preview"),
            "created_at": record.get("created_at"),
            "active": record.get("active", True),
            "usage_count": record.get("usage_count", 0)
        })
    return result


def revoke_key(key_id: str) -> bool:
    """Revoke an API key by its ID."""
    for key, record in _api_keys.items():
        if record.get("id") == key_id:
            record["active"] = False
            _save_keys()
            logger.info(f"Revoked API key: {key_id}")
            return True
    return False


def get_stats() -> dict:
    """Get API key statistics."""
    total = len(_api_keys)
    active = sum(1 for r in _api_keys.values() if r.get("active", False))
    total_usage = sum(r.get("usage_count", 0) for r in _api_keys.values())
    return {
        "total_keys": total,
        "active_keys": active,
        "revoked_keys": total - active,
        "total_api_calls": total_usage
    }


# Load keys on module import
_load_keys()
