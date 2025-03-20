import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class APIResponse(BaseModel):
    """Base API response model"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class APIRequest(BaseModel):
    """Base API request model"""
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    operation: Optional[str] = None
    api_key: Optional[str] = None

    @validator('request_id', 'user_id', 'session_id', pre=True, always=True)
    def set_id_if_none(cls, v, values, **kwargs):
        return v or str(uuid.uuid4())


class ComponentAPIRequest(APIRequest):
    """API request for component operations"""
    component_type: str
    operation: str
    config: Optional[Dict[str, Any]] = None
    data: Optional[Any] = None


class ComponentAPIResponse(APIResponse):
    """API response for component operations"""
    component_type: Optional[str] = None
    operation: Optional[str] = None


class PipelineConfig(BaseModel):
    """Configuration for the audio pipeline"""
    name: str
    log_level: str = "INFO"
    log_format: str = "console"
    components: List[Any]
    api_enabled: bool = True
    api_auth_required: bool = False
    api_key: Optional[str] = None