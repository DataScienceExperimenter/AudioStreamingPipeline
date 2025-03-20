import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel

from src.api.models import ComponentAPIRequest, ComponentAPIResponse
from src.utils.logger import logger


class ComponentConfig(BaseModel):
    """Base configuration for pipeline components"""
    name: str
    log_level: str = "INFO"
    api_enabled: bool = True

    class Config:
        extra = "allow"  # Allow extra fields for component-specific configs


class Component(ABC):
    """Base class for pipeline components"""

    def __init__(self, config: ComponentConfig):
        self.name = config.name
        self.log_level = config.log_level
        self.api_enabled = config.api_enabled
        self.initialized = False
        self._config = config

    async def initialize(self) -> None:
        """Initialize the component"""
        self.initialized = True

    @abstractmethod
    async def process(self, data: Any) -> Any:
        """Process data"""
        pass

    async def shutdown(self) -> None:
        """Clean up resources"""
        self.initialized = False

    async def handle_api_request(self, request: ComponentAPIRequest) -> ComponentAPIResponse:
        """Handle API requests to this component

        Args:
            request: The API request

        Returns:
            API response
        """
        if not self.api_enabled:
            return ComponentAPIResponse(
                success=False,
                message="API access is disabled for this component",
                component_type=self.__class__.__name__,
                operation=request.operation
            )

        try:
            # Handle different operations
            if request.operation == "status":
                return ComponentAPIResponse(
                    success=True,
                    message="Component status retrieved",
                    component_type=self.__class__.__name__,
                    operation=request.operation,
                    data={"initialized": self.initialized, "name": self.name}
                )
            elif request.operation == "process" and request.data:
                result = await self.process(request.data)
                return ComponentAPIResponse(
                    success=True,
                    message="Data processed successfully",
                    component_type=self.__class__.__name__,
                    operation=request.operation,
                    data={"result": result}
                )
            else:
                return ComponentAPIResponse(
                    success=False,
                    message=f"Unknown operation: {request.operation}",
                    component_type=self.__class__.__name__,
                    operation=request.operation
                )
        except Exception as e:
            logger.error(f"Error in API request: {str(e)}")
            return ComponentAPIResponse(
                success=False,
                message="Error processing API request",
                component_type=self.__class__.__name__,
                operation=request.operation,
                error=str(e)
            )