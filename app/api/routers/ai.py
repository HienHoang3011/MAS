"""AI-related API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.api.dependencies import get_current_user
from app.core.models.auth import User
from app.core.use_cases.ai_use_cases import AIUseCases

router = APIRouter(prefix="/ai", tags=["ai"])


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str
    context: str | None = None


class ChatResponse(BaseModel):
    """Chat response model."""

    response: str
    tokens_used: int


class EmbeddingRequest(BaseModel):
    """Embedding request model."""

    text: str


class EmbeddingResponse(BaseModel):
    """Embedding response model."""

    embedding: list[float]
    dimensions: int


@router.post("/chat", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    current_user: Annotated[User, Depends(get_current_user)],
) -> ChatResponse:
    """Generate AI chat completion."""
    try:
        # TODO: Implement actual AI use case
        use_cases = AIUseCases()
        response = await use_cases.generate_chat_response(
            message=request.message,
            context=request.context,
            user_id=current_user.id,
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI processing failed: {str(e)}",
        ) from e


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embedding(
    request: EmbeddingRequest,
    current_user: Annotated[User, Depends(get_current_user)],
) -> EmbeddingResponse:
    """Generate text embeddings."""
    try:
        # TODO: Implement actual embedding use case
        use_cases = AIUseCases()
        response = await use_cases.generate_embedding(
            text=request.text,
            user_id=current_user.id,
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}",
        ) from e
