"""AI use cases."""

from app.api.routers.ai import ChatResponse, EmbeddingResponse


class AIUseCases:
    """AI use cases."""

    async def generate_chat_response(
        self, message: str, context: str | None = None, user_id: str | None = None
    ) -> ChatResponse:
        """Generate chat response."""
        # TODO: Implement actual AI logic
        return ChatResponse(response=f"Mock response to: {message}", tokens_used=10)

    async def generate_embedding(self, text: str, user_id: str | None = None) -> EmbeddingResponse:
        """Generate text embedding."""
        # TODO: Implement actual embedding logic
        mock_embedding = [0.1] * 1536  # Mock OpenAI embedding size
        return EmbeddingResponse(embedding=mock_embedding, dimensions=len(mock_embedding))
