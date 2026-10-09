"""API dependencies."""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

from app.core.models.auth import User

security = HTTPBearer()


async def get_current_user(token: Annotated[str, Depends(security)]) -> User:
    """Get current authenticated user."""
    # TODO: Implement actual authentication logic
    # For now, return a mock user
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )

    return User(id="mock-user-id", email="user@example.com", name="Mock User")
