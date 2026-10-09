"""Authentication models."""

from pydantic import BaseModel


class User(BaseModel):
    """User model."""

    id: str
    email: str
    name: str
