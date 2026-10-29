"""Local, normalized dense BGE-M3 embeddings shared by indexing and retrieval."""

import asyncio
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class BgeM3Embedder:
    def __init__(self, model_name: str, cache_dir: str) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: SentenceTransformer | None = None
        self._lock = Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _encode(self, texts: list[str]) -> list[list[float]]:
        # Loading and encoding stay off the event loop, including on the first request.
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self._model_name, cache_folder=self._cache_dir, device="cpu"
                )
            embeddings = self._model.encode(
                texts, normalize_embeddings=True, batch_size=8, show_progress_bar=False
            )
            result = [[float(value) for value in vector] for vector in embeddings]
        if any(len(vector) != 1024 for vector in result):
            raise ValueError("BGE-M3 embeddings must have 1024 dimensions")
        return result

    async def encode(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode, texts)
