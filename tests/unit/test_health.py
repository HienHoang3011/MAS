import asyncio

from httpx import ASGITransport, AsyncClient

from app.main import create_app


def test_health() -> None:
    async def check() -> None:
        async with AsyncClient(
            transport=ASGITransport(app=create_app()),
            base_url="http://test",
            follow_redirects=True,
        ) as client:
            response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    asyncio.run(check())
