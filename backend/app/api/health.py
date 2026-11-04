from __future__ import annotations

from fastapi import APIRouter, Response, status

from app import __version__
from app.api.deps import ContainerDep

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(c: ContainerDep, response: Response) -> dict[str, object]:
    """Unauthenticated; the cron that keeps the free host warm calls this."""
    db_ok = await c.store.ping()
    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ok" if db_ok else "degraded",
        "database": db_ok,
        "tracing": c.tracer.enabled,
        "version": __version__,
    }
