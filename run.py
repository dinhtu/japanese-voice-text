"""Run the pronunciation API + practice page:  python run.py

Host, port and the trusted proxy IPs come from .env (see .env.example).
"""

from app.core.config import get_settings
from app.main import app

settings = get_settings()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
        # Needed so https:// and the public host survive the reverse proxy.
        proxy_headers=True,
        forwarded_allow_ips=settings.forwarded_allow_ips,
    )
