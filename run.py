"""Run the pronunciation API + practice page:  python run.py

Equivalent to:  uvicorn app.main:app --host 0.0.0.0 --port 8000
Override with the APP_HOST / APP_PORT environment variables.
"""

import os

from app.main import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "8000")),
        reload=False,
    )
