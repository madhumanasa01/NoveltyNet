"""Dev entry point — runs the NOVELTYNET app with hot reload."""

import os

# Force offline mode for the embedding model. Without these, startup hangs
# 60+ seconds retrying HEAD requests to huggingface.co to check the cached
# MiniLM model for updates.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
