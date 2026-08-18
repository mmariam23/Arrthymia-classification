"""
main.py — Entry point
Run locally  :  python main.py
Hugging Face :  started automatically via Dockerfile CMD
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,        # HF Spaces requires 7860
        reload=False,
        log_level="info",
    )
