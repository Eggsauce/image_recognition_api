from fastapi import FastAPI
from dotenv import load_dotenv
from routers.detection import router as detection_router
import uvicorn
import os

load_dotenv()

app = FastAPI(
    title="YOLO Coordinate API",
    description="API for handling YOLO v11 coordinates and annotations",
    version="1.0.0"
)

app.include_router(detection_router)


def main():
    workers = int(os.getenv("WORKERS", os.cpu_count() or 1))
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=workers)


if __name__ == "__main__":
    main()
