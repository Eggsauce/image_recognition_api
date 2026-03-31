from fastapi import FastAPI
from dotenv import load_dotenv
from routers.detection import router as detection_router
import uvicorn

load_dotenv()

app = FastAPI(
    title="YOLO Coordinate API",
    description="API for handling YOLO v11 coordinates and annotations",
    version="1.0.0"
)

app.include_router(detection_router)


def main():
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
