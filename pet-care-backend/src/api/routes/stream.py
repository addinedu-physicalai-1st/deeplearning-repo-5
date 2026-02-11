"""GET /api/video_feed - MJPEG 실시간 스트리밍"""
import time
import numpy as np
import cv2
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter()


def _loading_generator():
    """모델 로딩 중 대기 화면"""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(frame, "System Initializing... Please Wait.",
                (350, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    _, buf = cv2.imencode(".jpg", frame)
    data = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
    while True:
        yield data
        time.sleep(0.5)


@router.get("/video_feed")
async def video_feed():
    from src.api.main import get_app_state
    state = get_app_state()

    if not state.ready or not state.streamer:
        return StreamingResponse(
            _loading_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
    return StreamingResponse(
        state.streamer.generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
