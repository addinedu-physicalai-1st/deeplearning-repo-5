"""FastAPI 앱 - lifespan, CORS, WebSocket 매니저, 백그라운드 초기화 (v3 신규 작성)"""
import os
import sys
import asyncio
import queue
import json
import time
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.model_loader import ModelLoader
from src.services.visualizer import Visualizer
from src.core.streamer import PetStreamer
from src.db.database import init_db, close_db, async_session
from src.services.log_service import save_emotion_log, save_action_guide
from src.api.routes import stream, status, members, logs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# /status 엔드포인트 로그 필터
class StatusFilter(logging.Filter):
    def filter(self, record):
        return "/status" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(StatusFilter())


# ── 전역 상태 ──
class AppState:
    streamer: PetStreamer = None
    log_queue: queue.Queue = queue.Queue(maxsize=200)
    ready: bool = False
    ws_manager: "ConnectionManager" = None

app_state = AppState()


class ConnectionManager:
    """WebSocket 연결 관리"""

    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logger.info(f"WebSocket 연결 ({len(self.active)}개)")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        logger.info(f"WebSocket 해제 ({len(self.active)}개)")

    async def broadcast(self, data: dict):
        msg = json.dumps(data, ensure_ascii=False)
        closed = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                closed.append(ws)
        for ws in closed:
            self.disconnect(ws)


# ── 백그라운드 로그 소비자 ──
async def log_consumer():
    """큐에서 로그를 꺼내 DB 저장 + WebSocket 브로드캐스트"""
    while True:
        try:
            log_data = await asyncio.get_event_loop().run_in_executor(
                None, lambda: app_state.log_queue.get(timeout=1.0)
            )
        except queue.Empty:
            await asyncio.sleep(0.1)
            continue

        msg_type = log_data.get("type", "log")

        # WebSocket 브로드캐스트
        if app_state.ws_manager:
            await app_state.ws_manager.broadcast(log_data)

        # DB 저장
        try:
            async with async_session() as session:
                if msg_type == "log":
                    await save_emotion_log(session, log_data)
                elif msg_type == "action_guide":
                    await save_action_guide(session, None, log_data)
        except Exception as e:
            logger.error(f"로그 DB 저장 실패: {e}")


# ── 모델 초기화 (백그라운드 스레드) ──
def initialize_models():
    """동기 모델 로딩 — 별도 스레드에서 실행"""
    try:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
        pose_config_path = PROJECT_ROOT / "config" / "pose_config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        loader = ModelLoader(config)
        loader.load_all_models()

        visualizer = Visualizer(str(pose_config_path))
        streamer = PetStreamer(loader, visualizer, app_state.log_queue)

        # 기본 활성 반려동물 설정 (시드 데이터 첫 번째)
        streamer.set_active_pet("김하준", "바둑이", "진돗개")

        app_state.streamer = streamer
        app_state.ready = True
        logger.info("모든 서비스 초기화 완료")
    except Exception as e:
        logger.error(f"초기화 실패: {e}", exc_info=True)


# ── Lifespan ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    # DB 테이블 생성
    await init_db()
    logger.info("DB 초기화 완료")

    # WebSocket 매니저
    app_state.ws_manager = ConnectionManager()

    # 모델 로딩 (백그라운드 스레드)
    threading.Thread(target=initialize_models, daemon=True).start()

    # 로그 소비자 태스크
    consumer_task = asyncio.create_task(log_consumer())

    yield

    # 종료
    consumer_task.cancel()
    if app_state.streamer:
        app_state.streamer.release()
    await close_db()


# ── App ──
app = FastAPI(title="Pet Care V3 API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(stream.router, prefix="/api")
app.include_router(status.router, prefix="/api")
app.include_router(members.router, prefix="/api")
app.include_router(logs.router, prefix="/api")


# WebSocket 엔드포인트
@app.websocket("/api/ws/logs")
async def ws_logs(ws: WebSocket):
    await app_state.ws_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        app_state.ws_manager.disconnect(ws)


def get_app_state() -> AppState:
    return app_state
