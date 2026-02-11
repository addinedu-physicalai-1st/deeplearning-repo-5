# Pet Care Robot - 실시간 반려동물 감정 분석 및 행동 지시 시스템

웹캠으로 촬영한 반려동물(개/고양이)의 **포즈 키포인트**와 **영상 프레임**을 멀티모달 딥러닝 모델로 분석하여 실시간 감정 상태를 판별하고, 부정 감정이 지속되면 **VLM/LLM 파이프라인**을 통해 상황 인식 기반 행동 지시를 자동 생성합니다.

## 주요 기능

- **실시간 MJPEG 스트리밍** - 웹캠 영상에 BBox, 15개 키포인트 스켈레톤, 감정 상태를 오버레이하여 브라우저에 전송
- **멀티모달 감정 분류** - EfficientNet-B0(이미지) + ST-GCN(키포인트 시퀀스) + 도메인 피처(50개)를 Gated Fusion으로 결합한 HybridPetNet
- **계층적 분류 전략** - 1단계 긍정/부정 이진분류 후, 부정인 경우 2단계에서 공포·공격성 / 불안·슬픔 / 화남·불쾌로 세분화
- **규칙 기반 후처리** - 꼬리 흔들림, 엎드림, 앉기, 달리기 등의 자세 패턴으로 모델 출력을 실시간 교정
- **GenAI 행동 지시** - 부정 감정 3초 이상 지속 시 SmolVLM2-256M(장면 묘사) + Qwen2.5-3B(지시 생성) 순차 파이프라인 실행, 6GB VRAM 환경에서 온디맨드 로딩/해제
- **실시간 대시보드** - Vue 3 기반 다크 테마 UI에서 영상 피드, 감정 패널, 행동 지시, 로그 콘솔을 한 화면에 제공
- **로그 이력 관리** - 5초 주기 감정 로그 + GenAI 행동 지시를 MySQL에 저장, 페이지네이션 조회 지원

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│  Frontend (Vue 3 + Pinia + Vite)                 :5173             │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌────────────┐           │
│  │VideoFeed │ │Emotion   │ │Action     │ │Log         │           │
│  │(MJPEG)   │ │Panel     │ │Guide      │ │Console     │           │
│  └────┬─────┘ └────┬─────┘ └─────┬─────┘ └─────┬──────┘           │
│       │ <img>      │ polling     │ WS          │ WS              │
├───────┼────────────┼─────────────┼─────────────┼──────────────────┤
│  Backend (FastAPI + Uvicorn)                     :8100             │
│       │            │             │             │                   │
│  ┌────▼────────────▼─────────────▼─────────────▼──────────┐       │
│  │                    PetStreamer                          │       │
│  │  Webcam → YOLO26 Detect → YOLO Pose → Keypoint Buffer │       │
│  │         → HybridPetNet(Binary) → Rule Correction       │       │
│  │         → HybridPetNet(Negative) → Emotion Output      │       │
│  │         → [3s neg] → GenAI Worker (VLM → LLM)          │       │
│  └────────────────────────┬───────────────────────────────┘       │
│                           │ log_queue                              │
│  ┌────────────────────────▼───────────────────────────────┐       │
│  │  Log Consumer → MySQL (emotion_logs, action_guides)    │       │
│  │              → WebSocket Broadcast                      │       │
│  └────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

## 기술 스택

| 영역 | 기술 |
|------|------|
| **AI 추론** | PyTorch, Ultralytics YOLO26, EfficientNet-B0, ST-GCN |
| **GenAI** | HuggingFace Transformers, SmolVLM2-256M, Qwen2.5-3B (4bit) |
| **백엔드** | FastAPI, Uvicorn, SQLAlchemy (async), aiomysql, WebSocket |
| **프론트엔드** | Vue 3, TypeScript, Pinia, Vite |
| **데이터베이스** | MySQL 8.0 |
| **학습 도구** | LDAM Loss, Mixup, 2-Stage Hierarchical Training |

## 디렉토리 구조

```
v3/
├── pet-care-backend/                # FastAPI 백엔드 서버
│   ├── app.py                       # Uvicorn 진입점
│   ├── .env                         # DB/카메라 환경 변수
│   ├── requirements.txt             # Python 의존성
│   ├── yolo26m.pt                   # YOLO26 객체 탐지 모델
│   ├── config/
│   │   ├── config.yaml              # AI 파이프라인 설정 (임계값, 샘플링 등)
│   │   └── pose_config.yaml         # 15개 키포인트 정의 및 스켈레톤 연결
│   └── src/
│       ├── api/                     # FastAPI 라우터
│       │   ├── main.py              # 앱 팩토리 (lifespan, CORS, WS 매니저)
│       │   └── routes/              # video_feed, status, members, logs
│       ├── core/
│       │   ├── streamer.py          # 실시간 추론 파이프라인 (PetStreamer)
│       │   └── vlm_worker.py        # GenAI 파이프라인 (VLM→LLM, 온디맨드 로딩)
│       ├── models/
│       │   ├── multimodal_net.py    # HybridPetNet (ST-GCN + EfficientNet + Gated Fusion)
│       │   └── best.pt             # YOLO Pose 모델 (15 keypoints)
│       ├── ai/
│       │   ├── checkpoints/         # 감정 분류 모델 4개 (cat/dog x binary/negative)
│       │   └── utils/
│       │       └── feature_engineering.py  # 50개 도메인 피처 추출기
│       ├── db/
│       │   ├── database.py          # SQLAlchemy async 엔진
│       │   ├── models.py            # ORM (Member, Pet, EmotionLog, ActionGuide)
│       │   ├── crud.py              # CRUD 함수
│       │   ├── schemas.py           # Pydantic 스키마
│       │   └── seed.py              # 시드 데이터 (회원 3명, 펫 5마리)
│       ├── services/
│       │   ├── visualizer.py        # BBox/키포인트/감정 텍스트 오버레이
│       │   └── log_service.py       # 로그 DB 저장 래퍼
│       └── utils/
│           └── model_loader.py      # 모델 일괄 로딩 (YOLO + Pose + Emotion x4)
│
├── pet-care-front/                  # Vue 3 프론트엔드
│   ├── vite.config.ts               # /api 프록시 → localhost:8100
│   └── src/
│       ├── views/                   # DashboardView, LogHistoryView
│       ├── components/
│       │   ├── layout/AppHeader.vue # 네비게이션 헤더
│       │   └── dashboard/           # VideoFeed, EmotionPanel, ActionGuide,
│       │                            # PetInfoCard, LogConsole
│       ├── stores/                  # Pinia (petStatus, logStore, memberStore)
│       ├── api/index.ts             # REST + WebSocket 클라이언트
│       └── types/index.ts           # TypeScript 인터페이스
│
├── createBasicCode/                 # 모델 학습 파이프라인
│   ├── src/
│   │   ├── 01_analyze_emotion_action_mapping.py   # 감정-행동 매핑 분석
│   │   ├── 02_create_balanced_dataset.py          # 균형 데이터셋 생성
│   │   ├── 03_prepare_balanced_yolo_data.py       # YOLO Pose 학습 데이터 변환
│   │   ├── 04_train_balanced_yolo_pose.py         # YOLO26 Pose 모델 학습
│   │   ├── 05_create_multimodal_dataset.py        # 멀티모달 시퀀스 데이터 생성
│   │   ├── 06_cat_train_hierarchical.py           # 고양이 계층적 감정 분류기 학습
│   │   ├── 06_dog_train_hierarchical.py           # 개 계층적 감정 분류기 학습
│   │   └── 07_test_inference_final.py             # 추론 테스트 (영상 입력)
│   └── utils/
│       ├── feature_engineering.py                  # 피처 추출기 (학습용)
│       └── metrics.py                              # 평가 메트릭
```

## AI 파이프라인

### 감정 분류 모델 (HybridPetNet)

```
                    ┌──────────────────┐
  Video Frame ─────►│ EfficientNet-B0  │──► 1280d
                    └──────────────────┘        ┐
                    ┌──────────────────┐        │   ┌──────────────┐     ┌────────────┐
  Keypoint Seq ────►│ ST-GCN (3-layer) │──► 128d├──►│ Gated Fusion │────►│ Classifier │──► 2 or 3 classes
  (30 frames)       └──────────────────┘        │   │ (Bottleneck) │     │ (MLP+SiLU) │
                    ┌──────────────────┐        │   └──────────────┘     └────────────┘
  Domain Feats ────►│ 1D Conv+Attn     │──► 128d┘
  (50 features)     └──────────────────┘
```

- **3가지 모달리티**를 Gated Fusion으로 동적 가중 결합
- 종별 독립 학습: `cat_binary`, `cat_negative`, `dog_binary`, `dog_negative` 총 4개 체크포인트
- 학습: LDAM Loss + Mixup + 2-Stage (Frozen → Unfrozen) + Weighted Sampling

### 도메인 피처 (50개)

| 카테고리 | 피처 수 | 예시 |
|----------|---------|------|
| 얼굴/머리 | 7 | 머리 높이, 기울기, 입 벌림 정도 |
| 자세/척추 | 7 | 척추 각도, 웅크림 비율, 좌우 대칭도 |
| 앞다리 | 6 | 벌림 정도, 발 높이, 뻗음 거리 |
| 뒷다리 | 6 | 벌림 정도, 움츠림 정도 |
| 꼬리 | 8 | 꼬리 각도, 들림, 말림, 좌우 치우침 |
| 종합 자세 | 5 | 무게중심, 전체 웅크림, 서있는 정도 |
| 시계열 | 11 | 머리/꼬리 속도·가속도, 꼬리 흔들림 |

### GenAI 행동 지시 파이프라인

```
부정 감정 3초 지속
  → SmolVLM2-256M (장면 묘사, ~50 tokens)
  → GPU 메모리 해제
  → Qwen2.5-3B-Instruct (4bit, 행동 지시 JSON 생성)
  → GPU 메모리 해제
  → {"action": "한글 행동 지시", "control": {"cmd": "WAIT", "param": "2000"}}
```

- 6GB VRAM 제약으로 VLM/LLM 순차 로딩 후 즉시 해제
- 트리거 시점 감정/객체 정보를 스냅샷으로 캡처하여 DB 정합성 보장

### 규칙 기반 후처리 (Rule Correction)

| 종 | 자세 패턴 | 교정 |
|----|----------|------|
| 개 | 뛰기 (고속 모션) | 부정 임계값 +0.40 상향 |
| 개 | 플레이 바우 (어깨↓ 엉덩이↑) | 부정 확률 -0.30 |
| 고양이 | 하악질/등 아치 | 부정 확률 +0.35 |
| 고양이 | 꼬리 흔들기 | 부정 확률 +0.25 |
| 고양이 | 냄새 맡기 | 부정 확률 -0.15 |
| 고양이 | 앉기/꼬리 세움 | 부정 확률 -0.30 |

## 사전 요구사항

- Ubuntu 24.04 / Python 3.12
- Node.js 20.19+
- MySQL 8.0+
- NVIDIA GPU (CUDA, VRAM 6GB+)
- 웹캠

## 설치 및 실행

### 1. 의존성 설치

```bash
# 백엔드
cd v3/pet-care-backend
pip install -r requirements.txt

# 프론트엔드
cd v3/pet-care-front
npm install
```

### 2. 데이터베이스 설정

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS pet_care_v3 CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
cd v3/pet-care-backend
python -m src.db.seed
```

### 3. 환경 변수 (`v3/pet-care-backend/.env`)

```env
PORT=8100
DB_HOST=localhost
DB_PORT=3306
DB_USER=USER_NAME
DB_PASSWORD=USER_PASSWORD
DB_NAME=pet_care_v3
CAMERA_INDEX=0
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
```

### 4. 서비스 시작

```bash
# 터미널 1: 백엔드
cd v3/pet-care-backend
python app.py

# 터미널 2: 프론트엔드
cd v3/pet-care-front
npm run dev
```

### 5. 접속

| 페이지 | URL |
|--------|-----|
| 대시보드 | http://localhost:5173 |
| 로그 이력 | http://localhost:5173/logs |
| API 상태 확인 | http://localhost:8100/api/status |

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/api/video_feed` | MJPEG 실시간 스트리밍 |
| GET | `/api/status` | 현재 감정/탐지 상태 JSON |
| GET | `/api/members` | 회원 목록 |
| GET | `/api/pets` | 반려동물 목록 |
| GET | `/api/pets/{id}` | 반려동물 상세 + 최근 로그 |
| GET | `/api/logs?page=1&page_size=20` | 감정 로그 이력 (페이지네이션) |
| WS | `/api/ws/logs` | 실시간 로그/행동지시 브로드캐스트 |

## DB 스키마

```
members (1) ──── (N) pets (1) ──── (N) emotion_logs (1) ──── (1) action_guides
   id                  id                  id                      id
   name                member_id (FK)      pet_id (FK)             log_id (FK)
   email               name                timestamp               emotion_trigger
   phone               species (dog/cat)   object_type             action_text
   created_at          breed               emotion (pos/neg)       control_cmd
                       age                 emotion_detail          control_param
                                           emotion_conf
                                           fps
                                           order_text
```

## 데이터셋 출처

본 프로젝트의 모델 학습에는 **AI 허브(AI Hub)**에서 제공하는 공공 데이터를 활용하였습니다. 학습을 재현하거나 데이터를 확충하려면 아래 링크에서 데이터를 다운로드해야 합니다.

* **데이터셋 명칭**: 반려동물 구분을 위한 동물 영상 데이터
* **제공처**: AI 허브 (AI Hub)
* **다운로드 링크**: [https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=59](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&searchKeyword=%EB%B0%98%EB%A0%A4%EB%8F%99%EB%AC%BC%20%EA%B5%AC%EB%B6%84%EC%9D%84%20%EC%9C%84%ED%95%9C%20%EB%8F%99%EB%AC%BC%20%EC%98%81%EC%83%81&aihubDataSe=data&dataSetSn=59)
* **참고**: 해당 데이터셋의 JSON 라벨을 `v3/createBasicCode`의 학습 포맷으로 변환하는 과정이 필요합니다.

## 학습 파이프라인 (createBasicCode)

데이터셋에서 서비스용 모델까지 7단계 순차 실행:

```
01. 감정-행동 매핑 분석 (데이터 품질 검증)
02. 균형 데이터셋 생성 (긍정 1000 / 부정 5000 per action)
03. YOLO Pose 학습 데이터 변환 (sparse sampling, ~40K images)
04. YOLO26-Pose 모델 학습 (15 keypoints, RTX 4060 기준)
05. 멀티모달 시퀀스 데이터셋 생성 (PKL)
06. 계층적 감정 분류기 학습 (cat/dog 독립, 2-stage)
07. 추론 테스트 (영상 입력 → 감정 + 행동 출력)
```
