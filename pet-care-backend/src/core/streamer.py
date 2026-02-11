"""
PetStreamer - 실시간 반려동물 감정 추론 파이프라인 (비동기 스냅샷 & 룰 교정 패치)

[적용된 핵심 패치]
1. 객체 소실(None) 독립적인 글로벌 Lock (is_genai_busy) 적용하여 5초 타이머 완벽 동기화
2. 트리거 발동 시점 스냅샷(Snapshot) 기반으로 DB/UI 동시 로그 발송
3. 다음번 5초 주기가 오면 화면에 표시된 행동 지시 자동 초기화 (clear_order_next)
4. 고양이 평화 상태 오탐 방지를 위한 Threshold 복구 (0.55) 및 패널티 안정화 (+0.35)
"""

import cv2
import time
import queue
import numpy as np
import torch
import logging
from collections import deque
from torchvision import transforms
from PIL import Image

from src.ai.utils.feature_engineering import PetFeatureExtractor
from src.core.vlm_worker import GenAIWorker

logger = logging.getLogger(__name__)

COCO_DOG_ID = 16
COCO_CAT_ID = 15

NEGATIVE_LABELS = {0: "공포/공격성", 1: "불안/슬픔", 2: "화남/불쾌"}

IMG_SIZE = 224
transform_pipeline = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class TrackState:
    def __init__(self, species: str):
        self.species = species
        self.kp_buffer: deque = deque(maxlen=30)
        self.kp_times: deque = deque(maxlen=30)
        self.emo_buffer: deque = deque(maxlen=20)
        self.species_votes: deque = deque(maxlen=30)

        self.last_seen: float = time.time()
        self.display_emotion: str = ""
        self.display_conf: float = 0.0
        self.last_debug: str = ""

        self.neg_start_time = None
        self.analyzing_situation = False
        self.last_action_guide = None


class PetStreamer:
    def __init__(self, model_loader, visualizer, log_queue: queue.Queue):
        self.model_loader = model_loader
        self.visualizer = visualizer
        self.log_queue = log_queue

        self.cap = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = PetFeatureExtractor(fps=5.0)
        self.genai_worker = GenAIWorker()

        self._primary_state: TrackState | None = None
        self._primary_lost_time: float = 0.0

        # 🚀 [추가] 객체 소실과 무관하게 유지되는 글로벌 상태 변수들
        self._global_order = "none"
        self.clear_order_next = False
        self.is_genai_busy = False

        self.status = {
            "fps": 0,
            "detected": False,
            "object_type": "none",
            "emotion": "대기중",
            "emotion_detail": None,
            "emotion_conf": 0.0,
            "order": "none",
        }
        self._last_log_time = time.time()
        self._active_pet_info = {"member_name": "", "pet_name": "", "breed": ""}

    def set_active_pet(self, member_name: str, pet_name: str, breed: str):
        self._active_pet_info = {
            "member_name": member_name,
            "pet_name": pet_name,
            "breed": breed,
        }

    def get_status(self) -> dict:
        return {**self.status, **self._active_pet_info}

    def start_camera(self, index: int = 0, width: int = 1280, height: int = 720):
        if self.cap is not None and self.cap.isOpened():
            return
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -4)

        logger.info(f"카메라 시작: index={index} {width}x{height}")

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def generate(self):
        while not self.model_loader.is_loaded:
            frame = self._loading_frame()
            yield self._encode(frame)
            time.sleep(0.2)

        self.start_camera()
        prev_time = time.time()

        while True:
            if not self.cap or not self.cap.isOpened():
                time.sleep(1)
                self.start_camera()
                continue

            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            self.status["fps"] = int(fps)

            vis = self._process_frame(frame, now)

            # 🚨 [수정] 객체 유무와 무관한 글로벌 Lock으로 5초 타이머 제어
            if (now - self._last_log_time >= 5.0) and not self.is_genai_busy:
                self._emit_log()
                self._last_log_time = now

            yield self._encode(vis)

    def _process_frame(self, frame: np.ndarray, now: float) -> np.ndarray:
        vis = frame.copy()
        h, w = frame.shape[:2]

        det_results = None
        if self.model_loader.yolo_model:
            names = self.model_loader.yolo_model.names
            dog_id = next(
                (k for k, v in names.items() if "dog" in v.lower()), COCO_DOG_ID
            )
            cat_id = next(
                (k for k, v in names.items() if "cat" in v.lower()), COCO_CAT_ID
            )

            det_results = self.model_loader.yolo_model.track(
                frame,
                persist=True,
                conf=0.4,
                iou=0.5,
                classes=[cat_id, dog_id],
                tracker="botsort.yaml",
                verbose=False,
                retina_masks=False,
            )[0]

        pose_results = None
        if self.model_loader.pose_model:
            pose_results = self.model_loader.pose_model(frame, verbose=False, conf=0.3)[
                0
            ]

        target_species, final_bbox, max_conf = None, None, 0.0

        if det_results and det_results.boxes is not None:
            for box in det_results.boxes:
                cls_id = int(box.cls.item())
                conf = box.conf.item()
                sp = (
                    "dog" if cls_id == dog_id else ("cat" if cls_id == cat_id else None)
                )
                if sp and conf > max_conf:
                    max_conf = conf
                    target_species = sp
                    final_bbox = box.xyxy[0].cpu().numpy()

        final_kps = None
        if (
            final_bbox is not None
            and pose_results
            and pose_results.keypoints is not None
        ):
            if len(pose_results.keypoints.data) > 0:
                poses = pose_results.keypoints.data.cpu().numpy()
                pose_boxes = pose_results.boxes.xyxy.cpu().numpy()
                best_iou = 0.0

                for i, p_box in enumerate(pose_boxes):
                    iou = self._iou(final_bbox, p_box)
                    if iou > best_iou:
                        best_iou = iou
                        final_kps = poses[i]

                if best_iou < 0.3:
                    final_kps = None

        if final_bbox is not None and target_species:
            state = self._get_or_create_state(target_species, now)
            state.last_seen = now

            state.species_votes.append(target_species)
            dog_votes = sum(1 for v in state.species_votes if v == "dog")
            cat_votes = len(state.species_votes) - dog_votes
            species = "dog" if dog_votes >= cat_votes else "cat"

            if final_kps is not None and len(final_kps) >= 15:
                norm_kps = final_kps[:, :2].copy()
                norm_kps[:, 0] /= float(w)
                norm_kps[:, 1] /= float(h)
                state.kp_buffer.append(norm_kps)
                state.kp_times.append(now)

            if len(state.kp_buffer) >= 30:
                self._run_inference(frame, state, species, final_kps, final_bbox, now)

            vis = self.visualizer.draw(
                vis, final_bbox, final_kps, species, max_conf, state
            )

            self.status.update(
                {
                    "detected": True,
                    "object_type": species,
                    "emotion": state.display_emotion or "분석중",
                    "emotion_conf": round(state.display_conf, 2),
                    "order": self._global_order,
                }
            )
        else:
            if self._primary_state is not None:
                if self._primary_lost_time == 0.0:
                    self._primary_lost_time = now
                elif (now - self._primary_lost_time) > 3.0:
                    self._primary_state = None
                    self._primary_lost_time = 0.0
                    # 🚨 [수정] 객체가 사라져도 함부로 화면 텍스트를 초기화하지 않고 유지
                    self.status.update(
                        {
                            "detected": False,
                            "object_type": "none",
                            "emotion": "대기중",
                            "emotion_detail": None,
                            "emotion_conf": 0.0,
                            "order": self._global_order,
                        }
                    )
            else:
                self.status.update(
                    {
                        "detected": False,
                        "object_type": "none",
                        "emotion": "대기중",
                        "emotion_detail": None,
                        "emotion_conf": 0.0,
                        "order": self._global_order,
                    }
                )

        return vis

    def _get_or_create_state(self, species: str, now: float) -> TrackState:
        self._primary_lost_time = 0.0
        if self._primary_state is None:
            self._primary_state = TrackState(species)
        return self._primary_state

    def _run_inference(
        self,
        frame: np.ndarray,
        state: TrackState,
        species: str,
        final_kps: np.ndarray,
        final_bbox: np.ndarray,
        now: float,
    ):
        if final_kps is None or len(final_kps) < 15:
            return

        seq = np.array(state.kp_buffer)
        feats = self.extractor.extract(seq)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_t = transform_pipeline(frame_rgb).unsqueeze(0).to(self.device).float()
        kpt_t = torch.from_numpy(seq).float().unsqueeze(0).to(self.device)
        feat_t = torch.from_numpy(feats).float().unsqueeze(0).to(self.device)

        models = self.model_loader.emotion_models
        m_bin = models.get(f"{species}_bin")
        if not m_bin:
            return

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            prob = torch.softmax(m_bin(img_t, kpt_t, feat_t), dim=1)
            neg_prob = prob[0][0].item()

        state.emo_buffer.append(neg_prob)
        smoothed = sum(state.emo_buffer) / len(state.emo_buffer)

        # 🚀 [수정] 고양이 오탐지(False Positive) 방지를 위해 임계값을 0.55로 복구
        base_threshold = 0.55 if species == "cat" else 0.40

        motion_score = 0.0
        target_time = now - 0.2
        times = list(state.kp_times)
        kps_list = list(state.kp_buffer)
        if len(times) >= 2:
            prev_idx = -1
            for i in range(len(times) - 2, -1, -1):
                if times[i] <= target_time:
                    prev_idx = i
                    break
            if prev_idx >= 0:
                curr_kps = kps_list[-1][:, :2]
                prev_kps = kps_list[prev_idx][:, :2]
                motion_score = float(np.linalg.norm(curr_kps - prev_kps, axis=1).mean())

        behavior_debug = ""
        bbox_h = max(final_bbox[3] - final_bbox[1], 1.0)
        bbox_w = max(final_bbox[2] - final_bbox[0], 1.0)

        nose_y = (final_kps[0][1] - final_bbox[1]) / bbox_h
        shoulder_y = (
            (final_kps[5][1] + final_kps[6][1]) / 2.0 - final_bbox[1]
        ) / bbox_h
        hip_y = ((final_kps[9][1] + final_kps[10][1]) / 2.0 - final_bbox[1]) / bbox_h
        tail_start_y = (final_kps[13][1] - final_bbox[1]) / bbox_h
        tail_end_y = (final_kps[14][1] - final_bbox[1]) / bbox_h

        if species == "dog":
            is_high_motion = motion_score > 0.025
            is_uncertain_neg = smoothed <= 0.80
            if is_high_motion and is_uncertain_neg:
                base_threshold += 0.40
                behavior_debug = "Run"

            play_bow_score = shoulder_y - hip_y
            if play_bow_score > 0.05:
                smoothed -= 0.30
                if smoothed < 0:
                    smoothed = 0.0
                behavior_debug = "Bow"

        elif species == "cat":
            # 🚀 밥 먹는 자세 / 바닥 깊은 탐색 (코가 BBox 맨 밑바닥에 위치)
            is_eating = nose_y > 0.80

            # 공격 웅크림 기준을 조금 더 엄격하게 (0.05 -> 0.10) 조정
            is_head_low = (nose_y - shoulder_y) > 0.10
            is_hip_high = (shoulder_y - hip_y) > 0.10
            tail_up_score = tail_start_y - tail_end_y
            is_tail_rigid_up = tail_up_score > 0.10

            # 식사 중이 아닐 때만 공격/경계 자세로 인정
            is_arched_back = (
                (shoulder_y > 0.50)
                and (hip_y > 0.50)
                and (nose_y > 0.50)
                and (not is_tail_rigid_up)
                and not is_eating
            )
            is_sniffing = is_head_low and is_tail_rigid_up
            is_aggressive_crouch = (
                (is_head_low and is_hip_high)
                and (not is_tail_rigid_up)
                and not is_eating
            )

            # 🚨 [수정] 이전 코드에서 지워졌던 꼬리 세움 변수 복구!
            is_tail_up = is_tail_rigid_up and (not is_aggressive_crouch)

            wag_score = 0.0
            if len(state.kp_buffer) >= 10:
                recent_kps = list(state.kp_buffer)[-10:]
                tail_x_norm = [k[14][0] for k in recent_kps]
                wag_score = float(np.std(tail_x_norm))
            is_wagging = wag_score > 0.08

            is_sitting = (hip_y > 0.70) and ((hip_y - shoulder_y) > 0.15)

            # 🚀 종합 판단 (Safety Lock 적용)
            if is_eating:
                # 밥 먹거나 바닥 냄새 맡을 때는 평온함 보장 (-0.30)
                smoothed -= 0.30
                behavior_debug = "Eating/DeepSniff"
            elif is_aggressive_crouch or is_arched_back:
                # 🚨 AI 모델이 이미 어느 정도 의심(0.30 이상)하고 있을 때만 강제 룰 개입
                if smoothed > 0.30:
                    smoothed += 0.35
                    behavior_debug = "Crouch/Arch"
                else:
                    # AI가 평화롭다고 봤다면 단순 걷기/휴식으로 판정
                    behavior_debug = "Walking/Resting"
            elif is_wagging:
                smoothed += 0.25
                behavior_debug = f"Wag({wag_score:.2f})"
            elif is_sniffing:
                smoothed -= 0.15
                behavior_debug = "Sniff"
            elif is_sitting or is_tail_up:
                if smoothed > 0.70:
                    smoothed -= 0.15
                    behavior_debug = "Relaxing"
                else:
                    smoothed -= 0.30
                    tag = []
                    if is_sitting:
                        tag.append("Sit")
                    if is_tail_up:
                        tag.append("TailUp")
                    behavior_debug = "+".join(tag)

        smoothed = max(0.0, min(1.0, smoothed))

        if smoothed > base_threshold:
            current_emotion = "부정"
            current_conf = smoothed
            m_neg = models.get(f"{species}_neg")
            if m_neg:
                with torch.no_grad():
                    np_ = torch.softmax(m_neg(img_t, kpt_t, feat_t), dim=1)
                    n_pred = torch.argmax(np_, dim=1).item()
                    current_emotion = NEGATIVE_LABELS.get(n_pred, "부정")
        else:
            current_emotion = "긍정"
            current_conf = 1.0 - smoothed

        state.last_debug = f"Neg:{smoothed:.2f} Mot:{motion_score:.3f} {behavior_debug}"
        state.display_emotion = current_emotion
        state.display_conf = current_conf

        # ── LLM 트리거 검사 ──
        is_neg = current_emotion != "긍정"
        if is_neg:
            if state.neg_start_time is None:
                state.neg_start_time = time.time()
            if (time.time() - state.neg_start_time) >= 3.0:
                if not self.is_genai_busy:
                    state.analyzing_situation = True
                    self.is_genai_busy = True  # 🚨 시스템 전체 로그 정지 Lock 발동!

                    # 🚨 [스냅샷] 분석 시작 시점의 과거 상태를 영구 보존
                    trigger_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
                    trigger_species = species
                    trigger_emotion = current_emotion
                    trigger_conf = current_conf

                    def cb(res):
                        action = (
                            res.get("action", "") if isinstance(res, dict) else str(res)
                        )
                        if "." in action:
                            action = action.split(".")[0].strip() + "."

                        control = res.get("control") if isinstance(res, dict) else None

                        state.last_action_guide = action
                        self._global_order = action
                        self.status["order"] = action

                        # 🚨 [DB 로그] 발동 시점 스냅샷으로 DB 저장
                        self.log_queue.put_nowait(
                            {
                                "type": "action_guide",
                                "timestamp": trigger_time_str,
                                "object": trigger_species,
                                "emotion_trigger": trigger_emotion,
                                "action": action,
                                "control": control,
                            }
                        )

                        # 🚨 [UI 로그] 일반 로그 형태를 빌려 화면 좌측 하단 콘솔에 강제 출력
                        emotion_map = {
                            "긍정": "positive",
                            "부정": "negative",
                            "공포/공격성": "negative",
                            "불안/슬픔": "negative",
                            "화남/불쾌": "negative",
                        }
                        emotion_en = emotion_map.get(trigger_emotion, "none")

                        self.log_queue.put_nowait(
                            {
                                "type": "log",
                                "timestamp": trigger_time_str,
                                "object": trigger_species,
                                "emotion": emotion_en,
                                "emotion_detail": trigger_emotion,
                                "emotion_conf": trigger_conf,
                                "member_name": self._active_pet_info.get(
                                    "member_name", ""
                                ),
                                "pet_name": self._active_pet_info.get("pet_name", ""),
                                "breed": self._active_pet_info.get("breed", ""),
                                "order": action,
                                "fps": self.status.get("fps", 0),
                            }
                        )

                        # 🚨 [동기화] 메세지 출력 완료 '지금 이 순간'부터 5초 뒤에 다음 로그 예약
                        self._last_log_time = time.time()
                        self.clear_order_next = True

                        # 🚨 [핵심 수정] 분석이 끝났으니, 지금까지 쌓인 분노 시간을 초기화합니다!
                        # 그래야 바로 또 트리거가 걸리지 않고, '새롭게' 3초를 다시 셉니다.
                        state.neg_start_time = None

                        # Lock 해제
                        if self._primary_state:
                            self._primary_state.analyzing_situation = False
                        self.is_genai_busy = False

                    self.genai_worker.run_async(frame.copy(), current_emotion, cb)
        else:
            state.neg_start_time = None
            self._global_order = "none"
            self.status["order"] = "none"
            state.last_action_guide = None
            self.clear_order_next = False

    def _emit_log(self):
        # 🚨 [초기화] 다음 5초 로그 발송 시 화면에 떠 있던 이전 행동 지시 완벽 삭제
        if self.clear_order_next:
            self._global_order = "none"
            self.status["order"] = "none"
            if self._primary_state:
                self._primary_state.last_action_guide = None
            self.clear_order_next = False

        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        emotion = self.status.get("emotion", "none")

        emotion_map = {
            "긍정": "positive",
            "부정": "negative",
            "공포/공격성": "negative",
            "불안/슬픔": "negative",
            "화남/불쾌": "negative",
        }
        emotion_en = emotion_map.get(emotion, "none")

        log_data = {
            "type": "log",
            "timestamp": now_str,
            "object": self.status.get("object_type", "none"),
            "emotion": emotion_en,
            "emotion_detail": (
                emotion
                if emotion_en == "negative"
                else ("긍정" if emotion_en == "positive" else "none")
            ),
            "emotion_conf": self.status.get("emotion_conf", 0.0),
            "member_name": self._active_pet_info.get("member_name", ""),
            "pet_name": self._active_pet_info.get("pet_name", ""),
            "breed": self._active_pet_info.get("breed", ""),
            "order": self.status.get("order", "none"),
            "fps": self.status.get("fps", 0),
        }
        try:
            self.log_queue.put_nowait(log_data)
        except queue.Full:
            pass

    @staticmethod
    def _iou(box1, box2) -> float:
        xa, ya = max(box1[0], box2[0]), max(box1[1], box2[1])
        xb, yb = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / float(a1 + a2 - inter + 1e-6)

    @staticmethod
    def _loading_frame() -> np.ndarray:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "System Initializing... Please Wait.",
            (350, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2,
        )
        return frame

    @staticmethod
    def _encode(frame: np.ndarray) -> bytes:
        _, buf = cv2.imencode(".jpg", frame)
        return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
