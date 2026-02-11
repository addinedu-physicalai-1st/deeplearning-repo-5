import cv2
import torch
import numpy as np
import sys
import os
import time
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import yt_dlp

# Qt Warning 억제
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from main_ai_root.models.multimodal_net import HybridPetNet
    from main_ai_root.new_project.utils.feature_engineering import PetFeatureExtractor
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# ==========================================
# Configuration
# ==========================================
# 현재 디렉토리 (new_project)
CURRENT_DIR = Path(__file__).parent
TEST_DIR = CURRENT_DIR / "test"
CHECKPOINT_DIR = CURRENT_DIR / "checkpoints"

# 사용자 지정 모델 (Dual Model Strategy)
# 1. Object Detection (Species Classification)
DET_MODEL_PATH = CURRENT_DIR / "yolo26n.pt"  # COCO Pretrained (Dog=16, Cat=15)
# 2. Pose Estimation (Keypoints)
# POSE_MODEL_PATH = CURRENT_DIR / "yolo26n-pose.pt"  # Pose Only (May be person-only)
POSE_MODEL_PATH = (
    PROJECT_ROOT / "main_ai_root" / "models" / "best.pt"
)  # Custom Animal Pose Model

YOUTUBE_URL = "https://www.youtube.com/shorts/ti4vYgIH9j8?feature=share"
DOWNLOADED_VIDEO_PATH = TEST_DIR / "youtube_test(cat_angry).mp4"
VIDEO_PATH = DOWNLOADED_VIDEO_PATH
# VIDEO_PATH = TEST_DIR / "youtube_test(cat_2).mp4"


def download_youtube_video(url, output_path):
    print(f"📥 Downloading YouTube Video: {url}")
    ydl_opts = {
        "format": "best[ext=mp4]",
        "outtmpl": str(output_path),
        "quiet": True,
        "overwrites": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"✅ Download Complete: {output_path}")


FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
OUTPUT_VIDEO_PATH = TEST_DIR / "inference_result(cat_angry).mp4"
OUTPUT_CSV_PATH = TEST_DIR / "inference_result(cat_angry).csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 30
IMG_SIZE = 224

# Emotions & Species Maps
# COCO Class IDs
COCO_DOG_ID = 16
COCO_CAT_ID = 15

# Custom Labels
BINARY_LABELS = {0: "부정", 1: "긍정"}
NEGATIVE_LABELS = {0: "공포/공격성", 1: "불안/슬픔", 2: "화남/불쾌"}
SPECIES_LABELS = {0: "개", 1: "고양이"}  # Internal Logic ID

# Colors
COLOR_POS = (0, 255, 0)
COLOR_NEG = (0, 0, 255)
COLOR_BBOX = (255, 0, 255)
COLOR_SKELETON = (255, 255, 0)

# Image Transform
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_one_model(path, num_classes, num_features):
    model = HybridPetNet(
        num_classes=num_classes,
        num_domain_features=num_features,
        dropout=0.5,
        fusion_type="bottleneck",
    ).to(DEVICE)
    try:
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            sd = (
                checkpoint["model_state_dict"]
                if "model_state_dict" in checkpoint
                else checkpoint
            )
            model.load_state_dict(sd)
            model.eval()
            return model
        else:
            print(f"⚠️ Model Checkpoint not found: {path}")
    except Exception as e:
        print(f"⚠️ Error loading model {path}: {e}")
        return None
    return None


def put_text_korean(img, text, pos, font_size, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def format_time(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


def calculate_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
    return iou


def run_inference():
    # 0. Download Video (Disabled for local file)
    # global VIDEO_PATH
    # if YOUTUBE_URL:
    #     download_youtube_video(YOUTUBE_URL, DOWNLOADED_VIDEO_PATH)
    #     VIDEO_PATH = DOWNLOADED_VIDEO_PATH
    # else:
    #     print("❌ No YouTube URL provided.")
    # return

    print(f"🎥 Inference Target: {VIDEO_PATH}")
    if not VIDEO_PATH.exists():
        print(f"❌ Video file not found: {VIDEO_PATH}")
        return

    # 1. Load Detection Model (Species Classifier)
    if not DET_MODEL_PATH.exists():
        print(f"❌ Detection Model not found: {DET_MODEL_PATH}")
        return
    print(f"🚀 Loading Detector: {DET_MODEL_PATH}")
    det_model = YOLO(str(DET_MODEL_PATH))
    print(f"🔎 Detector Classes: {det_model.names}")

    # 2. Load Pose Model (Keypoints)
    if not POSE_MODEL_PATH.exists():
        print(f"❌ Pose Model not found: {POSE_MODEL_PATH}")
        return
    print(f"🚀 Loading Pose Model: {POSE_MODEL_PATH}")
    pose_model = YOLO(str(POSE_MODEL_PATH))
    print(f"🔎 Pose Model Classes: {pose_model.names}")

    extractor = PetFeatureExtractor(fps=5.0)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"❌ Cannot open video: {VIDEO_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, fps, (width, height))

    kp_buffer = deque(maxlen=WINDOW_SIZE)
    emotion_models = {}
    models_loaded = False

    csv_rows = []
    print(f"▶ START: {VIDEO_PATH.name}")

    emo_buffer = deque(maxlen=20)  # [Update] 감정 결과 스무딩용 버퍼 (20프레임 평균)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        curr_time = frame_count / fps

        # --- Step 1: Object Detection (Species) ---
        # conf=0.4: 개/고양이를 확실하게 잡기 위함
        det_results = det_model(frame, verbose=False, conf=0.4, iou=0.5)[0]

        detected_species_name = "미탐지"
        final_bbox = None
        final_kps = None
        final_conf = 0.0
        emo_text = "..."
        emo_conf = 0.0

        target_model_key = None  # 'dog' or 'cat'

        # Find best object (Dog or Cat)
        best_det_box = None
        max_det_conf = 0.0

        if det_results.boxes is not None:
            for box in det_results.boxes:
                cls_id = int(box.cls.item())
                conf = box.conf.item()
                class_name = det_model.names[cls_id]

                # [DEBUG]
                if frame_count <= 10:
                    print(
                        f"[Frame {frame_count}] Detect: ID={cls_id} Name='{class_name}' Conf={conf:.2f}"
                    )

                # Check Species
                current_species = None
                if cls_id == COCO_DOG_ID:  # 16
                    current_species = "dog"
                elif cls_id == COCO_CAT_ID:  # 15
                    current_species = "cat"

                if current_species and conf > max_det_conf:
                    max_det_conf = conf
                    best_det_box = box.xyxy[0].cpu().numpy()
                    target_model_key = current_species
                    detected_species_name = (
                        "개" if current_species == "dog" else "고양이"
                    )
                    final_conf = conf
                    final_bbox = best_det_box

        # --- Step 2: Pose Estimation (Keypoints) ---
        # If we detected a valid Dog/Cat, try to find matching Skeleton
        if final_bbox is not None:
            # Run Pose Model on Full Frame (Better Context)
            pose_results = pose_model(frame, verbose=False, conf=0.3)[0]

            # [DEBUG] Pose Results
            if frame_count <= 10 and pose_results.boxes is not None:
                for pbox in pose_results.boxes:
                    pid = int(pbox.cls.item())
                    pname = pose_model.names[pid]
                    print(
                        f"[Frame {frame_count}] Pose Detect: ID={pid} Name='{pname}' Conf={pbox.conf.item():.2f}"
                    )

            best_iou = 0.0
            matched_kps = None

            if (
                pose_results.keypoints is not None
                and len(pose_results.keypoints.data) > 0
            ):
                # Find Pose Box that matches Detection Box
                poses = pose_results.keypoints.data.cpu().numpy()  # (N, K, 3)
                pose_boxes = pose_results.boxes.xyxy.cpu().numpy()  # (N, 4)

                for i, p_box in enumerate(pose_boxes):
                    iou = calculate_iou(final_bbox, p_box)
                    if iou > best_iou:
                        best_iou = iou
                        matched_kps = poses[i]

            # If match found (IOU > 0.3), use it
            if best_iou > 0.3:
                final_kps = matched_kps
            else:
                # Fallback: If no heavy overlap, maybe Pose model missed detection or is drift.
                # Just take the highest confidence pose if only one person/animal? (Risky)
                # For now, require match.
                pass

        # --- Step 3: Emotion Inference ---
        if final_kps is not None and target_model_key:
            # Normalize Handlers
            kps = final_kps  # (K, 3)
            norm_kps = kps[:, :2].copy()
            norm_kps[:, 0] /= width
            norm_kps[:, 1] /= height

            kp_buffer.append(norm_kps)

            if len(kp_buffer) == WINDOW_SIZE:
                try:
                    # 1. Extract Features
                    seq = np.array(kp_buffer)
                    feats = extractor.extract(seq)  # (T, D)

                    # 2. Load Models Once
                    if not models_loaded:
                        fd = feats.shape[1]
                        for sp in ["cat", "dog"]:
                            emotion_models[f"{sp}_bin"] = load_one_model(
                                CHECKPOINT_DIR / f"hierarchical_{sp}_binary.pth", 2, fd
                            )
                            emotion_models[f"{sp}_neg"] = load_one_model(
                                CHECKPOINT_DIR / f"hierarchical_{sp}_negative.pth",
                                3,
                                fd,
                            )
                        models_loaded = True

                    # 3. Model Inference
                    bin_m = emotion_models.get(f"{target_model_key}_bin")
                    neg_m = emotion_models.get(f"{target_model_key}_neg")

                    if bin_m:
                        img_t = transform(frame).unsqueeze(0).to(DEVICE)
                        kpt_t = torch.from_numpy(seq).float().unsqueeze(0).to(DEVICE)
                        feat_t = torch.from_numpy(feats).float().unsqueeze(0).to(DEVICE)

                        with torch.no_grad():
                            # Binary Check
                            prob = torch.softmax(bin_m(img_t, kpt_t, feat_t), dim=1)
                            neg_prob = prob[0][0].item()

                            # [Buffer] 현재 프레임의 부정 확률 저장
                            emo_buffer.append(neg_prob)

                            # [Smoothing] 최근 20프레임 평균 부정 확률 계산 (버퍼 증량)
                            smoothed_neg_prob = sum(emo_buffer) / len(emo_buffer)

                            # [Species-Specific Threshold] 종별 최적화 V9 (Geometric Logic)

                            # 기본 임계값 설정
                            if target_model_key == "cat":
                                base_threshold = 0.60  # 하악질 포착 강화를 위해 상향 (Sit 로직이 있으므로 안전)
                            else:  # dog
                                base_threshold = 0.40

                            # [Motion Check] 움직임 계산
                            motion_score = 0.0
                            if len(kp_buffer) >= 5:
                                curr = kp_buffer[-1][:, :2]
                                prev = kp_buffer[-5][:, :2]
                                dist = np.linalg.norm(curr - prev, axis=1).mean()
                                motion_score = dist

                            posture_debug = 0.0
                            behavior_debug = ""

                            # =========================================================
                            # 🐾 행동학적 특징 분석 (Ethological Feature Analysis)
                            # =========================================================
                            if final_kps is not None:
                                # Keypoints: 0:Nose, 5:R_Sh, 6:L_Sh, 9:R_Hip, 10:L_Hip, 13:TailStart, 14:TailEnd
                                nose_y = final_kps[0][1]
                                shoulder_y = (final_kps[5][1] + final_kps[6][1]) / 2.0
                                hip_y = (final_kps[9][1] + final_kps[10][1]) / 2.0
                                tail_start_y = final_kps[13][1]
                                tail_end_y = final_kps[14][1]

                                # 1. DOG Logic
                                if target_model_key == "dog":
                                    # A. Motion (기존)
                                    is_high_motion = motion_score > 0.025
                                    is_uncertain_neg = smoothed_neg_prob <= 0.80

                                    if is_high_motion and is_uncertain_neg:
                                        base_threshold += 0.40
                                        behavior_debug = "Run"

                                    # B. Play Bow (신규): 어깨가 엉덩이보다 낮음 (y값이 큼)
                                    # Play Bow: Shoulder Y > Hip Y (꽤 많이)
                                    play_bow_score = shoulder_y - hip_y
                                    if (
                                        play_bow_score > 0.05
                                    ):  # 어깨가 엉덩이보다 5% 이상 아래
                                        smoothed_neg_prob -= 0.30
                                        if smoothed_neg_prob < 0:
                                            smoothed_neg_prob = 0.0
                                        behavior_debug = "Bow"

                                # 2. CAT Logic (V20: Sniffing vs Crouch)
                                elif target_model_key == "cat":
                                    # 좌표계: Y가 클수록 화면 아래쪽(Low), 작을수록 위쪽(High)

                                    # [지표 1] 자세 분석의 기초
                                    # 공격 자세: 머리는 낮고(Low), 엉덩이는 높음(High) -> 튀어나갈 준비

                                    # A. 머리가 어깨보다 낮은가? (Head Low)
                                    is_head_low = (nose_y - shoulder_y) > 0.05

                                    # B. 엉덩이가 어깨보다 확실히 높은가? (Hip High)
                                    is_hip_high = (shoulder_y - hip_y) > 0.05

                                    # C. 꼬리 상태 (Tail Check) - 중요!
                                    # 꼬리가 서 있는가? (TailStart > TailEnd)
                                    # 기준을 조금 낮춰(>0.10) 수평보다 약간만 높아도 인정 (탐색 시 꼬리는 보통 45도 이상)
                                    tail_up_score = tail_start_y - tail_end_y
                                    is_tail_rigid_up = tail_up_score > 0.10

                                    # [자세 분류]

                                    # 1. 탐색 자세 (Sniffing) - 긍정/중립
                                    # 머리는 낮지만(냄새 맡음), 꼬리는 세우고 있음(경계 없음)
                                    is_sniffing = is_head_low and is_tail_rigid_up

                                    # 2. 공격 웅크림 (Aggressive Crouch) - 부정
                                    # 머리 낮고, 엉덩이 높고, **꼬리도 낮음(은폐)**
                                    # Sniffing이 아니면서 Head Low + Hip High인 경우
                                    is_aggressive_crouch = (
                                        is_head_low and is_hip_high
                                    ) and (not is_tail_rigid_up)

                                    # [지표 2] 꼬리 높이 (Smart Tail Up)
                                    # 공격 자세만 아니면 Tail Up 인정
                                    is_tail_up = is_tail_rigid_up and (
                                        not is_aggressive_crouch
                                    )

                                    # [지표 3] 꼬리 흔들림 (Wagging)
                                    wag_score = 0.0
                                    if len(kp_buffer) >= 10:
                                        tail_x_seq = [
                                            k[14][0] for k in list(kp_buffer)[-10:]
                                        ]
                                        wag_score = np.std(tail_x_seq)
                                    is_wagging = wag_score > 0.08

                                    # [지표 4] 앉아있는지 (Sitting)
                                    cat_posture_score = hip_y - nose_y
                                    is_sitting = cat_posture_score > 0.20

                                    # [종합 판단 로직]

                                    # 1. 부정 요인 (우선순위 높음)
                                    if is_aggressive_crouch:
                                        # 진짜 공격 자세 (꼬리 내림)
                                        smoothed_neg_prob += 0.35
                                        behavior_debug = "Crouch"

                                    elif is_wagging:
                                        # 꼬리 흔듦
                                        smoothed_neg_prob += 0.25
                                        behavior_debug = f"Wag({wag_score:.2f})"

                                    # 2. 긍정 요인 (Sniffing 포함)
                                    elif is_sniffing:
                                        # 냄새 맡는 중 (호기심) -> 긍정으로 유도
                                        smoothed_neg_prob -= 0.15
                                        behavior_debug = "Sniff"

                                    elif is_sitting or is_tail_up:
                                        # ✨ Safety Lock: 분노 관성
                                        if smoothed_neg_prob > 0.70:
                                            # 화난 상태에선 진정 속도 가속 (-0.15)
                                            # 얌전한 고양이가 오판되었을 때 빠르게 복구하고,
                                            # 진짜 화난 고양이도 자세를 풀면 서서히 돌아오도록 함.
                                            smoothed_neg_prob -= 0.15
                                            behavior_debug = "Relaxing"
                                        else:
                                            # 평온한 상태에서만 즉각적인 긍정 보정 적용
                                            smoothed_neg_prob -= 0.30
                                            tag = []
                                            if is_sitting:
                                                tag.append("Sit")
                                            if is_tail_up:
                                                tag.append("TailUp")
                                            behavior_debug = "+".join(tag)

                                    tag = []
                            # [Threshold Check]
                            if smoothed_neg_prob > base_threshold:
                                pred = 0  # Negative
                                score = smoothed_neg_prob
                            else:
                                pred = 1  # Positive
                                score = 1.0 - smoothed_neg_prob

                            # [DEBUG] 화면에 수치 표시
                            area_debug = 0.0
                            if final_bbox is not None:
                                x1, y1, x2, y2 = final_bbox
                                area_debug = ((x2 - x1) * (y2 - y1)) / (width * height)

                            debug_str = f"Neg:{smoothed_neg_prob:.2f} Mot:{motion_score:.3f} {behavior_debug}"
                            cv2.putText(
                                frame,
                                debug_str,
                                (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2,
                            )

                            if pred == 1:
                                emo_text = "긍정"
                                emo_conf = score
                            else:
                                # Negative Detail (평균적으로 부정일 때만 수행)
                                if neg_m:
                                    np_ = torch.softmax(
                                        neg_m(img_t, kpt_t, feat_t), dim=1
                                    )
                                    n_pred = torch.argmax(np_, dim=1).item()
                                    n_score = np_[0][n_pred].item()
                                    emo_text = NEGATIVE_LABELS.get(n_pred, "부정")
                                    emo_conf = score  # Binary Score 사용 (신뢰도)
                                else:
                                    emo_text = "부정"
                                    emo_conf = score
                except Exception as e:
                    # print(f"Infer Error: {e}")
                    pass

        # --- Visualization ---
        # 1. BBox & Species (From Detection Model)
        if final_bbox is not None:
            x1, y1, x2, y2 = map(int, final_bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BBOX, 2)
            label = f"{detected_species_name} ({final_conf:.2f})"
            frame = put_text_korean(frame, label, (x1, y1 - 25), 20, (255, 255, 255))

        # 2. Skeleton (From Pose Model)
        if final_kps is not None:
            for kp in final_kps:
                if kp[2] > 0.3:  # Conf Threshold
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, COLOR_SKELETON, -1)

        # 3. Emotion Label
        if emo_text != "...":
            disp_str = f"{emo_text} ({emo_conf:.0%})"
            cv2.rectangle(frame, (20, 20), (280, 70), (0, 0, 0), -1)
            c = COLOR_POS if "긍정" in emo_text else COLOR_NEG
            frame = put_text_korean(frame, disp_str, (30, 30), 30, c)

        out.write(frame)
        cv2.imshow("Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if frame_count % 10 == 0:
            print(f"\rProcessing frame {frame_count}...", end="")

        # CSV Log
        csv_rows.append(
            {
                "영상파일이름": VIDEO_PATH.name,
                "재생시간": format_time(curr_time),
                "탐지된객체": detected_species_name,
                "객체탐지 정확도": f"{final_conf:.2f}" if final_conf > 0 else "0.00",
                "감정추론": emo_text,
                "감정추론 정확도": f"{emo_conf:.2f}",
            }
        )

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("\n✅ Processing Complete.")

    # Save CSV
    pd.DataFrame(csv_rows).to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ Video: {OUTPUT_VIDEO_PATH}")
    print(f"✅ CSV: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    run_inference()
