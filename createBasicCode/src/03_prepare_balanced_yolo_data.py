"""
균형 잡힌 메타데이터 기반 YOLO 데이터셋 준비

목적:
1. balanced_metadata에서 균형 잡힌 데이터 로드
2. [최적화] 비디오별 희소 샘플링(Sparse Sampling)으로 중복 제거
3. [최적화] 학습 효율을 위한 전체 데이터 수 제한 (Train 4만장 수준)
4. YOLO-Pose 포맷(이미지, 라벨) 생성 및 심볼릭 링크 연결
5. data.yaml 생성

입력:
- balanced_metadata/{SPECIES}/{split}_{species}.parquet

출력:
- balanced_yolo_dataset/
    ├── images/ ...
    ├── labels/ ...
    └── data.yaml
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# ============================================================================
# 경로 설정
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# METADATA_ROOT: src/ai/balanced_metadata
METADATA_ROOT = Path(__file__).resolve().parent / "balanced_metadata"
# YOLO_ROOT: src/ai/balanced_yolo_dataset
YOLO_ROOT = Path(__file__).resolve().parent / "balanced_yolo_dataset"

YOLO_ROOT.mkdir(exist_ok=True, parents=True)

print(f"[INFO] 프로젝트 루트: {PROJECT_ROOT}")
print(f"[INFO] 균형 메타데이터: {METADATA_ROOT}")
print(f"[INFO] YOLO 데이터셋 출력: {YOLO_ROOT}")


# ============================================================================
# YOLO 포맷 변환 설정 & 데이터 다이어트 전략
# ============================================================================
# 클래스 맵핑 (DOG: 0, CAT: 1)
CLASS_MAP = {
    "DOG": 0,
    "CAT": 1,
}

# [전략 1] 비디오 내 희소 샘플링 (중복 방지)
# 한 비디오에서 최대 30장까지만 뽑되, 영상 전체 구간에서 골고루 뽑음
MAX_FRAMES_PER_VIDEO = 30

# [전략 2] 전체 데이터셋 규모 제한 (학습 시간 최적화)
# COCO Keypoints가 약 6만장임. 우리는 약 4만장이면 충분함.
TARGET_DATASET_SIZE = {
    "train": 40000,  # 학습용 (개 2만 + 고양이 2만)
    "val": 4000,  # 검증용 (개 2천 + 고양이 2천)
    "test": 2000,  # 테스트용
}

KEYPOINT_NAMES = [
    "nose",  # 0: 코
    "forehead_center",  # 1: 이마 정중앙
    "mouth_corner",  # 2: 입꼬리/입끝
    "lower_lip_center",  # 3: 아래 입술 중앙
    "neck",  # 4: 목
    "right_front_shoulder",  # 5: 앞다리 오른쪽 시작
    "left_front_shoulder",  # 6: 앞다리 왼쪽 시작
    "right_front_ankle",  # 7: 앞다리 오른쪽 발목
    "left_front_ankle",  # 8: 앞다리 왼쪽 발목
    "right_femur",  # 9: 오른쪽 대퇴골
    "left_femur",  # 10: 왼쪽 대퇴골
    "right_back_ankle",  # 11: 뒷다리 오른쪽 발목
    "left_back_ankle",  # 12: 뒷다리 왼쪽 발목
    "tail_start",  # 13: 꼬리 시작 ⭐
    "tail_tip",  # 14: 꼬리 끝 ⭐⭐⭐
]


def setup_directories():
    """YOLO 디렉토리 구조 생성"""
    for split in ["train", "val", "test"]:
        (YOLO_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
    print("[Setup] 디렉토리 구조 생성 완료")


def convert_bbox_to_yolo(bbox_x, bbox_y, bbox_w, bbox_h, img_w, img_h):
    """Bounding Box를 YOLO 정규화 형식으로 변환"""
    center_x = (bbox_x + bbox_w / 2) / img_w
    center_y = (bbox_y + bbox_h / 2) / img_h
    norm_w = bbox_w / img_w
    norm_h = bbox_h / img_h

    return (
        max(0, min(1, center_x)),
        max(0, min(1, center_y)),
        max(0, min(1, norm_w)),
        max(0, min(1, norm_h)),
    )


def convert_keypoints_to_yolo(keypoints_json: str, img_w: int, img_h: int) -> str:
    """Keypoints를 YOLO-Pose 형식으로 변환"""
    try:
        keypoints = (
            json.loads(keypoints_json)
            if isinstance(keypoints_json, str)
            else keypoints_json
        )
    except:
        keypoints = {}

    yolo_kpts = []

    for i, kp_name in enumerate(KEYPOINT_NAMES, 1):
        kp = None
        idx_str = str(i)

        if kp_name in keypoints and keypoints[kp_name] is not None:
            kp = keypoints[kp_name]
        elif idx_str in keypoints and keypoints[idx_str] is not None:
            kp = keypoints[idx_str]

        if kp is not None:
            x = kp.get("x", 0) / img_w if kp.get("x") is not None else 0
            y = kp.get("y", 0) / img_h if kp.get("y") is not None else 0
            v = 2  # visible
            x = max(0, min(1, x))
            y = max(0, min(1, y))
        else:
            x, y, v = 0, 0, 0

        yolo_kpts.extend([f"{x:.6f}", f"{y:.6f}", str(v)])

    return " ".join(yolo_kpts)


def create_safe_filename(video_id: str, frame_number: int) -> str:
    """안전한 파일명 생성"""
    safe_id = "".join(c for c in video_id if c.isalnum() or c in "-_")
    safe_id = safe_id.encode("ascii", "ignore").decode("ascii")
    return f"{safe_id}_f{frame_number}"


def apply_sparse_sampling(
    df: pd.DataFrame, max_frames=MAX_FRAMES_PER_VIDEO
) -> pd.DataFrame:
    """
    비디오별 희소 샘플링 (Sparse Sampling)
    - 비디오 프레임이 너무 많으면 등간격으로 max_frames 만큼만 추출
    - 연속된 프레임 중복을 방지하여 학습 효율 극대화
    """
    if df.empty:
        return df

    # 비디오 ID별 그룹화
    video_groups = df.groupby("video_id")
    sampled_indices = []

    for _, group in video_groups:
        n_frames = len(group)
        if n_frames <= max_frames:
            # 프레임 수가 적으면 다 씀
            sampled_indices.extend(group.index.tolist())
        else:
            # 프레임 수가 많으면 등간격 추출 (예: 100장 중 30장)
            # np.linspace로 균등한 간격의 인덱스 생성
            indices = np.linspace(0, n_frames - 1, max_frames, dtype=int)
            # 그룹 내에서의 상대적 위치를 전체 df의 인덱스로 변환 필요
            # group.iloc[indices]를 쓰면 됨
            sampled_indices.extend(group.iloc[indices].index.tolist())

    return df.loc[sampled_indices].copy()


def process_split(split: str, species_list=["DOG", "CAT"]):
    """특정 Split의 데이터를 YOLO 형식으로 변환"""
    img_dir = YOLO_ROOT / "images" / split
    lbl_dir = YOLO_ROOT / "labels" / split

    # 이미 많이 생성되어 있으면 스킵 (재실행 방지)
    existing = len(list(img_dir.glob("*.jpg")))
    if existing > 100:
        print(f"\n[Skip] {split.upper()} 이미 {existing:,}개 존재. 건너뜀.")
        return

    print(f"\n{'='*60}")
    print(f"🔧 {split.upper()} 데이터셋 생성 (Target: {TARGET_DATASET_SIZE[split]:,})")
    print("=" * 60)

    # 종별 목표 할당량 (반반)
    target_per_species = TARGET_DATASET_SIZE[split] // len(species_list)

    total_success = 0
    total_fail = 0

    for species in species_list:
        parquet_path = METADATA_ROOT / species / f"{split}_{species.lower()}.parquet"
        if not parquet_path.exists():
            print(f"  [{species}] 메타데이터 없음: {parquet_path}")
            continue

        # 1. 로드
        df = pd.read_parquet(parquet_path)
        print(f"  [{species}] 원본 로드: {len(df):,}개 프레임")

        # 2. 1차 필터링: 비디오 내 희소 샘플링 (중복 제거)
        df = apply_sparse_sampling(df, MAX_FRAMES_PER_VIDEO)
        print(
            f"  [{species}] 희소 샘플링 후: {len(df):,}개 (비디오당 Max {MAX_FRAMES_PER_VIDEO}장)"
        )

        # 3. 2차 필터링: 전체 개수 제한 (목표량 맞추기)
        if len(df) > target_per_species:
            df = df.sample(n=target_per_species, random_state=42)
            print(
                f"  [{species}] 최종 리사이징: {len(df):,}개 (Target: {target_per_species:,})"
            )

        # 4. 변환 및 저장
        success = 0
        fail = 0

        # tqdm으로 진행 상황 표시
        for _, row in tqdm(
            df.iterrows(), total=len(df), desc=f"  Generating {species}"
        ):
            try:
                orig_img_path = Path(row["image_path"])
                if not orig_img_path.exists():
                    fail += 1
                    continue

                base_name = create_safe_filename(row["video_id"], row["frame_number"])
                dst_img = img_dir / f"{base_name}.jpg"
                dst_lbl = lbl_dir / f"{base_name}.txt"

                # 심볼릭 링크
                if not dst_img.exists():
                    os.symlink(orig_img_path, dst_img)

                # 라벨 저장
                cx, cy, w, h = convert_bbox_to_yolo(
                    row["bbox_x"],
                    row["bbox_y"],
                    row["bbox_width"],
                    row["bbox_height"],
                    row["video_width"],
                    row["video_height"],
                )
                kpts_str = convert_keypoints_to_yolo(
                    row["keypoints_json"], row["video_width"], row["video_height"]
                )
                class_id = CLASS_MAP.get(row["species"], 0)

                with open(dst_lbl, "w", encoding="utf-8") as f:
                    f.write(
                        f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kpts_str}\n"
                    )

                success += 1
            except Exception:
                fail += 1

        print(f"  [{species}] 완료: {success:,}장 (실패 {fail:,})")
        total_success += success
        total_fail += fail

    print(f"\n✅ {split.upper()} 최종 합계: {total_success:,}장 생성됨")


def generate_data_yaml():
    """YOLO data.yaml 설정 파일 생성"""
    yaml_content = f"""# 균형 잡힌 반려동물 YOLO-Pose 데이터셋 (Optimized)
# Generated by 03_prepare_balanced_yolo_data.py

path: {YOLO_ROOT}
train: images/train
val: images/val
test: images/test

# Classes
nc: 2
names:
  0: dog
  1: cat

# Keypoints (15 points)
kpt_shape: [15, 3]

# Keypoint names
# 0: nose, 1: forehead_center, 2: mouth_corner, 3: lower_lip_center, 4: neck
# 5: right_front_shoulder, 6: left_front_shoulder, 7: right_front_ankle, 8: left_front_ankle
# 9: right_femur, 10: left_femur, 11: right_back_ankle, 12: left_back_ankle, 13: tail_start, 14: tail_tip

flip_idx: [0, 1, 2, 3, 4, 6, 5, 8, 7, 10, 9, 12, 11, 13, 14]
"""
    with open(YOLO_ROOT / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"\n[Config] data.yaml 생성 완료")


if __name__ == "__main__":
    print("=" * 80)
    print("YOLO-Pose 데이터셋 준비 (Diet Version)")
    print("=" * 80)

    setup_directories()

    for split in ["train", "val", "test"]:
        process_split(split)

    generate_data_yaml()

    print("\n" + "=" * 80)
    print("✅ 작업 완료: 메모리와 학습 속도를 고려한 최적 데이터셋 구축됨")
    print("=" * 80)
