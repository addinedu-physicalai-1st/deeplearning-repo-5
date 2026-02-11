"""
멀티모달 학습용 데이터셋 생성 (Clean Sequence Ver.)

기능:
1. balanced_metadata(02번 결과물)를 읽어옵니다. (이미 Train/Val/Test 분할 및 밸런싱 완료됨)
2. 비디오 ID별로 프레임을 묶어 시퀀스 데이터로 변환합니다.
3. 키포인트 좌표를 [0, 1] 범위로 정규화합니다.
4. 학습(06번 코드)에서 바로 로드할 수 있는 PKL 파일로 저장합니다.

출력:
- preprocessed_data/{species}_{split}.pkl
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pickle
import random
from pathlib import Path
from tqdm import tqdm

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# METADATA_DIR: src/ai/balanced_metadata (02번 코드의 출력)
METADATA_DIR = Path(__file__).resolve().parent / "balanced_metadata"
# OUTPUT_DIR: src/ai/preprocessed_data
OUTPUT_DIR = Path(__file__).resolve().parent / "preprocessed_data"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================
# 설정
# ============================================================================

NUM_KEYPOINTS = 15

# 데이터셋 명세서 기준 Keypoint 이름 (0-indexed)
KEYPOINT_NAMES = [
    "nose",                 # 0: 코
    "forehead_center",      # 1: 이마 정중앙
    "mouth_corner",         # 2: 입꼬리
    "lower_lip_center",     # 3: 아래 입술 중앙
    "neck",                 # 4: 목
    "right_front_leg_start",# 5: 오른쪽 어깨
    "left_front_leg_start", # 6: 왼쪽 어깨
    "right_front_ankle",    # 7: 오른쪽 앞발목
    "left_front_ankle",     # 8: 왼쪽 앞발목
    "right_hip",            # 9: 오른쪽 엉덩이
    "left_hip",             # 10: 왼쪽 엉덩이
    "right_back_ankle",     # 11: 오른쪽 뒷발목
    "left_back_ankle",      # 12: 왼쪽 뒷발목
    "tail_start",           # 13: 꼬리 시작
    "tail_end",             # 14: 꼬리 끝
]
NAME_TO_IDX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}


def parse_keypoints(json_str, width, height):
    """
    JSON 문자열 파싱 -> 정규화된 Numpy Array 반환 ([0, 1] 범위)
    """
    try:
        kpts_map = json.loads(json_str)
    except:
        return np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)

    arr = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)

    # 해상도 정보가 유효하지 않으면 0으로 반환 (추후 필터링됨)
    if width <= 0 or height <= 0:
        return arr

    for name, idx in NAME_TO_IDX.items():
        # JSON 키가 이름("nose")일 수도, 인덱스 문자열("1")일 수도 있음
        kp = kpts_map.get(name)
        if not kp:
            kp = kpts_map.get(str(idx + 1))

        if kp:
            # 정규화: 0~1 사이 값으로 변환
            arr[idx] = [kp["x"] / width, kp["y"] / height]
        else:
            # 관측되지 않은 키포인트는 (0, 0) 처리
            arr[idx] = [0.0, 0.0]

    return arr


def create_sequence_dataset(species: str, split: str):
    """
    특정 종(species)과 분할(split)에 대한 Clean Sequence Dataset 생성
    """
    parquet_path = METADATA_DIR / species.upper() / f"{split}_{species.lower()}.parquet"

    if not parquet_path.exists():
        print(f"[Skip] 파일 없음: {parquet_path}")
        return

    print(f"\n[{species.upper()}-{split.upper()}] 데이터 로드 중...")
    
    # [수정] 중복 읽기 제거 및 예외 처리
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"  [Error] Parquet 로드 실패: {e}")
        return

    # 02번 코드에서 이미 밸런싱 완료되었으므로 전량 사용
    # 비디오 ID 리스트 추출 (순서 보장)
    target_videos = df["video_id"].unique().tolist()

    # 빠른 처리를 위해 video_id로 그룹화
    video_groups = df.groupby("video_id")

    sequences = []
    skipped_count = 0

    print(f"  Processing {len(target_videos)} videos...")

    for vid in tqdm(target_videos, desc=f"  Extracting"):
        if vid not in video_groups.groups:
            continue

        group = video_groups.get_group(vid)

        # 프레임 순서 정렬 (매우 중요)
        group = group.sort_values("frame_number")

        # 비디오 해상도 가져오기
        v_w = group.iloc[0].get("video_width", 0)
        v_h = group.iloc[0].get("video_height", 0)

        # [안전장치] 해상도 정보가 없으면 정규화 불가능하므로 스킵
        if v_w <= 0 or v_h <= 0:
            skipped_count += 1
            continue

        # 데이터 추출
        img_paths = group["image_path"].tolist()
        json_strs = group["keypoints_json"].tolist()

        # 키포인트 파싱 및 정규화
        kpts_list = [parse_keypoints(js, v_w, v_h) for js in json_strs]
        kpts_np = np.array(kpts_list, dtype=np.float32)

        # BBox 정보 저장 (나중에 Crop이나 디버깅용으로 사용 가능)
        bbox_cols = ["bbox_x", "bbox_y", "bbox_width", "bbox_height"]
        bboxes_np = group[bbox_cols].fillna(0).values.astype(np.float32)

        action = group.iloc[0]["action"]
        emotion = group.iloc[0]["emotion"]

        # 최종 딕셔너리 생성
        sequences.append(
            {
                "video_id": vid,
                "image_paths": img_paths,
                "keypoints": kpts_np,  # (T, 15, 2) Normalized
                "bboxes": bboxes_np,   # (T, 4) Raw Pixels
                "action": action,
                "emotion": emotion,
                "species": species,
                "video_size": (v_w, v_h),
            }
        )

    # 저장
    save_path = OUTPUT_DIR / f"{species.lower()}_{split}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(sequences, f)

    print(f"  ✅ 저장 완료: {save_path.name}")
    print(f"     -> 저장된 시퀀스 수: {len(sequences):,}")
    if skipped_count > 0:
        print(f"     -> [Warning] 해상도 정보 누락으로 스킵됨: {skipped_count}건")


if __name__ == "__main__":
    print("=" * 80)
    print("멀티모달 학습용 데이터셋 생성 (Stratified Reduction & Clean Ver)")
    print(" - 02번 코드에서 생성된 Balanced Parquet 파일을 PKL로 변환합니다.")
    print(" - Keypoint는 이미지 크기에 맞춰 [0, 1]로 정규화됩니다.")
    print("=" * 80)

    # 재현성을 위한 시드 고정
    random.seed(42)
    np.random.seed(42)

    # DOG 데이터셋 생성
    create_sequence_dataset("DOG", "train")
    create_sequence_dataset("DOG", "val")
    create_sequence_dataset("DOG", "test")

    # CAT 데이터셋 생성
    create_sequence_dataset("CAT", "train")
    create_sequence_dataset("CAT", "val")
    create_sequence_dataset("CAT", "test")