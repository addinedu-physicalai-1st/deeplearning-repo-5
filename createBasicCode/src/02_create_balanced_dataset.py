"""
행동(Action) 기반 고품질 균형 데이터셋 생성 (Full Ver.)

[기능 명세]
1. 데이터 파싱 및 필터링
   - JSON 메타데이터 로드
   - 유효 Keypoint 개수 필터링 (13개 이상)
   - BBox 크기 필터링 (2500px² 이상)
   - 다수 개체(2마리 이상) 영상 제외 (단일 개체 학습 집중)

2. 계층적 학습을 위한 데이터 밸런싱 (전략 수정됨)
   - 기존 Flat Model 전략 폐기 (무조건 300~500개 제한)
   - [수정] Hierarchical Model 전략 적용:
     (1) 긍정 클래스(편안/행복): 1000개로 상향 (정상 행동의 다양성 확보)
     (2) 부정 클래스(공포/불안/화남): 5000개 (최대한 보존)
   - 행동(Action) 단위로 그룹화하여 특정 행동에 편중되지 않도록 제어

3. 분석 및 시각화
   - 샘플링 전/후의 히트맵(Heatmap) 생성
   - 행동/감정 분포 막대 그래프 생성
   - 메모리 사용량 및 소요 시간 로깅

작성일: 2026-02-09
"""

import os
import sys
import json
import gc
import math
import time
from pathlib import Path
import pandas as pd
import numpy as np
import psutil  # 메모리 측정을 위해 필수
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ============================================================================
# 1. 시각화 설정 (한글 폰트)
# ============================================================================
font_prop = fm.FontProperties(family="sans-serif", size=10)
# 리눅스 환경(Ubuntu) NanumGothic 경로 확인
if os.path.exists("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    font_prop = fm.FontProperties(fname=font_path, size=10)
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "NanumGothic"
    print(f"[INFO] 한글 폰트 로드 성공: {font_path}")
else:
    # 폰트가 없을 경우 기본 폰트 사용 (한글 깨짐 주의)
    plt.rcParams["font.family"] = "DejaVu Sans"
    print("[WARNING] NanumGothic 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")

plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# ============================================================================
# 2. 경로 및 전역 설정
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# DATA_ROOT: pet-care-robot/data (v2의 형제 디렉토리)
DATA_ROOT = PROJECT_ROOT.parent / "data"

# OUTPUT_DIR: src/ai/balanced_metadata (현재 디렉토리 내)
OUTPUT_DIR = Path(__file__).resolve().parent / "balanced_metadata"
# ANALYSIS_DIR: src/ai/sampled_analysis_output (현재 디렉토리 내)
ANALYSIS_DIR = Path(__file__).resolve().parent / "sampled_analysis_output"

# 디렉토리 생성
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
ANALYSIS_DIR.mkdir(exist_ok=True, parents=True)

print(f"[INFO] 프로젝트 루트: {PROJECT_ROOT}")
print(f"[INFO] 데이터 루트: {DATA_ROOT}")
print(f"[INFO] 메타데이터 저장소: {OUTPUT_DIR}")
print(f"[INFO] 분석 결과 저장소: {ANALYSIS_DIR}")

# ----------------------------------------------------------------------------
# [설정] 샘플링 파라미터
# ----------------------------------------------------------------------------
# 행동별 기본 최대치 (Emotion Cap에 걸리지 않는 경우의 안전장치)
MAX_VIDEOS_PER_ACTION = 1500

# [중요] 감정별 쿼터 (Quota) 설정
# 계층적 분류(Binary -> Detail)를 위해 긍정 데이터의 다양성을 확보합니다.
EMOTION_CAPS = {
    # 1. 긍정 클래스 (Positive): 다양성 확보를 위해 1000개로 설정
    "편안/안정": 1000,
    "행복/즐거움": 1000,

    # 2. 부정 클래스 (Negative): 희소 데이터이므로 최대한 보존 (사실상 무제한)
    "공포/공격성": 5000,
    "불안/슬픔": 5000,
    "화남/불쾌": 5000,

    # 예외 처리
    "Unknown": 100
}

# ----------------------------------------------------------------------------
# [매핑] 행동 코드 -> 한글 명칭
# ----------------------------------------------------------------------------
CAT_ACTION_MAP = {
    "ARCH": "허리를 아치로 세움",
    "ARMSTRETCH": "앞발을 뻗어 휘적거리는 동작",
    "FOOTPUSH": "앞발로 꾹꾹 누르는 동작",
    "GETDOWN": "납작 엎드리는 동작",
    "GROOMING": "그루밍하는 동작",
    "HEADING": "머리를 들이대는 동작",
    "LAYDOWN": "옆으로 눕는 동작",
    "LYING": "옆으로 누워 있음",
    "ROLL": "좌우로 뒹구는 동작",
    "SITDOWN": "발을 숨기고 웅크리고 앉는 동작",
    "TAILING": "꼬리를 흔드는 동작",
    "WALKRUN": "걷거나 달리는 동작",
}

DOG_ACTION_MAP = {
    "BODYLOWER": "엎드리는 동작",
    "BODYSCRATCH": "몸을 긁는 동작",
    "BODYSHAKE": "몸을 터는 동작",
    "FEETUP": "두 앞발을 들어 올리는 동작",
    "FOOTUP": "앞발 하나를 들어 올리는 동작",
    "HEADING": "머리를 앞으로 들이미는 동작",
    "LYING": "배와 목을 보여주며 눕는 동작",
    "MOUNTING": "마운팅",
    "SIT": "앉는 동작",
    "TAILING": "꼬리를 위로 올리고 흔드는 동작",
    "TAILLOW": "꼬리를 아래로 내리는 동작",
    "TURN": "빙글빙글 도는 동작",
    "WALKRUN": "걷거나 달리는 동작",
}


# ============================================================================
# 3. 데이터 파싱 및 처리 함수
# ============================================================================
def get_action_label(json_path: Path) -> tuple:
    """
    JSON 파일 경로를 분석하여 종(Species)과 행동(Action)을 추출합니다.
    예: .../CAT/ARCH/video.json -> ('CAT', '허리를 아치로 세움', 'ARCH')
    """
    try:
        action_code = json_path.parent.name.upper()
        species_folder = json_path.parents[2].name.upper()

        if "CAT" in species_folder:
            species = "CAT"
            action = CAT_ACTION_MAP.get(action_code, action_code)
        elif "DOG" in species_folder:
            species = "DOG"
            action = DOG_ACTION_MAP.get(action_code, action_code)
        else:
            species = "Unknown"
            action = action_code

        return species, action, action_code
    except Exception:
        return "Unknown", "Unknown", "Unknown"


def get_image_dir_from_json(json_path: Path) -> Path:
    """JSON 파일 경로에 대응하는 이미지 폴더 경로를 반환합니다."""
    species_dir = json_path.parents[2]
    action_name = json_path.parent.name
    video_name = json_path.stem
    return species_dir / "images" / action_name / video_name


def process_video_frames(json_path: Path) -> list:
    """
    단일 JSON 파일을 파싱하여 유효한 프레임 정보를 리스트로 반환합니다.
    
    [필터링 조건]
    1. 다수 개체(2마리 이상) 영상 제외
    2. Keypoint 13개 미만 검출 프레임 제외
    3. Bounding Box 크기 2500px 미만 제외
    4. 이미지 파일이 실제로 존재하지 않으면 제외
    """
    frames = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        annotations = data.get("annotations", [])
        species, action, action_code = get_action_label(json_path)

        # 1. 다수 개체 필터링 (노이즈 방지)
        owner_meta = meta.get("owner", {})
        animal_count = owner_meta.get("animalCount", 1)
        if animal_count > 1:
            return []  # 다수 개체 영상은 통째로 스킵

        # 감정 추출 및 통합
        inspect_meta = meta.get("inspect", {})
        emotion = inspect_meta.get("emotion", owner_meta.get("emotion", "Unknown"))
        
        # [통합] 공포와 공격성은 상황적/신체적 특징이 유사하므로 하나로 묶음
        if emotion in ["공포", "공격성"]:
            emotion = "공포/공격성"

        # 메타데이터 추출
        width = meta.get("width", 0)
        height = meta.get("height", 0)
        animal_info = meta.get("animal", {})
        breed = animal_info.get("breed", "Unknown")
        age = animal_info.get("age", -1)

        # 비디오 ID 생성 (파일명 중복 방지를 위해 폴더 구조 포함)
        video_file = data.get("file_video", "")
        raw_video_name = Path(video_file).stem if video_file else json_path.stem
        video_id = f"{species}_{action_code}_{raw_video_name}".replace(" ", "_")

        image_dir = get_image_dir_from_json(json_path)

        # 프레임 순서대로 정렬 (시계열 데이터이므로 중요)
        annotations.sort(key=lambda x: x.get("frame_number", 0))

        for ann in annotations:
            keypoints = ann.get("keypoints", {})
            bbox = ann.get("bounding_box", {})
            frame_num = ann.get("frame_number")
            timestamp = ann.get("timestamp")

            # 2. 유효 키포인트 개수 확인 (13개 이상만 허용)
            valid_cnt = sum(1 for v in keypoints.values() if v is not None)
            if valid_cnt < 13:
                continue

            # 3. BBox 크기 확인 (너무 작은 객체는 제외)
            b_w = bbox.get("width", 0)
            b_h = bbox.get("height", 0)
            if (b_w * b_h) < 2500:  # 50x50 픽셀 미만
                continue

            # 이미지 파일 존재 여부 확인
            image_name = f"frame_{frame_num}_timestamp_{timestamp}.jpg"
            image_path = image_dir / image_name

            if not image_path.exists():
                continue

            # 유효한 프레임 정보 저장
            frames.append({
                "json_path": str(json_path),
                "image_path": str(image_path),
                "video_id": video_id,
                "frame_number": frame_num,
                "species": species,
                "action": action,
                "emotion": emotion,
                "action_code": action_code,
                "breed": breed,
                "age": age,
                "video_width": width,
                "video_height": height,
                "bbox_x": bbox.get("x", 0),
                "bbox_y": bbox.get("y", 0),
                "bbox_width": bbox.get("width", 0),
                "bbox_height": bbox.get("height", 0),
                "keypoints_json": json.dumps(keypoints), # 문자열로 저장 (Parquet 호환)
                "valid_keypoints_count": valid_cnt,
            })

    except Exception as e:
        # 파일 손상 등으로 인한 에러 발생 시 로그 출력 후 건너뜀
        print(f"\n[ERROR] JSON 처리 중 오류: {json_path}")
        print(f"  -> {e}")
        pass

    return frames


def collect_and_filter_data(species: str) -> pd.DataFrame:
    """
    지정된 종(Species)의 모든 데이터를 스캔하여 고품질 프레임만 수집합니다.
    """
    json_files = []
    # Training/Validation 폴더 모두 탐색
    for root_type in ["Training", "Validation"]:
        for species_folder in [species.upper(), species.capitalize(), species.lower()]:
            search_path = DATA_ROOT / root_type / species_folder
            if search_path.exists():
                files = list(search_path.rglob("*.json"))
                json_files.extend(files)

    print(f"\n[{species.upper()}] 데이터 수집 시작...")
    print(f"  - 검색된 JSON 파일 수: {len(json_files):,}개")

    all_records = []
    
    # tqdm을 사용하여 진행 상황 시각화
    for json_path in tqdm(json_files, desc=f"  Parsing {species}"):
        frames = process_video_frames(json_path)
        all_records.extend(frames)

    df = pd.DataFrame(all_records)
    print(f"[{species.upper()}] 1차 수집 완료: 총 {len(df):,} 프레임")
    return df


# ============================================================================
# 4. 데이터 밸런싱 (Undersampling) 함수
# ============================================================================
def undersample_by_action(df: pd.DataFrame, species: str) -> pd.DataFrame:
    """
    [핵심 로직] 행동(Action) 내 감정별 쿼터제 적용 (Stratified Balancing)
    
    기존에는 단순히 행동별로 N개를 뽑았으나, 감정 불균형이 심해
    '긍정' 감정은 1000개 제한, '부정' 감정은 5000개(전량) 보존하는 전략을 사용합니다.
    """
    print(f"\n[{species.upper()}] 행동 기반 스마트 밸런싱 시작")
    print(f"  - 긍정 감정 Cap: 1,000 비디오 (다양성 확보)")
    print(f"  - 부정 감정 Cap: 5,000 비디오 (최대한 보존)")

    actions = df["action"].unique()
    sampled_dfs = []

    # 각 행동(Action) 별로 순회
    for action in actions:
        df_action = df[df["action"] == action]
        
        # 비디오 단위로 그룹화 (Train/Val Split 시 누수 방지를 위해 비디오 단위 처리 필수)
        video_groups = df_action.groupby("video_id")
        
        # 각 비디오의 대표 감정 추출 (첫 번째 프레임 기준)
        video_emotions_series = video_groups["emotion"].first()
        unique_emotions = video_emotions_series.unique()
        
        print(f"  > Action: {action} (총 {len(video_emotions_series)}개 비디오)")
        
        for emo in unique_emotions:
            # 해당 감정을 가진 비디오 ID 리스트
            target_vids = video_emotions_series[video_emotions_series == emo].index.tolist()
            count = len(target_vids)
            
            # 설정된 쿼터(Limit) 적용. 공백 제거하여 매칭
            limit = EMOTION_CAPS.get(emo.strip(), 1500)
            
            if count > limit:
                # 제한보다 많으면 랜덤 샘플링 (Undersampling)
                selected_vids = np.random.choice(target_vids, limit, replace=False)
                status = f"📉 축소 ({count} -> {limit})"
            else:
                # 제한보다 적으면 전량 사용
                selected_vids = target_vids
                status = f"✅ 보존 ({count})"
            
            # 선택된 비디오 ID에 해당하는 프레임 데이터 추출
            df_selected = df_action[df_action["video_id"].isin(selected_vids)]
            sampled_dfs.append(df_selected)
            
            print(f"    - [{emo}]: {status}")

    # 모든 샘플링 결과 병합
    df_balanced = pd.concat(sampled_dfs, ignore_index=True)
    
    # 전체 데이터 셔플링
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"  => 최종 밸런싱 완료: {len(df_balanced):,} 프레임")
    return df_balanced


# ============================================================================
# 5. 시각화 함수
# ============================================================================
def visualize_heatmap(df: pd.DataFrame, title: str, filename: str):
    """행동-감정 간의 데이터 분포를 히트맵으로 시각화하여 저장합니다."""
    plt.figure(figsize=(14, 10))
    
    # 프레임 수가 아닌 '비디오 수' 기준으로 집계 (더 정확한 분포 확인)
    df_unique = df.drop_duplicates(subset=["video_id"])
    ct = pd.crosstab(df_unique["action"], df_unique["emotion"])
    
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5)
    plt.title(f"{title} (Unique Videos)", fontsize=16)
    plt.ylabel("행동 (Action)")
    plt.xlabel("감정 (Emotion)")
    plt.xticks(rotation=45)
    
    save_path = ANALYSIS_DIR / filename
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[Visual] 히트맵 저장 완료: {save_path}")
    plt.close()


def visualize_distributions(df: pd.DataFrame, species: str):
    """행동 및 감정별 분포를 막대 그래프로 시각화합니다."""
    df_unique = df.drop_duplicates(subset=["video_id"])
    
    # 1. 행동 분포
    plt.figure(figsize=(12, 6))
    action_counts = df_unique["action"].value_counts()
    ax = sns.barplot(x=action_counts.index, y=action_counts.values, hue=action_counts.index, palette="viridis", legend=False)
    
    # 막대 위에 수치 표시
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha="center", va="bottom", fontsize=10, xytext=(0, 5), textcoords="offset points")
    
    plt.title(f"{species} Action Distribution", fontsize=15)
    plt.xticks(rotation=45)
    plt.ylabel("Video Count")
    
    save_path_act = ANALYSIS_DIR / f"{species.lower()}_action_dist.png"
    plt.savefig(save_path_act, bbox_inches="tight")
    print(f"[Visual] 행동 분포 저장 완료: {save_path_act}")
    plt.close()


# ============================================================================
# 6. 메인 처리 프로세스
# ============================================================================
def process_species(species: str):
    """특정 종(Species)에 대한 전체 데이터 처리 파이프라인을 실행합니다."""
    print(f"\n{'='*60}")
    print(f"🚀 [{species.upper()}] 데이터셋 처리 시작")
    print(f"{'='*60}")
    
    # 시간 및 메모리 측정 시작
    start_time = time.time()
    process = psutil.Process(os.getpid())
    init_mem = process.memory_info().rss / 1024 / 1024  # MB

    # 1. 데이터 수집 및 기본 필터링
    df = collect_and_filter_data(species)
    
    if df.empty:
        print(f"[ERROR] {species} 데이터를 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    # [분석] 샘플링 전 히트맵 저장
    visualize_heatmap(
        df, 
        f"{species.upper()} BEFORE Sampling (All Videos)", 
        f"{species.lower()}_heatmap_before_sampling.png"
    )
    visualize_distributions(df, f"{species.upper()} (Before)")

    # 2. 데이터 밸런싱 (Undersampling)
    df_balanced = undersample_by_action(df, species.upper())

    # 3. 데이터 분할 (Train/Val/Test)
    print(f"\n[{species.upper()}] 데이터 분할 진행 (60:20:20)")
    unique_vids = df_balanced["video_id"].unique()
    
    # [수정] PyArrow 백엔드 호환성을 위해 명시적으로 리스트 변환 (sklearn indexing 에러 방지)
    unique_vids_list = unique_vids.tolist() if hasattr(unique_vids, "tolist") else list(unique_vids)
    
    # Stratified Split을 위해 비디오별 행동 라벨 추출
    try:
        vid_actions = df_balanced.groupby("video_id")["action"].first()
        # unique_vids_list 순서에 맞춰 라벨 정렬
        vid_actions_aligned = [vid_actions[vid] for vid in unique_vids_list]
        
        # 1차 분할: Train (60%) vs Temp (40%)
        train_vids, temp_vids = train_test_split(
            unique_vids_list, test_size=0.4, stratify=vid_actions_aligned, random_state=42
        )
        
        # Temp 데이터에 대한 라벨 다시 추출
        temp_df = df_balanced[df_balanced["video_id"].isin(temp_vids)]
        temp_actions = temp_df.groupby("video_id")["action"].first()
        temp_vids_aligned = [temp_actions[vid] for vid in temp_vids]
        
        # 2차 분할: Val (20%) vs Test (20%)
        val_vids, test_vids = train_test_split(
            temp_vids, test_size=0.5, stratify=temp_vids_aligned, random_state=42
        )
        print("  - Stratified Split 성공")
    except Exception as e:
        print(f"  - [WARNING] Stratified Split 실패 ({e}), Random Split으로 대체합니다.")
        # [수정] Random Split 시에도 리스트로 변환된 unique_vids_list 사용
        train_vids, temp_vids = train_test_split(unique_vids_list, test_size=0.4, random_state=42)
        val_vids, test_vids = train_test_split(temp_vids, test_size=0.5, random_state=42)

    # 비디오 ID를 기준으로 전체 프레임 데이터 분할
    train = df_balanced[df_balanced["video_id"].isin(train_vids)]
    val = df_balanced[df_balanced["video_id"].isin(val_vids)]
    test = df_balanced[df_balanced["video_id"].isin(test_vids)]

    # 4. 결과 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    train_path = OUTPUT_DIR / species.upper() / f"train_{species.lower()}.parquet"
    val_path = OUTPUT_DIR / species.upper() / f"val_{species.lower()}.parquet"
    test_path = OUTPUT_DIR / species.upper() / f"test_{species.lower()}.parquet"
    
    # 폴더가 없으면 생성
    train_path.parent.mkdir(parents=True, exist_ok=True)

    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)
    test.to_parquet(test_path, index=False)

    print(f"\n[저장 완료]")
    print(f"  - Train: {len(train):,} frames -> {train_path}")
    print(f"  - Val  : {len(val):,} frames -> {val_path}")
    print(f"  - Test : {len(test):,} frames -> {test_path}")

    # 5. 최종 분석
    visualize_heatmap(
        df_balanced, 
        f"{species.upper()} AFTER Sampling (Balanced)", 
        f"{species.lower()}_heatmap_after_sampling.png"
    )
    visualize_distributions(df_balanced, f"{species.upper()} (After)")

    # 6. 리소스 사용량 로그
    end_time = time.time()
    final_mem = process.memory_info().rss / 1024 / 1024
    print(f"\n[Performance Report]")
    print(f"  - 총 소요 시간: {end_time - start_time:.2f}초")
    print(f"  - 메모리 사용량: {final_mem:.2f} MB (증가: {final_mem - init_mem:.2f} MB)")


if __name__ == "__main__":
    process_species("dog")
    process_species("cat")

    print("\n" + "=" * 60)
    print("✅ 모든 데이터셋 처리 작업이 성공적으로 완료되었습니다.")
    print("=" * 60)