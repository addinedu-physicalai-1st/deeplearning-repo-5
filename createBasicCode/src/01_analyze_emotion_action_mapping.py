"""
감정-행동 매핑 비디오 단위 분석
- 유효한 키포인트(13개 이상)와 bbox 크기로 필터링
- 비디오 단위로 데이터 분석
- Undersampling 관련 코드 제거
"""

import os
import json
import warnings
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib

warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ============================================================================
# 경로 설정
# ============================================================================
# ============================================================================
# 경로 설정
# ============================================================================
# 현재 파일 위치: src/ai/01_analyze_emotion_action_mapping.py
# PROJECT_ROOT: v2
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# DATA_ROOT: pet-care-robot/data (v2의 형제 디렉토리)
DATA_ROOT = PROJECT_ROOT.parent / "data"

# OUTPUT_DIR: src/ai/analysis_output (현재 디렉토리 내)
OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_output"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 한글 폰트 설정
font_prop = fm.FontProperties(family="sans-serif", size=10)
if os.path.exists("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
    font_prop = fm.FontProperties(
        fname="/usr/share/fonts/truetype/nanum/NanumGothic.ttf", size=10
    )
    fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
    plt.rcParams["font.family"] = "NanumGothic"
else:
    plt.rcParams["font.family"] = "DejaVu Sans"

plt.rcParams["axes.unicode_minus"] = False

print(f"[INFO] 프로젝트 루트: {PROJECT_ROOT}")
print(f"[INFO] 데이터 루트: {DATA_ROOT}")
print(f"[INFO] 분석 결과 저장: {OUTPUT_DIR}")


# ============================================================================
# 행동 코드 → 한글 매핑
# ============================================================================
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


def get_action_label(json_path: Path) -> tuple:
    """JSON 파일 경로에서 종(species)과 행동(action) 추출"""
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
    """JSON 경로에서 이미지 디렉토리 경로 추론"""
    species_dir = json_path.parents[2]
    action_name = json_path.parent.name
    video_name = json_path.stem
    return species_dir / "images" / action_name / video_name


def process_video_annotations(json_path: Path) -> dict:
    """
    JSON 파일 파싱 및 유효 프레임 필터링
    - Keypoint 13개 이상
    - BBox 크기 10000 이상

    Returns:
        dict: {"species", "action", "emotion", "valid_frames", "total_frames"}
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        annotations = data.get("annotations", [])

        # 기본 정보 추출
        species, action, action_code = get_action_label(json_path)

        # 감정 추출
        inspect_meta = meta.get("inspect", {})
        owner_meta = meta.get("owner", {})

        # 단일 개체 필터링
        animal_count = owner_meta.get("animalCount", 1)
        if animal_count > 1:
            return None

        emotion = inspect_meta.get("emotion", owner_meta.get("emotion", "Unknown"))

        # 감정 클래스 통합 (공포 + 공격성)
        if emotion in ["공포", "공격성"]:
            emotion = "공포/공격성"

        if species == "Unknown" or emotion == "Unknown":
            return None

        # 이미지 디렉토리 확인
        image_dir = get_image_dir_from_json(json_path)

        # 프레임 필터링
        valid_frames = 0
        total_frames = len(annotations)

        for ann in annotations:
            keypoints = ann.get("keypoints", {})
            bbox = ann.get("bounding_box", {})
            frame_num = ann.get("frame_number")
            timestamp = ann.get("timestamp")

            # 유효 키포인트 개수 확인
            valid_cnt = sum(1 for v in keypoints.values() if v is not None)

            # Keypoint 13개 이상
            if valid_cnt < 13:
                continue

            # BBox 크기 필터링
            b_w = bbox.get("width", 0)
            b_h = bbox.get("height", 0)
            if (b_w * b_h) < 10000:
                continue

            # 이미지 존재 확인
            image_name = f"frame_{frame_num}_timestamp_{timestamp}.jpg"
            image_path = image_dir / image_name
            if not image_path.exists():
                continue

            valid_frames += 1

        # 유효 프레임이 없으면 제외
        if valid_frames == 0:
            return None

        return {
            "species": species,
            "action": action,
            "action_code": action_code,
            "emotion": emotion,
            "valid_frames": valid_frames,
            "total_frames": total_frames,
            "json_path": str(json_path),
        }

    except Exception as e:
        return None


def analyze_video_distribution():
    """비디오 단위 데이터 분석"""
    print("\n" + "=" * 80)
    print("📊 비디오 단위 감정-행동 분석 (유효 Keypoint & BBox 필터링)")
    print("=" * 80)

    # 모든 JSON 파일 수집
    json_files = []
    for root_type in ["Training", "Validation"]:
        search_path = DATA_ROOT / root_type
        if search_path.exists():
            files = list(search_path.rglob("*.json"))
            print(f"  - {root_type}: {len(files):,}개 JSON 파일 발견")
            json_files.extend(files)

    if not json_files:
        print("[ERROR] JSON 파일을 찾을 수 없습니다.")
        return

    print(f"\n[분석] 총 {len(json_files):,}개 파일 처리 중...")

    # 비디오별 데이터 수집
    video_records = []
    for json_path in tqdm(json_files, desc="Processing"):
        result = process_video_annotations(json_path)
        if result:
            video_records.append(result)

    if not video_records:
        print("[ERROR] 유효한 데이터가 없습니다.")
        return

    df = pd.DataFrame(video_records)

    print(f"\n[결과] 유효 비디오 수: {len(df):,}개")
    print(f"        총 유효 프레임 수: {df['valid_frames'].sum():,}개")
    print(f"        비디오당 평균 프레임: {df['valid_frames'].mean():.1f}개")

    # 종별 분석
    for species in ["DOG", "CAT"]:
        df_sp = df[df["species"] == species].copy()
        if df_sp.empty:
            continue

        print(f"\n" + "=" * 80)
        print(f"📊 {species} 비디오 단위 분석")
        print("=" * 80)

        # 1. 감정별 비디오 수
        print(f"\n[{species}] 감정별 비디오 수:")
        emotion_video_counts = df_sp["emotion"].value_counts()
        for emo, cnt in emotion_video_counts.items():
            frames = df_sp[df_sp["emotion"] == emo]["valid_frames"].sum()
            avg_frames = frames / cnt
            pct = cnt / len(df_sp) * 100
            print(
                f"  {emo:20s}: {cnt:6,}개 비디오 ({pct:5.2f}%) | {frames:,}개 프레임 (평균 {avg_frames:.1f})"
            )

        # 2. 행동별 비디오 수
        print(f"\n[{species}] 행동별 비디오 수 (상위 15개):")
        action_video_counts = df_sp["action"].value_counts().head(15)
        for act, cnt in action_video_counts.items():
            frames = df_sp[df_sp["action"] == act]["valid_frames"].sum()
            avg_frames = frames / cnt
            pct = cnt / len(df_sp) * 100
            print(
                f"  {act:35s}: {cnt:6,}개 비디오 ({pct:5.2f}%) | {frames:,}개 프레임 (평균 {avg_frames:.1f})"
            )

        # 3. 감정-행동 크로스탭 (비디오 수)
        crosstab = pd.crosstab(df_sp["action"], df_sp["emotion"], margins=True)

        crosstab_path = OUTPUT_DIR / f"{species}_video_emotion_action_crosstab.csv"
        crosstab.to_csv(crosstab_path, encoding="utf-8-sig")
        print(f"\n[저장] 크로스탭: {crosstab_path}")

        # 4. 감정별 대표 행동 (비디오 수 기준)
        print(f"\n[{species}] 감정별 대표 행동 (상위 3개, 비디오 수):")
        for emotion in emotion_video_counts.index:
            df_emo = df_sp[df_sp["emotion"] == emotion]
            top_actions = df_emo["action"].value_counts().head(3)
            print(f"\n  [{emotion}] (총 {len(df_emo):,}개 비디오)")
            for act, cnt in top_actions.items():
                pct = cnt / len(df_emo) * 100
                print(f"    - {act}: {cnt:,}개 비디오 ({pct:.1f}%)")

        # 시각화
        visualize_video_analysis(df_sp, species)

    # 전체 데이터 저장
    summary_path = OUTPUT_DIR / "video_distribution_summary.csv"
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[저장] 전체 요약: {summary_path}")

    print("\n" + "=" * 80)
    print("✅ 분석 완료!")
    print("=" * 80)


def visualize_video_analysis(df: pd.DataFrame, species: str):
    """비디오 분석 결과 시각화"""
    print(f"\n[{species}] 시각화 생성 중...")
    sns.set_style("whitegrid")

    # 감정별 비디오 수
    emotion_counts = df["emotion"].value_counts()

    # 1. 감정 분포 (비디오 수)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x=emotion_counts.index,
        y=emotion_counts.values,
        hue=emotion_counts.index,
        palette="coolwarm",
        legend=False,
    )
    plt.title(
        f"{species} - 감정별 비디오 수",
        fontproperties=font_prop,
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("감정", fontproperties=font_prop, fontsize=12)
    plt.ylabel("비디오 수", fontproperties=font_prop, fontsize=12)
    ax.set_xticks(range(len(emotion_counts)))
    ax.set_xticklabels(
        emotion_counts.index, rotation=45, ha="right", fontproperties=font_prop
    )

    for i, v in enumerate(emotion_counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{species}_video_emotion_distribution.png", dpi=150)
    plt.close()

    # 2. 행동 분포 (비디오 수, 상위 12개)
    action_counts = df["action"].value_counts().head(12)

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        x=action_counts.values,
        y=action_counts.index,
        hue=action_counts.index,
        palette="viridis",
        legend=False,
    )
    plt.title(
        f"{species} - 행동별 비디오 수 (Top 12)",
        fontproperties=font_prop,
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("비디오 수", fontproperties=font_prop, fontsize=12)
    plt.ylabel("행동", fontproperties=font_prop, fontsize=12)
    ax.set_yticks(range(len(action_counts)))
    ax.set_yticklabels(action_counts.index, fontproperties=font_prop)

    for i, v in enumerate(action_counts.values):
        ax.text(v, i, f" {v:,}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{species}_video_action_distribution.png", dpi=150)
    plt.close()

    # 3. 감정-행동 크로스탭 히트맵
    crosstab = pd.crosstab(df["action"], df["emotion"])
    row_sums = crosstab.sum(axis=1).sort_values(ascending=False)
    crosstab_sorted = crosstab.loc[row_sums.index]

    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(
        crosstab_sorted,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        cbar_kws={"label": "비디오 수"},
        linewidths=0.5,
    )
    plt.title(
        f"{species} - 감정-행동 크로스탭 (비디오 수)",
        fontproperties=font_prop,
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("감정", fontproperties=font_prop, fontsize=12)
    plt.ylabel("행동", fontproperties=font_prop, fontsize=12)
    plt.xticks(rotation=45, ha="right", fontproperties=font_prop)
    plt.yticks(rotation=0, fontproperties=font_prop)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.set_ylabel("비디오 수", fontproperties=font_prop)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{species}_video_crosstab_heatmap.png", dpi=150)
    plt.close()

    print(f"  ✅ {species} 시각화 완료")


if __name__ == "__main__":
    analyze_video_distribution()
