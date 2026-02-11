"""
평가 지표 및 시각화 유틸리티
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from typing import Dict, List, Optional
from pathlib import Path

# ============================================================================
# 한글 폰트 설정 (Global)
# ============================================================================
FONT_LOADED = False
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

if os.path.exists(FONT_PATH):
    try:
        fm.fontManager.addfont(FONT_PATH)
        # get_name() 대신 'NanumGothic' 직접 지정이 더 안정적임
        plt.rcParams["font.family"] = "NanumGothic"
        plt.rcParams["axes.unicode_minus"] = False
        FONT_LOADED = True
        # print(f"[INFO] Metrics Utils: 한글 폰트 로드 성공 ({FONT_PATH})")
    except Exception as e:
        print(f"[WARNING] Metrics Utils: 한글 폰트 로드 실패: {e}")
else:
    # print("[WARNING] Metrics Utils: NanumGothic 폰트를 찾을 수 없습니다.")
    pass


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    average: str = "weighted",
) -> Dict[str, float]:
    """
    분류 성능 지표 계산
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
):
    """
    Confusion Matrix 시각화 및 저장
    """
    # 함수 호출 시점에 다시 한번 폰트 체크 (안전장치)
    if FONT_LOADED:
        plt.rcParams["font.family"] = "NanumGothic"

    cm = confusion_matrix(y_true, y_pred)

    # 정규화된 Confusion Matrix
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 10},
    )

    plt.title(title, fontsize=15)
    plt.ylabel("Reference (True)", fontsize=12)
    plt.xlabel("Prediction", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[Visual] Confusion Matrix 저장 완료: {save_path}")

    plt.close()


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """
    학습 히스토리 시각화
    """
    if FONT_LOADED:
        plt.rcParams["font.family"] = "NanumGothic"

    epochs = len(history.get("train_loss", []))
    if epochs == 0:
        return

    x = range(1, epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Loss
    axes[0].plot(x, history["train_loss"], "b-", label="Train Loss")
    axes[0].plot(x, history["val_loss"], "r-", label="Val Loss")
    axes[0].set_title("Loss Curve", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Accuracy
    if "train_acc" in history and "val_acc" in history:
        axes[1].plot(x, history["train_acc"], "b-", label="Train Acc")
        axes[1].plot(x, history["val_acc"], "r-", label="Val Acc")
        axes[1].set_title("Accuracy Curve", fontsize=12)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[Visual] 학습 히스토리 저장 완료: {save_path}")

    plt.close()
