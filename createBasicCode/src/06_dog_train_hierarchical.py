"""
계층적 감정 분류 모델 학습 스크립트 (Hierarchical Classification)
- 06번 코드의 검증된 기법 통합 버전

[설계 철학]
1단계: 긍정(Positive) vs 부정(Negative) 이진 분류
  - 긍정: 편안/안정, 행복/즐거움
  - 부정: 공포/공격성, 불안/슬픔, 화남/불쾌

2단계: 부정 감정 내 세부 분류 (3-class)
  - 공포/공격성, 불안/슬픔, 화남/불쾌

[적용된 기법 - 06번에서 가져옴]
- LDAM Loss (이중 가중치 중복 제거됨)
- Mixup Data Augmentation
- 2-Stage Training (Freeze → Unfreeze)
- Differential Learning Rates
- Warmup
- WeightedRandomSampler
"""

import os
import sys
import gc
import json
import time
import math
import torch
import random
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix, recall_score

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.multimodal_net import HybridPetNet
from src.ai.utils.metrics import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_training_history,
)
from src.ai.utils.feature_engineering import (
    PetFeatureExtractor,
    flip_keypoints_horizontal,
)

# 경로 설정
current_dir = Path(__file__).resolve().parent
DATA_DIR = current_dir / "preprocessed_data"
CHECKPOINT_DIR = current_dir / "checkpoints"
RESULTS_DIR = current_dir / "training_results"

CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# ============================================================================
# 1. 계층적 라벨 매핑
# ============================================================================

# Stage 1: Binary Classification (긍정 vs 부정)
BINARY_EMOTION_MAP = {
    "편안/안정": 0,  # Positive
    "행복/즐거움": 0,  # Positive
    "공포/공격성": 1,  # Negative
    "공포": 1,  # Negative
    "공격성": 1,  # Negative
    "불안/슬픔": 1,  # Negative
    "화남/불쾌": 1,  # Negative
}

BINARY_CLASS_NAMES = ["긍정", "부정"]

# Stage 2: Negative 세부 분류 (3-class)
NEGATIVE_EMOTION_MAP = {
    "공포/공격성": 0,
    "공포": 0,
    "공격성": 0,
    "불안/슬픔": 1,
    "화남/불쾌": 2,
}

NEGATIVE_CLASS_NAMES = ["공포/공격성", "불안/슬픔", "화남/불쾌"]

# 원본 감정 리스트 (Negative만)
NEGATIVE_EMOTIONS = {"공포/공격성", "공포", "공격성", "불안/슬픔", "화남/불쾌"}


# ============================================================================
# 2. 학습 설정 (06번 스타일 통합)
# ============================================================================

# Mixup 설정
MIXUP_ALPHA = 0.4

# Stage 1: Binary Classification Config
STAGE1_CONFIG = {
    "num_classes": 2,
    "window_size": 150,
    "num_keypoints": 15,
    "batch_size": 64,
    "epochs": 40,
    # 2-Stage Training 설정
    "stage1_epochs": 10,  # Image Encoder Frozen 기간
    "stage1_lr": 2e-4,
    "warmup_epochs": 5,
    # Stage 2 (Unfreeze 후) Differential LR
    "stage2_image_lr": 1e-6,  # Image Encoder 보호
    "stage2_keypoint_lr": 1e-4,
    "stage2_classifier_lr": 1e-4,
    "weight_decay": 1e-2,
    "num_workers": 4,
    "seed": 42,
    "early_stop_patience": 7,
    "use_augmentation": True,
    "fps": 5.0,
    # [Binary Tuning] 부정이 63%로 목표 달성, 긍정이 55%로 소폭 하락
    "class_weights": [1.0, 1.4],
    "mixup_prob": 0.5,
}

# Stage 2: Negative Classification Config
STAGE2_CONFIG = {
    "num_classes": 3,
    "window_size": 150,
    "num_keypoints": 15,
    "batch_size": 64,
    "epochs": 50,
    # 2-Stage Training 설정
    "stage1_epochs": 15,
    "stage1_lr": 2e-4,
    "warmup_epochs": 5,
    # Stage 2 Differential LR
    "stage2_image_lr": 1e-6,
    "stage2_keypoint_lr": 1e-4,
    "stage2_classifier_lr": 1e-4,
    "weight_decay": 1e-2,
    "num_workers": 4,
    "seed": 42,
    "early_stop_patience": 7,
    "use_augmentation": True,
    "fps": 5.0,
    # [V5.1 균형 회복] Epoch 12(0.5/0.5/0.45) 재현 목표
    # C1(3.0) 과다 -> 2.0으로 완화
    # C2(1.2) 부족 -> 1.5로 강화
    # C0(8.0) -> 9.0으로 미세 강화
    # Weights: [9.0, 2.0, 1.5]
    "class_weights": [13.0, 2.0, 2.0],
    # [V4] Mixup OFF
    "mixup_prob": 0.0,
}


# ============================================================================
# 3. LDAM Loss (가중치 기능 복구 - Negative Recall 향상)
# ============================================================================
class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss
    - 소수 클래스에 더 큰 마진 적용
    - weight 파라미터로 클래스별 손실 가중치 지원
    """

    def __init__(self, cls_num_list, max_m=0.5, s=30, weight=None):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.FloatTensor(m_list)
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.float()
        batch_m = torch.matmul(
            self.m_list[None, :].to(x.device), index_float.transpose(0, 1).to(x.device)
        )
        batch_m = batch_m.view((-1, 1))

        x_m = x - batch_m
        output = torch.where(index.bool(), x_m, x)

        return F.cross_entropy(self.s * output, target, weight=self.weight)


# ============================================================================
# 4. Mixup Data Augmentation
# ============================================================================
def mixup_data(x1, x2, x3, y, alpha=1.0, device="cuda"):
    """Mixup: 두 샘플을 섞어서 새로운 샘플 생성"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    mixed_x3 = lam * x3 + (1 - lam) * x3[index, :]

    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, mixed_x3, y_a, y_b, lam


# ============================================================================
# 5. 데이터 증강 및 데이터셋
# ============================================================================
def get_train_transform(use_augmentation=True):
    if use_augmentation:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                # [강화] 회전 및 스케일 변화 폭 증가
                A.Affine(
                    translate_percent={
                        "x": (-0.1, 0.1),
                        "y": (-0.1, 0.1),
                    },  # 0.05 -> 0.1
                    scale=(0.9, 1.1),  # [수정] 0.8~1.2 -> 0.9~1.1 (자세 크기 특징 보존)
                    rotate=(-20, 20),  # -10~10 -> -20~20
                    p=0.5,
                ),
                # [강화] 색상 변화 강하게 (조명 조건 무시하도록)
                A.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5
                ),
                A.CoarseDropout(
                    num_holes_range=(4, 8),
                    hole_height_range=(10, 20),
                    hole_width_range=(10, 20),
                    fill=0,
                    p=0.3,
                ),
                A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            keypoint_params=A.KeypointParams(
                format="xy", label_fields=["keypoint_indices"], remove_invisible=False
            ),
        )
    else:
        return get_val_transform()


def get_val_transform():
    return A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(
            format="xy", label_fields=["keypoint_indices"], remove_invisible=False
        ),
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class HierarchicalDataset(Dataset):
    """계층적 분류를 위한 데이터셋"""

    def __init__(
        self,
        pkl_path: Path,
        stage: str = "binary",
        mode: str = "train",
        transform=None,
        window_size: int = 150,
        fps: float = 5.0,
    ):
        self.stage = stage
        self.mode = mode
        self.transform = transform
        self.window_size = window_size
        self.feature_extractor = PetFeatureExtractor(fps=fps, include_temporal=True)
        self.num_features = self.feature_extractor.num_features

        self.label_map = (
            BINARY_EMOTION_MAP if stage == "binary" else NEGATIVE_EMOTION_MAP
        )

        print(f"[{mode.upper()}] 데이터 로드 중: {pkl_path}")
        with open(pkl_path, "rb") as f:
            raw_data = pickle.load(f)

        self.samples = self._process_data(raw_data)
        print(f"  -> 로드 완료: {len(self.samples)} 샘플")

    def _process_data(self, raw_data):
        processed = []

        # 클래스별 데이터 임시 저장
        class_samples = {k: [] for k in self.label_map.values()}

        for item in raw_data:
            emo = item["emotion"]
            if self.stage == "binary":
                if emo not in self.label_map:
                    continue
            else:
                if emo not in NEGATIVE_EMOTIONS:
                    continue
                if emo not in self.label_map:
                    continue

            sample = {
                "video_id": item["video_id"],
                "img_paths": item["image_paths"],
                "keypoints": item["keypoints"],
                "video_size": item.get("video_size", (1280, 720)),
                "label": self.label_map[emo],
                "emotion": emo,
            }
            processed.append(sample)
            class_samples[self.label_map[emo]].append(sample)

        # [수정] Train 모드이고 Negative Stage일 때, 소수 클래스(0번) 강제 복제
        if self.mode == "train" and self.stage == "negative":
            # Class 0 (공포) 데이터 가져오기
            target_class = 0
            minority_samples = class_samples[target_class]

            # 목표 개수: 가장 많은 클래스(화남/불쾌) 데이터 수의 50% 수준까지 증량
            majority_count = max([len(v) for v in class_samples.values()])
            target_count = int(majority_count * 0.5)

            current_count = len(minority_samples)
            if current_count > 0 and current_count < target_count:
                # 부족한 만큼 랜덤하게 복제하여 추가
                num_to_add = target_count - current_count
                print(
                    f"♻️ Oversampling: Class {target_class} ({current_count} -> {target_count}개로 증강)"
                )

                # 단순 복제가 아니라 랜덤하게 뽑아서 추가
                additional_samples = random.choices(minority_samples, k=num_to_add)
                processed.extend(additional_samples)

        return processed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        total_frames = len(item["img_paths"])
        win_size = self.window_size

        if total_frames <= win_size:
            indices = (list(range(total_frames)) * (win_size // total_frames + 1))[
                :win_size
            ]
            indices.sort()
        else:
            start_idx = (
                random.randint(0, total_frames - win_size)
                if self.mode == "train"
                else (total_frames - win_size) // 2
            )
            indices = list(range(start_idx, start_idx + win_size))

        mid_idx = indices[len(indices) // 2]
        try:
            image = cv2.imread(str(item["img_paths"][mid_idx]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w = image.shape[:2]
        except:
            image, img_h, img_w = np.zeros((224, 224, 3), dtype=np.uint8), 224, 224

        kpts_seq = item["keypoints"][indices]  # (Window, 15, 2)

        # [CRITICAL FIX] 시퀀스 전체 키포인트 증강 (Spatial Consistency)
        # 이미지에 Affine(회전/이동)이 걸리면 시퀀스 전체도 똑같이 돌아가야 함.
        # 중간 프레임만 돌리면 데이터 불일치 발생.

        # 1. 정규화된 좌표 -> 픽셀 좌표 변환 (전체 시퀀스)
        kpts_seq_pixel = kpts_seq.copy()
        kpts_seq_pixel[:, :, 0] *= img_w
        kpts_seq_pixel[:, :, 1] *= img_h

        # 2. Flatten for Albumentations (N*15, 2)
        flat_kpts = kpts_seq_pixel.reshape(-1, 2)
        kpts_list = [tuple(kp) for kp in flat_kpts]

        if self.transform:
            # 이미지와 "전체" 키포인트 시퀀스를 한 번에 변환
            transformed = self.transform(
                image=image,
                keypoints=kpts_list,
                keypoint_indices=list(range(len(kpts_list))),
            )
            image_tensor = transformed["image"]

            # 3. 변환된 키포인트 복원
            if len(transformed["keypoints"]) > 0:
                aug_kpts_pixel = np.array(transformed["keypoints"]).reshape(
                    len(indices), 15, 2
                )

                # 다시 정규화 (0~1)
                aug_kpts_norm = np.zeros_like(aug_kpts_pixel, dtype=np.float32)
                aug_kpts_norm[:, :, 0] = aug_kpts_pixel[:, :, 0] / 224.0
                aug_kpts_norm[:, :, 1] = aug_kpts_pixel[:, :, 1] / 224.0
                aug_kpts_norm = np.clip(aug_kpts_norm, 0.0, 1.0)

                # 업데이트된 시퀀스로 교체
                kpts_seq = aug_kpts_norm

            # [Note] 이미 aug_kpts_norm에 Flip/Rot/Scale이 다 적용되어 있으므로
            # 별도의 flip_keypoints_horizontal 호출은 불필요 (중복 적용 금지)
        else:
            image_tensor = torch.zeros(3, 224, 224)

        features = self.feature_extractor.extract(kpts_seq)
        return (
            image_tensor,
            torch.tensor(kpts_seq, dtype=torch.float32),
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(item["label"], dtype=torch.long),
        )

    def get_labels(self):
        return [s["label"] for s in self.samples]


# ============================================================================
# 7. 학습/검증 함수
# ============================================================================
def get_weighted_sampler(dataset):
    labels = dataset.get_labels()
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def train_one_epoch(
    model, loader, criterion, optimizer, device, scaler=None, mixup_prob=0.5
):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    use_amp = scaler is not None and device.type == "cuda"

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, keypoints, features, labels in pbar:
        images, keypoints, features, labels = (
            images.to(device),
            keypoints.to(device),
            features.to(device),
            labels.to(device),
        )
        optimizer.zero_grad()

        use_mixup = np.random.rand() < mixup_prob
        if use_mixup:
            images, keypoints, features, targets_a, targets_b, lam = mixup_data(
                images, keypoints, features, labels, alpha=MIXUP_ALPHA, device=device
            )

        if use_amp:
            with autocast(device_type="cuda"):
                outputs = model(images, keypoints, features)
                loss = (
                    lam * criterion(outputs, targets_a)
                    + (1 - lam) * criterion(outputs, targets_b)
                    if use_mixup
                    else criterion(outputs, labels)
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, keypoints, features)
            loss = (
                lam * criterion(outputs, targets_a)
                + (1 - lam) * criterion(outputs, targets_b)
                if use_mixup
                else criterion(outputs, labels)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=loss.item(), mixup=use_mixup)

    metrics = calculate_metrics(
        np.array(all_labels), np.array(all_preds), average="macro"
    )
    return running_loss / len(loader.dataset), metrics["accuracy"], metrics["f1"]


def validate(model, loader, criterion, device, stage="binary"):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for images, keypoints, features, labels in tqdm(
            loader, desc="Val", leave=False
        ):
            images, keypoints, features, labels = (
                images.to(device),
                keypoints.to(device),
                features.to(device),
                labels.to(device),
            )
            outputs = model(images, keypoints, features)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    metrics = calculate_metrics(y_true, y_pred, average="macro")

    if stage == "binary":
        neg_mask = y_true == 1
        neg_recall = (
            (y_pred[neg_mask] == 1).sum() / neg_mask.sum()
            if neg_mask.sum() > 0
            else 0.0
        )
    else:
        neg_recall = metrics["f1"]

    return (
        running_loss / len(loader.dataset),
        metrics["accuracy"],
        metrics["f1"],
        neg_recall,
        (y_true, y_pred),
    )


# ============================================================================
# 8. 단계별 학습 함수
# ============================================================================
def train_stage(species: str, stage: str, config: dict, device: torch.device):
    stage_name = "Stage1_Binary" if stage == "binary" else "Stage2_Negative"
    class_names = BINARY_CLASS_NAMES if stage == "binary" else NEGATIVE_CLASS_NAMES

    print(
        f"\n{'='*70}\n[{species.upper()}] {stage_name} 학습 시작\n  - LDAM Loss + Mixup + 2-Stage Training\n{'='*70}"
    )

    train_ds = HierarchicalDataset(
        DATA_DIR / f"{species.lower()}_train.pkl",
        stage,
        "train",
        get_train_transform(config["use_augmentation"]),
        config["window_size"],
        config["fps"],
    )
    val_ds = HierarchicalDataset(
        DATA_DIR / f"{species.lower()}_val.pkl",
        stage,
        "val",
        get_val_transform(),
        config["window_size"],
        config["fps"],
    )
    if len(train_ds) == 0:
        return None

    train_labels = train_ds.get_labels()
    cls_num_list = [train_labels.count(i) for i in range(config["num_classes"])]
    print(f"📊 클래스 분포 (LDAM용): {dict(zip(class_names, cls_num_list))}")

    sampler = get_weighted_sampler(train_ds)
    g = torch.Generator()
    g.manual_seed(config["seed"])
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        sampler=sampler,
        num_workers=config["num_workers"],
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    model = HybridPetNet(
        num_classes=config["num_classes"],
        num_domain_features=train_ds.num_features,
        dropout=0.4,
        fusion_type="bottleneck",
    ).to(device)

    class_weights = (
        torch.tensor(config["class_weights"], dtype=torch.float32).to(device)
        if config.get("class_weights")
        else None
    )
    if class_weights is not None:
        print(
            f"🔨 [{stage.capitalize()}] Sampler ON + Weight ON: {config['class_weights']}"
        )

    # [이중 가중치 방지 + Margin 강화]
    # "불안"과 "화남" 구분 위해 max_m=0.8로 상향 (기본 0.5)
    # Loss의 weight는 None으로 설정 (Sampler가 이미 균형 맞춤)
    criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.8, weight=None)
    # criterion = nn.CrossEntropyLoss(
    #     weight=class_weights,
    #     label_smoothing=0.1,  # 정답 확률을 1.0이 아닌 0.9로 설정 (불확실성 허용)
    # )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["stage1_lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    scaler = GradScaler() if device.type == "cuda" else None

    best_score, early_stop_counter, stage2_activated, switch_to_stage2 = (
        0.0,
        0,
        False,
        False,
    )
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "stage_markers": [],
    }

    for epoch in range(1, config["epochs"] + 1):
        if not stage2_activated and epoch <= config["warmup_epochs"]:
            warmup_lr = config["stage1_lr"] * (epoch / config["warmup_epochs"])
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        if switch_to_stage2 and not stage2_activated:
            print(
                f"\n{'='*50}\n🔓 [Internal Stage 2] Unfreezing Image Encoder\n{'='*50}"
            )
            model.image_encoder.unfreeze()
            optimizer = optim.AdamW(
                [
                    {
                        "params": model.image_encoder.parameters(),
                        "lr": config["stage2_image_lr"],
                    },
                    {
                        "params": model.keypoint_encoder.parameters(),
                        "lr": config["stage2_keypoint_lr"],
                    },
                    {
                        "params": model.classifier.parameters(),
                        "lr": config["stage2_classifier_lr"],
                    },
                ],
                weight_decay=config["weight_decay"],
            )
            stage2_activated, switch_to_stage2, early_stop_counter = (
                True,
                False,
                0,
            )
            # best_score = 0.0  # [FIX] Stage 1의 Best Score를 유지하여 성능 하락 시 덮어쓰기 방지
            history["stage_markers"].append(epoch)

        t_loss, t_acc, t_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            config.get("mixup_prob", 0.0),
        )
        v_loss, v_acc, v_f1, neg_recall, (y_true, y_pred) = validate(
            model, val_loader, criterion, device, stage
        )

        recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
        current_score = (
            (recalls[0] * recalls[1]) ** 0.5
            if stage == "binary"
            else np.prod(recalls) ** (1 / len(recalls))
        )
        # [Log Enhancement] Show Min-Recall to confirm 50/50/50 balance
        min_recall = np.min(recalls)
        log_msg = (
            f"⚖️ G-Mean: {current_score:.4f} (Min:{min_recall:.2f}) | "
            + " | ".join([f"C{i}:{r:.2f}" for i, r in enumerate(recalls)])
        )
        print(
            f"Epoch {epoch:3d} [{'S2' if stage2_activated else 'S1'}] | Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f} | {log_msg} (ES: {early_stop_counter}/{config['early_stop_patience']})"
        )

        # [Target Check] 모든 클래스 Recall 60% 이상이면 즉시 종료 (60/60 대각행렬 목표 - 반올림 고려 0.595)
        target_reached = min_recall >= 0.595

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["train_f1"].append(t_f1)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["val_f1"].append(v_f1)

        scheduler.step(current_score)

        # Best Score 갱신 또는 목표 달성 시 저장
        if current_score > best_score or target_reached:
            best_score = current_score
            early_stop_counter = 0

            torch.save(
                {"model_state_dict": model.state_dict(), "config": config},
                CHECKPOINT_DIR / f"hierarchical_{species}_{stage}.pth",
            )
            plot_confusion_matrix(
                y_true,
                y_pred,
                class_names=class_names,  # classes -> class_names 수정
                save_path=RESULTS_DIR / f"confusion_matrix_{species}_{stage}.png",
            )

            if target_reached:
                print(
                    f"  🚀 [TARGET REACHED] 목표 달성! (Min Recall {min_recall:.2f} >= 0.60) -> 저장 후 학습 종료"
                )
                plot_training_history(
                    history, RESULTS_DIR / f"history_{species}_{stage}.png"
                )
                return best_score
            else:
                print(f"  🔥 Best Model Saved! (G-Mean: {best_score:.4f})")
        else:
            early_stop_counter += 1

        if early_stop_counter >= config["early_stop_patience"]:
            if not stage2_activated:
                switch_to_stage2 = True
            else:
                print("🛑 Stage 2 Early Stopping -> 학습 종료")
                break

    plot_training_history(history, RESULTS_DIR / f"history_{species}_{stage}.png")
    return best_score


# ============================================================================
# 9. 메인 실행
# ============================================================================
def train_hierarchical(species: str = "cat"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'#'*70}\n# 계층적 감정 분류 모델 학습: {species.upper()}\n{'#'*70}")

    stage1_f1 = None # train_stage(species, "binary", STAGE1_CONFIG, device)
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    stage2_f1 = train_stage(species, "negative", STAGE2_CONFIG, device)

    print(
        f"\n📊 [{species.upper()}] 학습 완료 요약\n"
        + f"  Binary: {stage1_f1 if stage1_f1 else '스킵됨'}\n"
        + f"  Negative: {stage2_f1}\n"
    )


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    for species in ["dog"]:
        train_hierarchical(species)
