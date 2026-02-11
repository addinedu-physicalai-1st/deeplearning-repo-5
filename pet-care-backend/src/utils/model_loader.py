"""모델 로더 - v3 경로 구조에 맞게 새로 작성"""
import os
import logging
import torch
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ModelLoader:
    """YOLO 탐지/포즈 모델 + HybridPetNet 감정 모델 일괄 로딩"""

    def __init__(self, config: dict):
        self.config = config
        self.project_root = Path(config.get("project_root", "."))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.yolo_model = None
        self.pose_model = None
        self.emotion_models: dict = {}
        self.is_loaded = False

    def load_all_models(self):
        logger.info("모델 로딩 시퀀스 시작...")
        self._load_yolo()
        self._load_pose()
        self._load_emotion_models()
        self.is_loaded = True
        logger.info("모든 모델 로딩 완료.")

    def _load_yolo(self):
        """YOLO 탐지 모델 (yolo26s.pt)"""
        model_name = self.config["model"]["yolo"]["name"]
        model_path = self.project_root / model_name
        logger.info(f"YOLO 탐지 모델 로딩: {model_path}")
        try:
            self.yolo_model = YOLO(str(model_path))
            logger.info(f"YOLO 모델 로드 완료: {model_name}")
        except Exception as e:
            logger.error(f"YOLO 모델 로드 실패: {e}")

    def _load_pose(self):
        """포즈 추정 모델 (best.pt)"""
        pose_path = self.project_root / self.config["model"]["pose"]["path"]
        logger.info(f"포즈 모델 로딩: {pose_path}")
        try:
            self.pose_model = YOLO(str(pose_path))
            logger.info(f"포즈 모델 로드 완료. 클래스: {self.pose_model.names}")
        except Exception as e:
            logger.error(f"포즈 모델 로드 실패: {e}")
            raise

    def _load_emotion_models(self):
        """HybridPetNet 감정 모델 4개 로드"""
        from src.models.multimodal_net import HybridPetNet

        ckpt_cfg = self.config["model"]["checkpoints"]
        ckpt_dir = self.project_root / ckpt_cfg["dir"]
        logger.info(f"감정 모델 로딩 경로: {ckpt_dir}")

        model_specs = [
            ("cat_bin", ckpt_cfg["cat_binary"], 2),
            ("cat_neg", ckpt_cfg["cat_negative"], 3),
            ("dog_bin", ckpt_cfg["dog_binary"], 2),
            ("dog_neg", ckpt_cfg["dog_negative"], 3),
        ]

        for key, filename, num_classes in model_specs:
            path = ckpt_dir / filename
            if not path.exists():
                logger.warning(f"체크포인트 없음: {path}")
                continue
            try:
                model = HybridPetNet(
                    num_classes=num_classes,
                    num_domain_features=50,
                    dropout=0.5,
                    fusion_type="bottleneck",
                ).to(self.device)

                sd = torch.load(str(path), map_location=self.device)
                if "model_state_dict" in sd:
                    sd = sd["model_state_dict"]
                model.load_state_dict(sd)
                model.eval()
                self.emotion_models[key] = model
                logger.info(f"감정 모델 로드 완료: {key}")
            except Exception as e:
                logger.error(f"감정 모델 로드 실패 ({key}): {e}")

        logger.info(f"로드된 감정 모델: {list(self.emotion_models.keys())}")
