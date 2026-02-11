"""
균형 잡힌 데이터셋 기반 YOLO-Pose 학습 스크립트
- 감정 불균형을 해소한 데이터셋으로 YOLO26-Pose 모델을 학습합니다.
- balanced_yolo_dataset/data.yaml 설정을 기반으로 학습이 진행됩니다.
- RTX 4060 8GB VRAM 환경에 최적화되었습니다.
"""

import argparse
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def run_training(epochs=100, batch=-1, imgsz=640, device="0"):
    """
    균형 잡힌 데이터셋으로 YOLO-Pose 학습 실행
    """
    print("\n" + "=" * 60)
    print("균형 잡힌 데이터셋 기반 YOLO-Pose 학습")
    print("=" * 60)

    # 1. 데이터 설정 확인
    current_ai_dir = Path(__file__).resolve().parent
    data_yaml = current_ai_dir / "balanced_yolo_dataset" / "data.yaml"
    if not data_yaml.exists():
        print(f"❌ 설정 파일을 찾을 수 없습니다: {data_yaml}")
        print("먼저 src/ai/03_prepare_balanced_yolo_data.py를 실행하세요.")
        return

    # 2. 출력 경로 설정 (training_results 내)
    output_dir = current_ai_dir / "training_results" / "balanced_yolo_pose"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. 모델 로드
    model_variant = "yolo26n-pose.pt"
    # 모델 파일도 src/ai 내에서 찾음 (07번 코드와 동일)
    model_path = current_ai_dir / model_variant
    print(f"🔧 모델 불러오는 중: {model_path}")
    if not model_path.exists():
         # 없으면 다운로드? or fail. YOLO will download if not found locally, but let's point to it if valid.
         model = YOLO(model_variant) # Fallback to standard download if file missing
    else:
         model = YOLO(str(model_path))

    # 4. 커스텀 얼리 스탑핑 콜백 등록 (가 성능 0.85 이상 시 3에포크 정체되면 중단)
    def custom_early_stopping(trainer):
        # 성능 지표 가져오기 (Box mAP50 또는 Pose mAP50 중 높은 것 기준)
        map50_b = trainer.metrics.get("metrics/mAP50(B)", 0.0)
        map50_p = trainer.metrics.get("metrics/mAP50(P)", 0.0)
        current_perf = max(map50_b, map50_p)

        threshold = 0.85
        patience = 3  # 3~5 에포크 중 3 에포크 선택

        if not hasattr(trainer, "custom_best_perf"):
            trainer.custom_best_perf = 0.0
            trainer.custom_patience_count = 0

        if current_perf >= threshold:
            # 유의미한 향상(0.001 이상)이 있는지 확인
            if current_perf > trainer.custom_best_perf + 0.001:
                trainer.custom_best_perf = current_perf
                trainer.custom_patience_count = 0
            else:
                trainer.custom_patience_count += 1
                print(
                    f"\n[Custom ES] 성능 {current_perf:.4f} >= {threshold} 도달. "
                    f"정체 상태: {trainer.custom_patience_count}/{patience} 에포크"
                )

            if trainer.custom_patience_count >= patience:
                print(
                    f"\n[Custom ES] 성능이 {threshold} 이상에서 {patience} 에포크 동안 개선되지 않아 조기 종료합니다."
                )
                trainer.stop = True

    model.add_callback("on_train_epoch_end", custom_early_stopping)

    # 5. 장치 확인 (GPU/CPU)
    if device != "cpu" and not torch.cuda.is_available():
        print("⚠️ 가용 가능한 GPU가 없어 CPU로 전환합니다.")
        device = "cpu"

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU: {gpu_name} ({gpu_mem:.1f}GB)")

    # 5. 학습 시작
    print(f"\n🚀 학습 파라미터:")
    print(f"  - 에포크: {epochs}")
    print(f"  - 배치: {'Auto (60% VRAM)' if batch == -1 else batch}")
    print(f"  - 이미지 크기: {imgsz}")
    print(f"  - 장치: {device}")

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        # ===== 메모리 최적화 (핵심) =====
        batch=batch,  # RTX 4060 8GB 기준 64 권장
        workers=2,  # 데이터 로딩 속도 향상
        cache=False,  # RAM 캐싱 비활성화
        rect=True,  # [메모리 효율] 패딩 최소화
        # ===== 학습 효율 및 성능 =====
        amp=True,  # [GPU 효율] FP16 연산으로 VRAM 사용량 30% 감소
        # ===== 학습률 스케줄러 =====
        cos_lr=True,  # [수렴 안정성] 코사인 스케줄러
        lr0=0.001,  # AdamW와 궁합이 좋은 학습률
        lrf=0.01,  # 최종 학습률 = 0.00001
        warmup_epochs=3.0,  # 초기 불안정 방지
        # ===== 증강 전략 =====
        mosaic=1.0,  # 모자이크
        mixup=0.1,  # 믹스업
        fliplr=0.5,  # 좌우 반전
        scale=0.5,  # 크기 변화
        close_mosaic=10,  # 마지막 10 epoch에서 모자이크 끄기
        # ===== 손실 함수 가중치 =====
        box=7.5,
        cls=1.0,
        pose=15.0,
        kobj=1.0,
        # ===== 최적화 및 정규화 =====
        optimizer="AdamW",
        weight_decay=0.01,
        # ===== 기타 =====
        imgsz=imgsz,
        device=device,
        project=str(output_dir),
        name="pet_pose_balanced",
        exist_ok=True,
        plots=True,
        save=True,
        save_period=10,
        patience=10,
        verbose=False,
    )

    print("\n" + "=" * 60)
    print("✅ 학습이 완료되었습니다!")
    print(f"결과 위치: {output_dir}/pet_pose_balanced")
    print(f"Best 모델: {output_dir}/pet_pose_balanced/weights/best.pt")
    print("=" * 60)


def run_validation(weights_path=None, split="test", save_json=False, plots=True):
    """
    학습된 모델 검증 실행
    """
    print("\n" + "=" * 60)
    print("YOLO-Pose 모델 검증")
    print("=" * 60)

    # 1. 데이터 설정
    current_ai_dir = Path(__file__).resolve().parent
    data_yaml = current_ai_dir / "balanced_yolo_dataset" / "data.yaml"
    if not data_yaml.exists():
        print(f"❌ 설정 파일을 찾을 수 없습니다: {data_yaml}")
        return

    # 2. 가중치 파일 확인
    if weights_path is None:
        weights_path = (
            current_ai_dir
            / "training_results"
            / "balanced_yolo_pose"
            / "pet_pose_balanced"
            / "weights"
            / "best.pt"
        )

    if not Path(weights_path).exists():
        print(f"❌ 가중치 파일을 찾을 수 없습니다: {weights_path}")
        print("먼저 학습을 실행하세요.")
        return

    print(f"📦 가중치 로드: {weights_path}")

    # 3. 모델 로드
    model = YOLO(str(weights_path))

    # 4. 검증 실행
    print(f"🔍 검증 세트: {split}")
    metrics = model.val(
        data=str(data_yaml),
        split=split,  # 'val', 'test', 'train' 중 선택
        # ===== 검증 설정 =====
        imgsz=640,
        batch=8,  # 검증 시에는 고정 배치 사용
        conf=0.001,  # PR 곡선 계산을 위한 낮은 임계값
        iou=0.7,  # NMS IoU 임계값
        max_det=300,  # 이미지당 최대 검출 수
        # ===== 성능 최적화 =====
        half=True,  # FP16으로 속도 향상
        workers=4,  # 데이터 로딩 worker
        rect=True,  # 직사각형 추론 (패딩 감소)
        # ===== 출력 설정 =====
        plots=plots,  # 혼동 행렬, PR 곡선 등 플롯 생성
        save_json=save_json,  # JSON으로 결과 저장
        verbose=False,
    )

    # 5. 결과 출력
    print("\n" + "=" * 60)
    print("📊 검증 결과")
    print("=" * 60)
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP75:    {metrics.box.map75:.4f}")

    # Pose 메트릭 (YOLO-Pose 전용)
    if hasattr(metrics, "pose") and metrics.pose:
        print(f"\n[Pose 메트릭]")
        print(f"Pose mAP50-95: {metrics.pose.map:.4f}")
        print(f"Pose mAP50:    {metrics.pose.map50:.4f}")

    print("=" * 60)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Balanced Pet Pose Estimation Training & Validation"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "val"],
        help="실행 모드 (train/val)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Epoch count (학습 시)")
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--imgsz", type=str, default="640", help="Image size")
    parser.add_argument("--device", type=str, default="0", help="Device (0 or 'cpu')")
    parser.add_argument(
        "--weights", type=str, default=None, help="모델 가중치 경로 (검증 시)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test", "train"],
        help="검증 데이터 분할",
    )
    parser.add_argument(
        "--save-json", action="store_true", help="검증 결과를 JSON으로 저장"
    )
    parser.add_argument("--no-plots", action="store_true", help="플롯 생성 비활성화")

    args = parser.parse_args()

    # imgsz가 문자열로 들어올 수 있으므로 int 변환
    img_size = int(args.imgsz)

    if args.mode == "train":
        run_training(
            epochs=args.epochs, batch=args.batch, imgsz=img_size, device=args.device
        )
    else:  # val
        run_validation(
            weights_path=args.weights,
            split=args.split,
            save_json=args.save_json,
            plots=not args.no_plots,
        )
