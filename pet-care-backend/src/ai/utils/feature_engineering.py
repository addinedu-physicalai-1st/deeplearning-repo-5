"""
반려동물 행동/감정 분석을 위한 도메인 지식 기반 피처 엔지니어링 모듈

데이터셋 명세서 기준 Keypoint 정의 (1-indexed -> 0-indexed 변환):
  0 (1):  코 (nose)
  1 (2):  이마 정중앙 (forehead_center)
  2 (3):  입꼬리/입끝 (mouth_corner)
  3 (4):  아래 입술 중앙 (lower_lip_center)
  4 (5):  목 (neck)
  5 (6):  앞다리 오른쪽 시작 (right_front_leg_start / 오른쪽 어깨)
  6 (7):  앞다리 왼쪽 시작 (left_front_leg_start / 왼쪽 어깨)
  7 (8):  앞다리 오른쪽 발목 (right_front_ankle)
  8 (9):  앞다리 왼쪽 발목 (left_front_ankle)
  9 (10): 오른쪽 대퇴골 (right_hip / 오른쪽 엉덩이)
  10 (11): 왼쪽 대퇴골 (left_hip / 왼쪽 엉덩이)
  11 (12): 뒷다리 오른쪽 발목 (right_back_ankle)
  12 (13): 뒷다리 왼쪽 발목 (left_back_ankle)
  13 (14): 꼬리 시작 (tail_start)
  14 (15): 꼬리 끝 (tail_end)

주요 피처 카테고리:
1. 얼굴/머리 영역 피처 (Head Features) - 코, 이마, 입 관련
2. 자세/척추 피처 (Posture Features) - 목, 어깨, 엉덩이 관련
3. 앞다리 동작 피처 (Front Leg Features)
4. 뒷다리 동작 피처 (Back Leg Features)
5. 꼬리 피처 (Tail Features) - 꼬리 시작+끝 활용
6. 시계열 동적 피처 (Temporal Features)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# ============================================================================
# Keypoint 인덱스 상수 정의 (데이터셋 명세서 기준 - 0-indexed)
# ============================================================================
KP_NOSE = 0  # 코
KP_FOREHEAD = 1  # 이마 정중앙
KP_MOUTH_CORNER = 2  # 입꼬리(입끝)
KP_LOWER_LIP = 3  # 아래 입술 중앙
KP_NECK = 4  # 목
KP_R_FRONT_LEG_START = 5  # 앞다리 오른쪽 시작 (오른쪽 어깨)
KP_L_FRONT_LEG_START = 6  # 앞다리 왼쪽 시작 (왼쪽 어깨)
KP_R_FRONT_ANKLE = 7  # 앞다리 오른쪽 발목
KP_L_FRONT_ANKLE = 8  # 앞다리 왼쪽 발목
KP_R_HIP = 9  # 오른쪽 대퇴골 (엉덩이)
KP_L_HIP = 10  # 왼쪽 대퇴골 (엉덩이)
KP_R_BACK_ANKLE = 11  # 뒷다리 오른쪽 발목
KP_L_BACK_ANKLE = 12  # 뒷다리 왼쪽 발목
KP_TAIL_START = 13  # 꼬리 시작
KP_TAIL_END = 14  # 꼬리 끝

# Keypoint 이름 매핑 (디버깅/시각화용)
KEYPOINT_NAMES = [
    "코",
    "이마_정중앙",
    "입꼬리",
    "아래_입술_중앙",
    "목",
    "앞다리_오른쪽_시작",
    "앞다리_왼쪽_시작",
    "앞다리_오른쪽_발목",
    "앞다리_왼쪽_발목",
    "오른쪽_대퇴골",
    "왼쪽_대퇴골",
    "뒷다리_오른쪽_발목",
    "뒷다리_왼쪽_발목",
    "꼬리_시작",
    "꼬리_끝",
]

# 좌우 대칭 Keypoint 쌍 (Horizontal Flip 시 교환 필요)
FLIP_PAIRS = [
    (KP_R_FRONT_LEG_START, KP_L_FRONT_LEG_START),  # 앞다리 시작 좌우
    (KP_R_FRONT_ANKLE, KP_L_FRONT_ANKLE),  # 앞다리 발목 좌우
    (KP_R_HIP, KP_L_HIP),  # 대퇴골(엉덩이) 좌우
    (KP_R_BACK_ANKLE, KP_L_BACK_ANKLE),  # 뒷다리 발목 좌우
]


# ============================================================================
# 유틸리티 함수
# ============================================================================
def safe_divide(a: np.ndarray, b: np.ndarray, default: float = 0.0) -> np.ndarray:
    """0으로 나누기 방지 (안전한 나눗셈)"""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(b != 0, a / b, default)
    return result


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """두 점 사이의 유클리드 거리 계산"""
    return np.sqrt(np.sum((p1 - p2) ** 2, axis=-1) + 1e-8)


def angle_between_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """
    세 점 p1-p2-p3 사이의 각도 계산 (p2가 꼭짓점)
    반환값: 라디안 [0, π]
    """
    v1 = p1 - p2
    v2 = p3 - p2

    cos_angle = np.sum(v1 * v2, axis=-1) / (
        np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8
    )
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def vector_angle(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    두 점으로 이루어진 벡터의 수평면 대비 각도 계산
    반환값: 라디안 [-π, π]
    """
    diff = p2 - p1
    return np.arctan2(diff[..., 1], diff[..., 0])


def temporal_derivative(arr: np.ndarray, fps: float = 5.0) -> np.ndarray:
    """
    시간에 따른 변화율(속도) 계산
    중심 차분 사용, 첫/끝 프레임은 전방/후방 차분 사용
    """
    if len(arr) < 2:
        return np.zeros_like(arr)

    dt = 1.0 / fps
    deriv = np.zeros_like(arr)

    # 중심 차분 (Central Difference)
    deriv[1:-1] = (arr[2:] - arr[:-2]) / (2 * dt)
    # 전방/후방 차분
    deriv[0] = (arr[1] - arr[0]) / dt
    deriv[-1] = (arr[-1] - arr[-2]) / dt

    return deriv


def temporal_second_derivative(arr: np.ndarray, fps: float = 5.0) -> np.ndarray:
    """시간에 따른 가속도 계산 (2차 미분)"""
    velocity = temporal_derivative(arr, fps)
    return temporal_derivative(velocity, fps)


# ============================================================================
# 피처 계산 클래스
# ============================================================================
class PetFeatureExtractor:
    """
    반려동물 Keypoint 기반 도메인 지식 피처 추출기

    입력: keypoints (T, 15, 2) - 정규화된 좌표 [0, 1]
    출력: features (T, num_features) - 시계열 피처
    """

    def __init__(self, fps: float = 5.0, include_temporal: bool = True):
        """
        Args:
            fps: 프레임 레이트 (기본값 5fps - 데이터셋 기준)
            include_temporal: 시계열 피처(속도, 가속도) 포함 여부
        """
        self.fps = fps
        self.include_temporal = include_temporal
        self._feature_names = None

    @property
    def feature_names(self) -> List[str]:
        """피처 이름 리스트 반환"""
        if self._feature_names is None:
            self._feature_names = self._get_feature_names()
        return self._feature_names

    @property
    def num_features(self) -> int:
        """총 피처 개수"""
        return len(self.feature_names)

    def _get_feature_names(self) -> List[str]:
        """모든 피처 이름 정의"""
        names = []

        # 1. 얼굴/머리 피처 (7개) - 코, 이마, 입 관련
        names.extend(
            [
                "head_height",  # 머리(코) 높이 (낮을수록 엎드린 자세)
                "head_tilt",  # 머리 기울기 (이마-코 연결선 각도)
                "face_length",  # 얼굴 길이 (이마-아래입술 거리)
                "mouth_openness",  # 입 벌림 정도 (입꼬리-아래입술 거리)
                "nose_to_neck_dist",  # 코-목 거리 (머리 내밀기 정도)
                "forehead_to_neck_angle",  # 이마-목 각도 (고개 들기/숙이기)
                "mouth_position",  # 입 상대 위치 (얼굴 중심 대비)
            ]
        )

        # 2. 자세/척추 피처 (7개) - 목, 어깨, 엉덩이 관련
        names.extend(
            [
                "spine_angle",  # 척추 각도 (목-꼬리시작 연결선)
                "body_length",  # 몸 길이 (목-꼬리시작 거리)
                "body_height",  # 몸 높이 (어깨 평균 y좌표)
                "shoulder_width",  # 어깨 너비 (앞다리 시작점 간 거리)
                "hip_width",  # 엉덩이 너비 (대퇴골 간 거리)
                "crouch_ratio",  # 웅크림 정도 (높이/길이 비율)
                "body_symmetry",  # 좌우 대칭도 (비대칭 시 불안 표현)
            ]
        )

        # 3. 앞다리 피처 (6개)
        names.extend(
            [
                "front_leg_spread",  # 앞다리(발목) 벌림 정도
                "front_leg_height_r",  # 오른쪽 앞발 높이 (들기 감지)
                "front_leg_height_l",  # 왼쪽 앞발 높이
                "front_leg_extension_r",  # 오른쪽 앞다리 뻗음 (어깨-발목 거리)
                "front_leg_extension_l",  # 왼쪽 앞다리 뻗음
                "front_leg_angle",  # 앞다리 각도 (어깨-발목 연결선)
            ]
        )

        # 4. 뒷다리 피처 (6개)
        names.extend(
            [
                "back_leg_spread",  # 뒷다리(발목) 벌림 정도
                "back_leg_height_r",  # 오른쪽 뒷발 높이
                "back_leg_height_l",  # 왼쪽 뒷발 높이
                "back_leg_extension_r",  # 오른쪽 뒷다리 뻗음 (엉덩이-발목 거리)
                "back_leg_extension_l",  # 왼쪽 뒷다리 뻗음
                "hindquarter_tuck",  # 뒷다리 움츠림 정도 (꼬리시작-뒷발 거리)
            ]
        )

        # 5. 꼬리 피처 (8개) - 꼬리시작+꼬리끝 활용
        names.extend(
            [
                "tail_start_height",  # 꼬리 시작점 높이
                "tail_end_height",  # 꼬리 끝 높이
                "tail_length",  # 꼬리 길이 (시작-끝 거리)
                "tail_angle",  # 꼬리 각도 (시작-끝 연결선의 수평 대비)
                "tail_elevation",  # 꼬리 들림 정도 (끝이 시작보다 높은 정도)
                "tail_curl",  # 꼬리 말림 (몸쪽으로 오는 정도)
                "tail_lateral_offset",  # 꼬리 좌우 치우침 (흔들림 프록시)
                "tail_spine_alignment",  # 꼬리-척추 정렬도
            ]
        )

        # 6. 종합 자세 피처 (5개)
        names.extend(
            [
                "overall_compactness",  # 전체적 웅크림 (keypoint 분산)
                "center_of_mass_x",  # 무게중심 x (좌우 치우침)
                "center_of_mass_y",  # 무게중심 y (상하 치우침)
                "pose_energy",  # 자세 에너지 (관절 간 거리 합)
                "standing_score",  # 서있는 정도 (발목 높이 평균)
            ]
        )

        # 7. 시계열 피처 (속도, 가속도)
        if self.include_temporal:
            temporal_base_names = [
                "head_velocity",  # 머리(코) 움직임 속도
                "tail_velocity",  # 꼬리끝 움직임 속도
                "tail_wag_speed",  # 꼬리 흔들림 속도 (좌우 변위)
                "tail_angular_velocity",  # 꼬리 각속도 (각도 변화율)
                "body_velocity",  # 몸통(목) 움직임 속도
                "front_paw_velocity",  # 앞발 평균 움직임 속도
                "back_paw_velocity",  # 뒷발 평균 움직임 속도
                "head_acceleration",  # 머리 가속도
                "tail_acceleration",  # 꼬리 가속도
                "motion_intensity",  # 전체 동작 강도
                "motion_consistency",  # 동작 일관성 (속도 표준편차)
            ]
            names.extend(temporal_base_names)

        return names

    def extract(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Keypoint에서 전체 피처 추출

        Args:
            keypoints: (T, 15, 2) 정규화된 좌표

        Returns:
            features: (T, num_features) 피처 배열
        """
        T = keypoints.shape[0]
        features = []

        # 1. 얼굴/머리 피처
        features.append(self._extract_head_features(keypoints))

        # 2. 자세/척추 피처
        features.append(self._extract_posture_features(keypoints))

        # 3. 앞다리 피처
        features.append(self._extract_front_leg_features(keypoints))

        # 4. 뒷다리 피처
        features.append(self._extract_back_leg_features(keypoints))

        # 5. 꼬리 피처
        features.append(self._extract_tail_features(keypoints))

        # 6. 종합 자세 피처
        features.append(self._extract_overall_features(keypoints))

        # 7. 시계열 피처
        if self.include_temporal:
            features.append(self._extract_temporal_features(keypoints))

        # 모든 피처 결합
        all_features = np.concatenate(features, axis=-1)

        # NaN/Inf 처리
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=1.0, neginf=-1.0)

        return all_features.astype(np.float32)

    def _extract_head_features(self, kpts: np.ndarray) -> np.ndarray:
        """얼굴/머리 영역 피처 (7개)"""
        T = kpts.shape[0]
        features = np.zeros((T, 7), dtype=np.float32)

        nose = kpts[:, KP_NOSE]
        forehead = kpts[:, KP_FOREHEAD]
        mouth_corner = kpts[:, KP_MOUTH_CORNER]
        lower_lip = kpts[:, KP_LOWER_LIP]
        neck = kpts[:, KP_NECK]

        # 1. 머리(코) 높이 - y좌표 (0=상단, 1=하단이므로 반전)
        features[:, 0] = 1.0 - nose[:, 1]

        # 2. 머리 기울기 (이마-코 연결선의 수평 대비 각도)
        features[:, 1] = vector_angle(forehead, nose)

        # 3. 얼굴 길이 (이마-아래입술 거리)
        features[:, 2] = euclidean_distance(forehead, lower_lip)

        # 4. 입 벌림 정도 (입꼬리-아래입술 거리)
        features[:, 3] = euclidean_distance(mouth_corner, lower_lip)

        # 5. 코-목 거리 (머리 내밀기 정도)
        features[:, 4] = euclidean_distance(nose, neck)

        # 6. 이마-목 각도 (고개 들기/숙이기)
        features[:, 5] = vector_angle(neck, forehead)

        # 7. 입 상대 위치 (얼굴 중심 대비 - 입이 아래로 처진 정도)
        face_center_y = (forehead[:, 1] + nose[:, 1]) / 2.0
        features[:, 6] = lower_lip[:, 1] - face_center_y

        return features

    def _extract_posture_features(self, kpts: np.ndarray) -> np.ndarray:
        """자세/척추 피처 (7개)"""
        T = kpts.shape[0]
        features = np.zeros((T, 7), dtype=np.float32)

        neck = kpts[:, KP_NECK]
        r_shoulder = kpts[:, KP_R_FRONT_LEG_START]
        l_shoulder = kpts[:, KP_L_FRONT_LEG_START]
        r_hip = kpts[:, KP_R_HIP]
        l_hip = kpts[:, KP_L_HIP]
        tail_start = kpts[:, KP_TAIL_START]

        shoulder_center = (r_shoulder + l_shoulder) / 2.0
        hip_center = (r_hip + l_hip) / 2.0

        # 1. 척추 각도 (목-꼬리시작 연결선의 수평 대비 각도)
        features[:, 0] = vector_angle(neck, tail_start)

        # 2. 몸 길이 (목-꼬리시작 거리)
        features[:, 1] = euclidean_distance(neck, tail_start)

        # 3. 몸 높이 (어깨 중심 y좌표 - 높을수록 서있음)
        features[:, 2] = 1.0 - shoulder_center[:, 1]

        # 4. 어깨 너비 (앞다리 시작점 간 거리)
        features[:, 3] = euclidean_distance(r_shoulder, l_shoulder)

        # 5. 엉덩이 너비 (대퇴골 간 거리)
        features[:, 4] = euclidean_distance(r_hip, l_hip)

        # 6. 웅크림 정도 (높이/길이 비율)
        body_height = features[:, 2]
        body_length = features[:, 1] + 1e-8
        features[:, 5] = safe_divide(body_height, body_length)

        # 7. 좌우 대칭도 (0=완전 대칭, 높을수록 비대칭)
        r_dist = euclidean_distance(neck, r_shoulder) + euclidean_distance(
            tail_start, r_hip
        )
        l_dist = euclidean_distance(neck, l_shoulder) + euclidean_distance(
            tail_start, l_hip
        )
        features[:, 6] = np.abs(r_dist - l_dist)

        return features

    def _extract_front_leg_features(self, kpts: np.ndarray) -> np.ndarray:
        """앞다리 피처 (6개)"""
        T = kpts.shape[0]
        features = np.zeros((T, 6), dtype=np.float32)

        r_shoulder = kpts[:, KP_R_FRONT_LEG_START]
        l_shoulder = kpts[:, KP_L_FRONT_LEG_START]
        r_ankle = kpts[:, KP_R_FRONT_ANKLE]
        l_ankle = kpts[:, KP_L_FRONT_ANKLE]

        # 1. 앞다리(발목) 벌림 정도
        features[:, 0] = euclidean_distance(r_ankle, l_ankle)

        # 2. 오른쪽 앞발 높이 (1 - y, 높을수록 들고 있음)
        features[:, 1] = 1.0 - r_ankle[:, 1]

        # 3. 왼쪽 앞발 높이
        features[:, 2] = 1.0 - l_ankle[:, 1]

        # 4. 오른쪽 앞다리 뻗음 (어깨-발목 거리)
        features[:, 3] = euclidean_distance(r_shoulder, r_ankle)

        # 5. 왼쪽 앞다리 뻗음
        features[:, 4] = euclidean_distance(l_shoulder, l_ankle)

        # 6. 앞다리 평균 각도 (수직 대비 - 서있으면 수직에 가까움)
        r_angle = vector_angle(r_shoulder, r_ankle)
        l_angle = vector_angle(l_shoulder, l_ankle)
        features[:, 5] = (r_angle + l_angle) / 2.0

        return features

    def _extract_back_leg_features(self, kpts: np.ndarray) -> np.ndarray:
        """뒷다리 피처 (6개)"""
        T = kpts.shape[0]
        features = np.zeros((T, 6), dtype=np.float32)

        r_hip = kpts[:, KP_R_HIP]
        l_hip = kpts[:, KP_L_HIP]
        r_ankle = kpts[:, KP_R_BACK_ANKLE]
        l_ankle = kpts[:, KP_L_BACK_ANKLE]
        tail_start = kpts[:, KP_TAIL_START]

        # 1. 뒷다리(발목) 벌림 정도
        features[:, 0] = euclidean_distance(r_ankle, l_ankle)

        # 2. 오른쪽 뒷발 높이
        features[:, 1] = 1.0 - r_ankle[:, 1]

        # 3. 왼쪽 뒷발 높이
        features[:, 2] = 1.0 - l_ankle[:, 1]

        # 4. 오른쪽 뒷다리 뻗음 (엉덩이-발목 거리)
        features[:, 3] = euclidean_distance(r_hip, r_ankle)

        # 5. 왼쪽 뒷다리 뻗음
        features[:, 4] = euclidean_distance(l_hip, l_ankle)

        # 6. 뒷다리 움츠림 (꼬리시작-뒷발 평균 거리)
        r_tuck = euclidean_distance(tail_start, r_ankle)
        l_tuck = euclidean_distance(tail_start, l_ankle)
        features[:, 5] = (r_tuck + l_tuck) / 2.0

        return features

    def _extract_tail_features(self, kpts: np.ndarray) -> np.ndarray:
        """꼬리 피처 (8개) - 꼬리시작 + 꼬리끝 활용"""
        T = kpts.shape[0]
        features = np.zeros((T, 8), dtype=np.float32)

        neck = kpts[:, KP_NECK]
        tail_start = kpts[:, KP_TAIL_START]
        tail_end = kpts[:, KP_TAIL_END]
        r_hip = kpts[:, KP_R_HIP]
        l_hip = kpts[:, KP_L_HIP]

        hip_center = (r_hip + l_hip) / 2.0

        # 1. 꼬리 시작점 높이 (1 - y)
        features[:, 0] = 1.0 - tail_start[:, 1]

        # 2. 꼬리 끝 높이
        features[:, 1] = 1.0 - tail_end[:, 1]

        # 3. 꼬리 길이 (시작-끝 거리)
        features[:, 2] = euclidean_distance(tail_start, tail_end)

        # 4. 꼬리 각도 (시작-끝 연결선의 수평 대비 각도)
        features[:, 3] = vector_angle(tail_start, tail_end)

        # 5. 꼬리 들림 정도 (끝이 시작보다 높은 정도, 양수면 올라감)
        features[:, 4] = tail_start[:, 1] - tail_end[:, 1]  # y가 작을수록 높음

        # 6. 꼬리 말림 (몸쪽으로 오는 정도 - 꼬리끝이 엉덩이에 가까운 정도)
        tail_to_hip = euclidean_distance(tail_end, hip_center)
        tail_length = features[:, 2] + 1e-8
        features[:, 5] = 1.0 - safe_divide(tail_to_hip, tail_length + tail_to_hip)

        # 7. 꼬리 좌우 치우침 (흔들림 프록시)
        body_center_x = (neck[:, 0] + tail_start[:, 0]) / 2.0
        features[:, 6] = tail_end[:, 0] - body_center_x

        # 8. 꼬리-척추 정렬도 (척추 연장선과 꼬리 각도 차이)
        spine_angle = vector_angle(neck, tail_start)
        tail_angle = features[:, 3]
        features[:, 7] = np.abs(spine_angle - tail_angle)

        return features

    def _extract_overall_features(self, kpts: np.ndarray) -> np.ndarray:
        """종합 자세 피처 (5개)"""
        T = kpts.shape[0]
        features = np.zeros((T, 5), dtype=np.float32)

        # 모든 Keypoint의 좌표
        all_x = kpts[:, :, 0]
        all_y = kpts[:, :, 1]

        # 발목 keypoint 인덱스
        ankle_indices = [
            KP_R_FRONT_ANKLE,
            KP_L_FRONT_ANKLE,
            KP_R_BACK_ANKLE,
            KP_L_BACK_ANKLE,
        ]

        # 1. 전체적 웅크림 (Keypoint 분산 - 작을수록 웅크림)
        x_std = all_x.std(axis=1)
        y_std = all_y.std(axis=1)
        features[:, 0] = x_std * y_std

        # 2. 무게중심 x (좌우 치우침, 0 = 중앙)
        features[:, 1] = all_x.mean(axis=1) - 0.5

        # 3. 무게중심 y (상하 치우침, 0 = 중앙)
        features[:, 2] = all_y.mean(axis=1) - 0.5

        # 4. 자세 에너지 (주요 관절 간 거리 합)
        neck = kpts[:, KP_NECK]
        tail_start = kpts[:, KP_TAIL_START]
        r_shoulder = kpts[:, KP_R_FRONT_LEG_START]
        l_shoulder = kpts[:, KP_L_FRONT_LEG_START]

        energy = euclidean_distance(neck, tail_start)
        energy += euclidean_distance(neck, r_shoulder)
        energy += euclidean_distance(neck, l_shoulder)
        features[:, 3] = energy

        # 5. 서있는 정도 (발목 높이 평균 - 높을수록 들고 있음/점프 중)
        ankle_heights = 1.0 - kpts[:, ankle_indices, 1]  # (T, 4)
        features[:, 4] = ankle_heights.mean(axis=1)

        return features

    def _extract_temporal_features(self, kpts: np.ndarray) -> np.ndarray:
        """시계열 동적 피처 (속도, 가속도, 흔들림 등) - 11개"""
        T = kpts.shape[0]
        features = np.zeros((T, 11), dtype=np.float32)

        # 주요 Keypoint 추출
        nose = kpts[:, KP_NOSE]
        neck = kpts[:, KP_NECK]
        tail_start = kpts[:, KP_TAIL_START]
        tail_end = kpts[:, KP_TAIL_END]
        r_f_ankle = kpts[:, KP_R_FRONT_ANKLE]
        l_f_ankle = kpts[:, KP_L_FRONT_ANKLE]
        r_b_ankle = kpts[:, KP_R_BACK_ANKLE]
        l_b_ankle = kpts[:, KP_L_BACK_ANKLE]

        # === 위치 변화 기반 속도 ===
        # 1. 머리(코) 움직임 속도
        nose_vel = temporal_derivative(nose, self.fps)
        features[:, 0] = np.linalg.norm(nose_vel, axis=-1)

        # 2. 꼬리끝 움직임 속도
        tail_end_vel = temporal_derivative(tail_end, self.fps)
        features[:, 1] = np.linalg.norm(tail_end_vel, axis=-1)

        # 3. 꼬리 흔들림 속도 (꼬리끝 x좌표 좌우 변위)
        body_center_x = (neck[:, 0] + tail_start[:, 0]) / 2.0
        tail_offset_x = tail_end[:, 0] - body_center_x
        features[:, 2] = np.abs(temporal_derivative(tail_offset_x, self.fps))

        # 4. 꼬리 각속도 (꼬리 각도의 시간 변화율)
        tail_angle = vector_angle(tail_start, tail_end)
        features[:, 3] = np.abs(temporal_derivative(tail_angle, self.fps))

        # 5. 몸통(목) 움직임 속도
        neck_vel = temporal_derivative(neck, self.fps)
        features[:, 4] = np.linalg.norm(neck_vel, axis=-1)

        # 6. 앞발 평균 움직임 속도
        front_center = (r_f_ankle + l_f_ankle) / 2.0
        front_vel = temporal_derivative(front_center, self.fps)
        features[:, 5] = np.linalg.norm(front_vel, axis=-1)

        # 7. 뒷발 평균 움직임 속도
        back_center = (r_b_ankle + l_b_ankle) / 2.0
        back_vel = temporal_derivative(back_center, self.fps)
        features[:, 6] = np.linalg.norm(back_vel, axis=-1)

        # === 가속도 ===
        # 8. 머리 가속도
        nose_acc = temporal_second_derivative(nose, self.fps)
        features[:, 7] = np.linalg.norm(nose_acc, axis=-1)

        # 9. 꼬리 가속도
        tail_acc = temporal_second_derivative(tail_end, self.fps)
        features[:, 8] = np.linalg.norm(tail_acc, axis=-1)

        # === 종합 동작 지표 ===
        # 10. 전체 동작 강도 (모든 Keypoint 속도 평균)
        all_vel = temporal_derivative(kpts, self.fps)  # (T, 15, 2)
        all_speed = np.linalg.norm(all_vel, axis=-1)  # (T, 15)
        features[:, 9] = all_speed.mean(axis=1)

        # 11. 동작 일관성 (속도 표준편차 - 낮을수록 일정한 동작)
        features[:, 10] = all_speed.std(axis=1)

        return features


def flip_keypoints_horizontal(keypoints: np.ndarray) -> np.ndarray:
    """
    Keypoint를 수평 반전

    Args:
        keypoints: (..., 15, 2) 정규화된 좌표 [0, 1]

    Returns:
        flipped: 반전된 좌표
    """
    flipped = keypoints.copy()

    # x 좌표 반전 (정규화된 경우 1 - x)
    flipped[..., 0] = 1.0 - flipped[..., 0]

    # 좌우 대칭 Keypoint 쌍 교환
    for left_idx, right_idx in FLIP_PAIRS:
        temp = flipped[..., left_idx, :].copy()
        flipped[..., left_idx, :] = flipped[..., right_idx, :]
        flipped[..., right_idx, :] = temp

    return flipped


# ============================================================================
# 테스트 코드
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PetFeatureExtractor 테스트 (수정된 Keypoint 정의)")
    print("=" * 70)

    print("\n[Keypoint 정의]")
    for i, name in enumerate(KEYPOINT_NAMES):
        print(f"  {i:2d}: {name}")

    # 더미 데이터 생성 (30 frames, 15 keypoints, 2 coords)
    np.random.seed(42)
    dummy_kpts = np.random.rand(30, 15, 2).astype(np.float32)

    # 피처 추출기 초기화
    extractor = PetFeatureExtractor(fps=5.0, include_temporal=True)

    # 피처 추출
    features = extractor.extract(dummy_kpts)

    print(f"\n입력 Keypoint Shape: {dummy_kpts.shape}")
    print(f"출력 Feature Shape: {features.shape}")
    print(f"총 피처 수: {extractor.num_features}")
    print(f"\n피처 목록 ({extractor.num_features}개):")
    for i, name in enumerate(extractor.feature_names):
        print(f"  [{i:2d}] {name}")

    # Flip 테스트
    print("\n" + "=" * 70)
    print("Horizontal Flip 테스트")
    print("=" * 70)

    print(f"\nFLIP_PAIRS: {FLIP_PAIRS}")

    flipped_kpts = flip_keypoints_horizontal(dummy_kpts)
    flipped_features = extractor.extract(flipped_kpts)

    print(
        f"\n원본 첫 프레임 앞다리_왼쪽_시작(6) x: {dummy_kpts[0, KP_L_FRONT_LEG_START, 0]:.4f}"
    )
    print(
        f"반전 후 앞다리_오른쪽_시작(5) x위치: {flipped_kpts[0, KP_R_FRONT_LEG_START, 0]:.4f}"
    )
    print(f"기대값 (1 - 원본): {1.0 - dummy_kpts[0, KP_L_FRONT_LEG_START, 0]:.4f}")

    print("\n✅ 테스트 완료!")
