"""
하이브리드 멀티모달 반려동물 행동/감정 분류 모델 (Improved Full Ver.)

[구조 설명]
1. ImageEncoder (EfficientNet-B0):
   - 비디오 프레임에서 시각적 특징 추출 (224x224 -> 1280 dim)
   - 전이 학습(Transfer Learning) 적용

2. KeypointEncoder (ST-GCN):
   - 15개 관절점의 시공간적(Spatio-Temporal) 움직임 패턴 학습
   - 반려동물 신체 구조(PetGraph) 기반 그래프 컨볼루션 적용

3. DomainFeatureProjector:
   - 도메인 지식 기반 피처(속도, 각도 등 50개)를 고차원으로 투영
   - 단순 수치 데이터를 모델이 이해하기 쉬운 형태로 변환 (128 dim)

4. Gated Fusion (Bottleneck):
   - 이미지, 모션, 도메인 피처를 결합
   - 각 모달리티의 중요도(Gate)를 동적으로 학습하여 가중치 조절

[작성일] 2026-02-09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


# ============================================================================
# 1. Graph Definition (반려동물 15 Keypoints 신체 구조)
# ============================================================================
class PetGraph:
    """
    반려동물(15 Keypoints) 신체 연결 구조 그래프 정의
    ST-GCN이 관절 간의 관계를 이해하도록 인접 행렬(Adjacency Matrix) 생성
    """

    def __init__(self, strategy="uniform"):
        self.num_node = 15
        self.center = 4  # 목(Neck)을 중심점으로 설정 (척추의 시작점)
        self.edges = self._get_edges()
        self.A = self._get_adjacency_matrix(strategy)

    def _get_edges(self):
        # 데이터셋 명세서 기준 신체 연결 구조
        edges = [
            # === 얼굴/머리 연결 ===
            (0, 1),  # 코 -> 이마
            (0, 2),  # 코 -> 입꼬리
            (2, 3),  # 입꼬리 -> 아래입술
            (1, 4),  # 이마 -> 목
            (0, 4),  # 코 -> 목 (직접 연결)
            # === 앞다리 연결 ===
            (4, 5),  # 목 -> 오른쪽 어깨(앞다리 시작)
            (4, 6),  # 목 -> 왼쪽 어깨
            (5, 7),  # 오른쪽 어깨 -> 오른쪽 앞발목
            (6, 8),  # 왼쪽 어깨 -> 왼쪽 앞발목
            # === 몸통/엉덩이 연결 ===
            (5, 9),  # 오른쪽 어깨 -> 오른쪽 엉덩이 (몸통)
            (6, 10), # 왼쪽 어깨 -> 왼쪽 엉덩이
            (9, 10), # 오른쪽 엉덩이 <-> 왼쪽 엉덩이 (골반 연결)
            # === 뒷다리 연결 ===
            (9, 11),  # 오른쪽 엉덩이 -> 오른쪽 뒷발목
            (10, 12), # 왼쪽 엉덩이 -> 왼쪽 뒷발목
            # === 꼬리 연결 ===
            (9, 13),  # 오른쪽 엉덩이 -> 꼬리 시작
            (10, 13), # 왼쪽 엉덩이 -> 꼬리 시작
            (13, 14), # 꼬리 시작 -> 꼬리 끝
        ]
        return edges

    def _get_adjacency_matrix(self, strategy):
        # 인접 행렬 생성 및 정규화 (Normalization)
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edges:
            A[i, j] = 1
            A[j, i] = 1
        
        # 정규화: D^(-1) * A
        Dl = np.sum(A, 0)
        Dn = np.zeros((self.num_node, self.num_node))
        for i in range(self.num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)
        
        # Self-loop(자기 자신 연결) 추가하여 스택
        return np.stack([np.eye(self.num_node), AD])


# ============================================================================
# 2. ST-GCN Modules
# ============================================================================
class ConvTemporalGraphical(nn.Module):
    """그래프 컨볼루션 레이어 (공간적 특징 추출)"""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        t_kernel_size=1,
        stride=1,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_kernel_size // 2, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            bias=bias,
        )

    def forward(self, x, A):
        # x: (Batch, Channel, Time, Node)
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        
        # Einsum을 이용한 그래프 컨볼루션 연산 (Adjacency Matrix 곱)
        x = torch.einsum("nkctv,kvw->nctw", (x, A))
        return x.contiguous()


class STGCNBlock(nn.Module):
    """시공간 그래프 컨볼루션 블록 (GCN + TCN)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.1):
        super().__init__()
        # 공간적 특징 (GCN)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size)
        
        # 시간적 특징 (TCN)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        
        # Residual Connection (차원이 다를 경우 맞춤)
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x)
        return self.relu(x + res)


class KeypointEncoderSTGCN(nn.Module):
    """Keypoint 시퀀스를 입력받아 모션 임베딩을 출력하는 인코더"""
    def __init__(
        self, num_keypoints=15, in_channels=2, hidden_channels=64, out_channels=128
    ):
        super().__init__()
        self.graph = PetGraph()
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", A)

        # ST-GCN 레이어 스택
        self.st_gcn_networks = nn.ModuleList(
            (
                STGCNBlock(in_channels, hidden_channels, 2, stride=1),
                STGCNBlock(hidden_channels, hidden_channels, 2, stride=2),
                STGCNBlock(hidden_channels, out_channels, 2, stride=1),
            )
        )
        self.out_dim = out_channels

    def forward(self, keypoints):
        # Input: (Batch, Time, Nodes, Channels) -> (Batch, Channels, Time, Nodes)
        N, T, V, C = keypoints.size()
        x = keypoints.permute(0, 3, 1, 2).contiguous()
        
        for gcn in self.st_gcn_networks:
            x = gcn(x, self.A)
            
        # Global Average Pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1)
        return x


# ============================================================================
# 3. Image Encoder & Domain Projector
# ============================================================================
class ImageEncoder(nn.Module):
    """EfficientNet-B0 기반 이미지 특징 추출기"""
    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.effnet = models.efficientnet_b0(weights=weights)
        self.output_dim = 1280
        self.effnet.classifier = nn.Identity() # 분류기 제거하고 특징만 사용
        
        if freeze:
            self.freeze()

    def freeze(self):
        for param in self.effnet.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.effnet.parameters():
            param.requires_grad = True
        # Batch Norm 등은 학습 모드로 전환하되 안정성을 위해 일부 고정 가능 (여기선 전체 해제)

    def forward(self, x):
        return self.effnet(x)


class DomainFeatureProjector(nn.Module):
    """도메인 지식 피처(50개)를 고차원 임베딩으로 변환"""
    def __init__(self, num_features: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        # 1D Convolution을 통해 시간적 변화 패턴 감지
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(num_features, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Attention 메커니즘으로 중요 시점 강조
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, features):
        B, T, feat_dim = features.size()
        x = features.permute(0, 2, 1).contiguous()
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1).contiguous()
        
        # Attention Score 계산
        attn_scores = self.temporal_attention(x)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # 가중합 (Weighted Sum)
        context = (x * attn_weights).sum(dim=1)
        output = self.projector(context)
        return output


# ============================================================================
# 4. Hybrid Fusion & Main Model
# ============================================================================
class BottleneckFusion(nn.Module):
    """
    Gated Fusion 메커니즘이 적용된 Bottleneck 모듈
    각 모달리티의 중요도를 동적으로 판단하여 결합
    """
    def __init__(
        self, img_dim=1280, motion_dim=128, domain_dim=128, bottleneck_dim=128
    ):
        super().__init__()

        # 각 모달리티를 동일한 차원(bottleneck_dim)으로 압축/변환
        self.img_bottleneck = nn.Sequential(
            nn.Linear(img_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
        )

        self.motion_refine = nn.Sequential(
            nn.Linear(motion_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
        )

        self.domain_refine = nn.Sequential(
            nn.Linear(domain_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
        )

        # Gating Network: 어떤 모달리티를 얼마나 신뢰할지 결정 (0~1)
        total_dim = bottleneck_dim * 3
        self.gate = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(total_dim // 2, 3),  # 3개 모달리티에 대한 가중치
            nn.Softmax(dim=1),
        )

        self.out_dim = total_dim

    def forward(self, img_feat, motion_feat, domain_feat):
        img_compressed = self.img_bottleneck(img_feat)
        motion_refined = self.motion_refine(motion_feat)
        domain_refined = self.domain_refine(domain_feat)

        # 게이트 가중치 계산
        concat_raw = torch.cat(
            [img_compressed, motion_refined, domain_refined], dim=1
        )
        gates = self.gate(concat_raw)  # (Batch, 3)

        # 가중치 적용 (Element-wise multiplication)
        img_weighted = img_compressed * gates[:, 0:1]
        motion_weighted = motion_refined * gates[:, 1:2]
        domain_weighted = domain_refined * gates[:, 2:3]

        # 최종 결합
        fused_feat = torch.cat([img_weighted, motion_weighted, domain_weighted], dim=1)

        return fused_feat


class HybridPetNet(nn.Module):
    """최종 하이브리드 멀티모달 네트워크"""
    def __init__(
        self,
        num_classes: int,
        num_domain_features: int = 50,
        dropout: float = 0.5,  # [수정] Dropout 비율을 외부에서 제어 가능하도록 변경
        fusion_type: str = "bottleneck",
    ):
        super().__init__()

        # 1. Encoders
        self.image_encoder = ImageEncoder(pretrained=True, freeze=True)
        self.keypoint_encoder = KeypointEncoderSTGCN(in_channels=2, out_channels=128)

        # 도메인 피처 프로젝터
        self.feature_projector = DomainFeatureProjector(
            num_features=num_domain_features,
            hidden_dim=256,
            out_dim=128,
        )

        # 2. Fusion Strategy
        self.fusion_type = fusion_type
        if fusion_type == "bottleneck":
            self.fusion = BottleneckFusion(
                img_dim=1280,
                motion_dim=128,
                domain_dim=128,
                bottleneck_dim=128,
            )
            fusion_out_dim = self.fusion.out_dim  # 384
        else:
            # 단순 연결 (Concat)
            self.fusion = nn.Identity()
            fusion_out_dim = 1280 + 128 + 128

        # 3. Classifier
        # 이진 분류 시에는 Dropout을 낮추고, 다중 분류 시에는 높이는 것이 일반적
        self.classifier = nn.Sequential(
            nn.Linear(fusion_out_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),  # Config에서 받은 Dropout 적용
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, images, keypoints, features):
        img_feat = self.image_encoder(images)
        motion_feat = self.keypoint_encoder(keypoints)
        domain_feat = self.feature_projector(features)

        if self.fusion_type == "bottleneck":
            fused = self.fusion(img_feat, motion_feat, domain_feat)
        else:
            fused = torch.cat([img_feat, motion_feat, domain_feat], dim=1)

        logits = self.classifier(fused)
        return logits

    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    print("[HybridPetNet Improved Test]")
    model = HybridPetNet(num_classes=5, num_domain_features=50, dropout=0.3)
    print(f"Params: {model.get_num_params()}")