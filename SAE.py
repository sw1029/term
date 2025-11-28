import torch
import torch.nn as nn
import utils

"""
의문: feature간 상관관계를 attention 혹은 GNN으로 추가 설명가능성을 확보할 수는 없을까?
    이 경우 안쪽에 간단한 형식의 모델을 추가하는 것만으로 추가적인 시각화 가능성/해석가능성의 담보가 가능
    다만 이 경우 attention/GNN을 추가로 붙일 때 정말 간단하게만 붙여서 내부에 또 하나의 블랙박스가 생기지 않도록 주의가 필요할수도?

본 구현에서는:
    - SAE의 encoder/decoder 경로 및 재구성 손실은 그대로 유지한다.
    - SAE feature f를 입력으로 하는 1-layer GNN(feature_gnn)을 선택적으로 주입한다.
    - GNN은 설명(XAI)용으로만 사용되며, 학습 손실에는 포함하지 않는다.
"""


class H_SAE(nn.Module):
    def __init__(
        self,
        input_dim,
        sae_dim,
        c_vectors=None,
        feature_gnn: nn.Module | None = None,
        jumprelu_bandwidth: float = 0.001,
        jumprelu_init_threshold: float = 0.001,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.sae_dim = sae_dim
        self.last_l0_count = None  # Heaviside 기반 L0 카운트 (batch 단위)

        # 인코더 레이어
        self.encoder = nn.Linear(input_dim, sae_dim)
        self.enc_bias = nn.Parameter(torch.zeros(sae_dim))

        # 디코더 (출력 차원은 입력 차원과 동일)
        self.decoder = nn.Parameter(torch.randn(sae_dim, input_dim))  # 출력 크기 = 입력 크기
        self.dec_bias = nn.Parameter(torch.zeros(input_dim))

        # 컨셉 벡터 (디코더의 고정 구간)
        if c_vectors is not None:
            self.fixed_v_cnt = len(c_vectors)
            with torch.no_grad():
                self.decoder.data[: self.fixed_v_cnt] = c_vectors.clone()
        else:
            # concept-free 설정 (Option3 등)에서는 고정 벡터 없음
            self.fixed_v_cnt = 0

        # JumpReLU 활성화 모듈 (feature별 threshold 학습)
        self.jumprelu = utils.JumpReLU(
            feature_dim=self.sae_dim,
            bandwidth=jumprelu_bandwidth,
            init_threshold=jumprelu_init_threshold,
        )

        # 선택적으로 feature 상에서 동작하는 설명 전용 GNN 모듈
        self.feature_gnn = feature_gnn

    def set_feature_gnn(self, feature_gnn: nn.Module | None) -> None:
        """
        설명용 GNN 모듈을 외부에서 주입하기 위한 메서드.
        """
        self.feature_gnn = feature_gnn

    def forward(self, x):
        # SAE feature 추출 (구조를 단순/명시적으로 유지)
        z = self.encoder(x) - self.enc_bias
        # JumpReLU로 활성화
        f = self.jumprelu(z)

        # Heaviside step으로 L0 카운트 (sparsity 측정용)
        threshold = self.jumprelu.get_threshold()
        bandwidth = self.jumprelu.bandwidth
        l0_mask = utils.HeavisideStepFunction.apply(z, threshold, bandwidth)  # (batch, d_sae)
        self.last_l0_count = l0_mask.sum(dim=-1)  # (batch,)

        # 재구성 단계
        x_reconst = torch.matmul(f, self.decoder) + self.dec_bias

        # 선택적인 1-layer GNN (설명 전용 브랜치)
        g = None
        if self.feature_gnn is not None:
            g = self.feature_gnn(f)

        return x_reconst, f, g
