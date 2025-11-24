import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureGNN(nn.Module):
    """
    SAE feature 위에서만 동작하는 매우 단순한 1-layer GNN (설명 전용).

    - 노드: SAE feature 인덱스
    - 간선: 디코더 행(또는 미리 정의된 그래프) 간 유사도
    - 학습 파라미터: 없음 (그래프 고정) → 새로운 블랙박스 모듈이 되지 않도록 최소 설계
    """

    def __init__(self, A_norm: torch.Tensor):
        """
        Args:
            A_norm: (d_sae, d_sae) 형태의 행 정규화 인접 행렬.
        """
        super().__init__()
        # buffer: not trainable, moved with .to(device)
        self.register_buffer("A_norm", A_norm)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f: (batch, d_sae) SAE feature 활성값.

        Returns:
            g: (batch, d_sae) GNN을 통해 전파된 feature 중요도.
        """
        g = torch.matmul(f, self.A_norm.T)
        return F.relu(g)


def build_feature_graph_from_decoder(
    decoder: torch.Tensor,
    top_k: int = 10,
) -> torch.Tensor:
    """
    디코더 행벡터 간 유사도를 이용해 SAE feature 그래프(인접 행렬)를 구성한다.

    - 노드: 디코더의 각 행(= feature)
    - 간선 가중치(i,j): decoder[i], decoder[j] 간 cosine similarity
    - 희소성 유지를 위해 각 노드당 top_k개의 양수 이웃만 남긴다.

    Returns:
        A_norm: (d_sae, d_sae) 행 정규화 인접 행렬.
    """
    # decoder: (d_sae, d_model)
    with torch.no_grad():
        norm_dec = F.normalize(decoder, dim=1)
        sim = norm_dec @ norm_dec.T  # (d_sae, d_sae)

        d_sae = sim.size(0)
        A = torch.zeros_like(sim)

        for i in range(d_sae):
            sim_i = sim[i].clone()
            sim_i[i] = 0.0  # remove self-loop for neighbor selection
            # if top_k > available neighbors, torch.topk will handle
            vals, idx = torch.topk(sim_i, k=min(top_k, d_sae - 1))
            # keep only positive similarities
            vals = vals.clamp(min=0.0)
            A[i, idx] = vals

        # Row-normalize: D^{-1} A
        row_sum = A.sum(dim=1, keepdim=True) + 1e-8
        A_norm = A / row_sum

    return A_norm


def select_topk_neighbors(A_norm: torch.Tensor, center_idx: int, k: int) -> torch.Tensor:
    """
    중심 feature 인덱스와 그 주변 top-k 이웃 feature 인덱스를 선택한다.

    Args:
        A_norm: (d_sae, d_sae) 행 정규화 인접 행렬.
        center_idx: steering에 사용된 중심 feature 인덱스.
        k: 포함할 이웃 개수(중심 노드는 제외한 수).

    Returns:
        indices: (k+1,) 텐서, [center_idx, neighbor_1, ...] 형태
    """
    d_sae = A_norm.size(0)
    center_idx = int(center_idx)
    k = max(0, min(k, d_sae - 1))

    row = A_norm[center_idx].clone()
    row[center_idx] = 0.0
    if k > 0:
        _, idx = torch.topk(row, k=k)
        indices = torch.cat(
            [torch.tensor([center_idx], device=A_norm.device), idx],
            dim=0,
        )
    else:
        indices = torch.tensor([center_idx], device=A_norm.device)
    return indices
