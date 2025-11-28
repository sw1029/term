import torch
import torch.nn.functional as F


def reconstruction_loss(x_reconst: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    SAE 재구성 손실 (L2, MSE) 계산.
    """
    return F.mse_loss(x_reconst, x)


def sparsity_loss(f: torch.Tensor, l1_coeff: float) -> torch.Tensor:
    """
    SAE feature에 대한 L1 sparsity 손실 계산.

    Args:
        f: (batch, d_sae) 형태의 feature 텐서
        l1_coeff: L1 가중치 계수
    """
    if f is None:
        # feature가 없는 경우 손실을 0으로 반환 (CPU 기준)
        return torch.tensor(0.0)
    return l1_coeff * f.abs().sum(dim=1).mean()


def l0_sparsity_loss(l0_count, l0_coeff, device):
    """
    Heaviside 기반 L0 sparsity 손실 계산.

    Args:
        l0_count: (batch,) 형태의 활성 unit 수 텐서 또는 None
        l0_coeff: L0 가중치 계수
        device: 반환 텐서가 올라갈 디바이스 (str 또는 torch.device)
    """
    if l0_count is None:
        return torch.tensor(0.0, device=device)
    l0_count = l0_count.to(device)
    return l0_coeff * l0_count.mean()
