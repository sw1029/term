import torch
import torch.nn.functional as F


def _concept_loss_one(
    f_k: torch.Tensor,
    labels: torch.Tensor,
    positive_target: int = 1,
    method: str = "contrastive",
    margin: float = 0.0,
    mse_pos_value: float = 1.0,
    mse_neg_value: float = 0.0,
) -> torch.Tensor:
    """
    단일 feature k에 대한 supervision 손실을 계산한다.

    method:
        - "contrastive": -(mu_pos - mu_neg)
        - "margin": ReLU(margin - (mu_pos - mu_neg))
        - "mse": 양성/음성에 대한 target 값을 주고 MSE를 최소화
    """
    if labels is None or f_k is None:
        return torch.tensor(0.0, device=f_k.device if f_k is not None else "cpu")

    pos_mask = labels == positive_target
    neg_mask = labels == 1 - positive_target

    if not pos_mask.any() or not neg_mask.any():
        return torch.tensor(0.0, device=f_k.device)

    if method == "mse":
        target = torch.zeros_like(f_k, device=f_k.device, dtype=f_k.dtype)
        target[pos_mask] = mse_pos_value
        target[neg_mask] = mse_neg_value
        return F.mse_loss(f_k, target)

    mu_pos = f_k[pos_mask].mean()
    mu_neg = f_k[neg_mask].mean()

    if method == "margin":
        return F.relu(margin - (mu_pos - mu_neg))

    # default: 간단한 contrastive 차이 극대화
    return - (mu_pos - mu_neg)


def multi_concept_contrastive_loss(
    f: torch.Tensor,
    batch,
    concept_indices: dict,
    alpha_concept: dict,
    device,
    positive_targets: dict | None = None,
    method: str = "contrastive",
    margin: float = 0.0,
    mse_pos_value: float = 1.0,
    mse_neg_value: float = 0.0,
) -> torch.Tensor:
    """
    여러 개의 concept feature(code, harm, struct 등)에 대해
    contrastive loss를 동시에 계산한다.

    Args:
        f: (batch, d_sae) SAE feature 텐서
        batch: 학습 배치(dict), label_code / label_harm / label_struct 등을 포함 가능
        concept_indices: {"code": idx, "harm": idx, "struct": idx}
        alpha_concept:   {"code": α_code, ...} 각 concept loss의 가중치
        positive_targets: {"code": 1, "harm": 1, "struct": 1} 등
    """
    if f is None:
        return torch.tensor(0.0, device=device)

    if positive_targets is None:
        positive_targets = {k: 1 for k in concept_indices.keys()}

    total = torch.tensor(0.0, device=device)

    for name, idx in concept_indices.items():
        alpha = alpha_concept.get(name, 0.0)
        if alpha == 0.0:
            continue

        if idx < 0 or idx >= f.size(1):
            continue

        labels_key = f"label_{name}"
        if labels_key not in batch:
            continue

        labels = batch[labels_key].to(device)
        f_k = f[:, idx]
        pos_target = positive_targets.get(name, 1)

        L_c = _concept_loss_one(
            f_k,
            labels,
            positive_target=pos_target,
            method=method,
            margin=margin,
            mse_pos_value=mse_pos_value,
            mse_neg_value=mse_neg_value,
        )
        total = total + alpha * L_c

    return total
