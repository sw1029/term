import torch


def _concept_loss_one(
    f_k: torch.Tensor,
    labels: torch.Tensor,
    positive_target: int = 1,
) -> torch.Tensor:
    """
    단일 feature k에 대한 contrastive supervision 손실을 계산한다.

    - positive_target 레이블(예: 1)에서 f_k 평균이 크고,
      반대 레이블(0)에서 f_k 평균이 작도록 유도한다.
    """
    if labels is None or f_k is None:
        return torch.tensor(0.0, device=f_k.device if f_k is not None else "cpu")

    pos_mask = labels == positive_target
    neg_mask = labels == 1 - positive_target

    if not pos_mask.any() or not neg_mask.any():
        return torch.tensor(0.0, device=f_k.device)

    mu_pos = f_k[pos_mask].mean()
    mu_neg = f_k[neg_mask].mean()
    # pos에서 크고 neg에서 작아지도록 하는 손실
    return - (mu_pos - mu_neg)


def multi_concept_contrastive_loss(
    f: torch.Tensor,
    batch,
    concept_indices: dict,
    alpha_concept: dict,
    device,
    positive_targets: dict | None = None,
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

        L_c = _concept_loss_one(f_k, labels, positive_target=pos_target)
        total = total + alpha * L_c

    return total

