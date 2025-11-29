import torch

from base_losses import reconstruction_loss, sparsity_loss, l0_sparsity_loss
from contrastive_losses import multi_concept_contrastive_loss


def _alignment_penalty(
    f,
    batch,
    concept_indices: dict,
    device,
    align_pos_margin: float = 0.8,
    align_neg_margin: float = 0.1,
    align_coeff: float = 0.1,
) -> torch.Tensor:
    """
    추가 alignment 정렬 항:
        - 양성(해당 라벨=1)에서 feature가 충분히 크게 켜지도록
        - 음성(해당 라벨=0)에서 feature가 거의 0에 가깝도록
        - 다른 라벨이 양성인 경우에도 해당 feature가 작게 유지되도록
    """
    if f is None or not concept_indices:
        return torch.tensor(0.0, device=device)

    total = torch.tensor(0.0, device=device)

    # 메인 라벨별 정렬
    for name, idx in concept_indices.items():
        feat_idx = int(idx)
        if feat_idx < 0 or feat_idx >= f.size(1):
            continue

        labels_key = f"label_{name}"
        if labels_key not in batch:
            continue

        labels = batch[labels_key].to(device)
        feat = f[:, feat_idx]

        pos_mask = labels == 1
        neg_mask = labels == 0

        if pos_mask.any():
            mu_pos = feat[pos_mask].mean()
            total = total + torch.relu(align_pos_margin - mu_pos)

        if neg_mask.any():
            mu_neg = feat[neg_mask].mean()
            total = total + torch.relu(mu_neg - align_neg_margin)

    # cross-label 정렬: 예) code feature는 harm=1 / struct=1일 때 작게 유지
    concept_names = list(concept_indices.keys())
    for name, idx in concept_indices.items():
        feat_idx = int(idx)
        if feat_idx < 0 or feat_idx >= f.size(1):
            continue

        feat = f[:, feat_idx]

        for other in concept_names:
            if other == name:
                continue
            labels_key = f"label_{other}"
            if labels_key not in batch:
                continue
            labels_other = batch[labels_key].to(device)
            other_pos = labels_other == 1
            if other_pos.any():
                mu_cross = feat[other_pos].mean()
                total = total + torch.relu(mu_cross - align_neg_margin)

    return align_coeff * total


def compute_loss(
    sae_model,
    x,
    x_reconst,
    f,
    batch,
    config,
):
    """
    Option4: option2_loss + alignment 정렬 항
        - 재구성 + sparsity + multi-concept label guidance
        - 역할 분리(role_sep)
        - 추가 alignment 패널티:
            * 양성 라벨에서 해당 feature가 충분히 켜지도록
            * 음성 및 다른 라벨 양성인 경우에는 해당 feature가 작게 유지되도록
    """
    device = config["device"]
    l1_coeff = config["l1_coeff"]
    l0_coeff = config.get("l0_coeff", 0.0)
    use_l0 = config.get("use_l0", False)
    concept_indices = config.get("concept_feature_indices", {})
    alpha_concept = config.get("alpha_concept", {})
    positive_targets = config.get("positive_targets", None)
    guidance_method = config.get("guidance_method", "contrastive")
    guidance_margin = config.get("guidance_margin", 0.0)
    guidance_coeff = config.get("guidance_coeff", 1.0)
    mse_pos = config.get("guidance_mse_pos_value", 1.0)
    mse_neg = config.get("guidance_mse_neg_value", 0.0)
    role_sep_coeff = config.get("role_sep_coeff", 0.0)

    # alignment 정렬 관련 하이퍼파라미터 (config에 없으면 기본값 사용)
    align_pos_margin = float(config.get("align_pos_margin", 0.8))
    align_neg_margin = float(config.get("align_neg_margin", 0.1))
    align_coeff = float(config.get("align_coeff", 0.1))

    x = x.to(device)
    x_reconst = x_reconst.to(device)
    if f is not None:
        f = f.to(device)

    # 기본 재구성 + sparsity + L0
    l2 = reconstruction_loss(x_reconst, x)
    l1 = sparsity_loss(f, l1_coeff)
    l0 = torch.tensor(0.0, device=device)
    if use_l0:
        l0 = l0_sparsity_loss(
            getattr(sae_model, "last_l0_count", None),
            l0_coeff,
            device=device,
        )

    # multi-concept label guidance
    guidance = torch.tensor(0.0, device=device)
    if concept_indices and alpha_concept:
        raw_guidance = multi_concept_contrastive_loss(
            f=f,
            batch=batch,
            concept_indices=concept_indices,
            alpha_concept=alpha_concept,
            device=device,
            positive_targets=positive_targets,
            method=guidance_method,
            margin=guidance_margin,
            mse_pos_value=mse_pos,
            mse_neg_value=mse_neg,
        )
        guidance = guidance_coeff * raw_guidance

    total = l2 + l1 + l0 + guidance

    loss_items = {
        "l2": l2,
        "l1": l1,
        "l0": l0,
        "guidance": guidance,
    }

    # 역할 분리(decorrelation) loss
    role_sep = torch.tensor(0.0, device=device)
    if role_sep_coeff > 0.0 and f is not None:
        if concept_indices:
            idxs = [int(idx) for idx in concept_indices.values()]
            if len(idxs) > 1:
                feat = f[:, idxs]  # [batch, num_concepts]
                feat_centered = feat - feat.mean(dim=0, keepdim=True)
                cov = (feat_centered.T @ feat_centered) / max(
                    feat_centered.size(0), 1
                )
                diag = torch.diag(torch.diag(cov))
                offdiag = cov - diag
                role_sep_raw = (offdiag.pow(2)).mean()
                role_sep = role_sep_coeff * role_sep_raw
                total = total + role_sep
                loss_items["role_sep"] = role_sep

    # alignment 정렬 패널티
    align_loss = _alignment_penalty(
        f=f,
        batch=batch,
        concept_indices=concept_indices,
        device=device,
        align_pos_margin=align_pos_margin,
        align_neg_margin=align_neg_margin,
        align_coeff=align_coeff,
    )
    total = total + align_loss
    loss_items["align"] = align_loss

    return total, loss_items

