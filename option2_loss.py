import torch

from base_losses import reconstruction_loss, sparsity_loss, l0_sparsity_loss
from contrastive_losses import multi_concept_contrastive_loss


def compute_loss(
    sae_model,
    x,
    x_reconst,
    f,
    batch,
    config,
):
    """
    Option2: concept 고정 + 재구성 + sparsity + multi-concept label guidance.
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

    x = x.to(device)
    x_reconst = x_reconst.to(device)
    if f is not None:
        f = f.to(device)

    l2 = reconstruction_loss(x_reconst, x)
    l1 = sparsity_loss(f, l1_coeff)
    l0 = torch.tensor(0.0, device=device)
    if use_l0:
        l0 = l0_sparsity_loss(
            getattr(sae_model, "last_l0_count", None),
            l0_coeff,
            device=device,
        )

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

    # 역할 분리(decorrelation) loss: concept feature 간 공분산(off-diagonal)을 줄이는 항
    role_sep = torch.tensor(0.0, device=device)
    if role_sep_coeff > 0.0 and f is not None:
        if concept_indices:
            idxs = [int(idx) for idx in concept_indices.values()]
            if len(idxs) > 1:
                F = f[:, idxs]  # [batch, num_concepts]
                F_centered = F - F.mean(dim=0, keepdim=True)
                cov = (F_centered.T @ F_centered) / max(F_centered.size(0), 1)
                diag = torch.diag(torch.diag(cov))
                offdiag = cov - diag
                role_sep_raw = (offdiag.pow(2)).mean()
                role_sep = role_sep_coeff * role_sep_raw
                total = total + role_sep
                loss_items["role_sep"] = role_sep

    return total, loss_items
