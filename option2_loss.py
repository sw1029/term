import torch

from base_losses import reconstruction_loss, sparsity_loss
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
    concept_indices = config.get("concept_feature_indices", {})
    alpha_concept = config.get("alpha_concept", {})
    positive_targets = config.get("positive_targets", None)

    x = x.to(device)
    x_reconst = x_reconst.to(device)
    if f is not None:
        f = f.to(device)

    l2 = reconstruction_loss(x_reconst, x)
    l1 = sparsity_loss(f, l1_coeff)

    guidance = torch.tensor(0.0, device=device)
    if concept_indices and alpha_concept:
        guidance = multi_concept_contrastive_loss(
            f=f,
            batch=batch,
            concept_indices=concept_indices,
            alpha_concept=alpha_concept,
            device=device,
            positive_targets=positive_targets,
        )

    total = l2 + l1 + guidance

    loss_items = {
        "l2": l2,
        "l1": l1,
        "guidance": guidance,
    }
    return total, loss_items

