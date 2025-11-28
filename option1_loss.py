import torch

from base_losses import reconstruction_loss, sparsity_loss, l0_sparsity_loss


def compute_loss(
    sae_model,
    x,
    x_reconst,
    f,
    batch,
    config,
):
    """
    Option1: concept 고정 + 재구성 + sparsity만 사용하는 손실.
    """
    device = config["device"]
    l1_coeff = config["l1_coeff"]
    l0_coeff = config.get("l0_coeff", 0.0)
    use_l0 = config.get("use_l0", False)

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

    total = l2 + l1 + l0

    loss_items = {
        "l2": l2,
        "l1": l1,
        "l0": l0,
    }
    return total, loss_items
