import torch

from base_losses import reconstruction_loss, sparsity_loss


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

    x = x.to(device)
    x_reconst = x_reconst.to(device)
    if f is not None:
        f = f.to(device)

    l2 = reconstruction_loss(x_reconst, x)
    l1 = sparsity_loss(f, l1_coeff)

    total = l2 + l1

    loss_items = {
        "l2": l2,
        "l1": l1,
    }
    return total, loss_items

