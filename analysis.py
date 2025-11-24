import torch

from utils import generate_text, get_activation, get_steering_hook


def run_steering_analysis(
    model,
    tokenizer,
    sae,
    prompt: str,
    layer_idx: int,
    steering_vector: torch.Tensor,
    strength: float,
):
    """
    steering 적용 전/후의 SAE feature, GNN 출력, LLM 텍스트를 비교한다.

    반환되는 dict에는 다음 항목이 포함된다.
        - f_before / f_after
        - g_before / g_after (존재하는 경우)
        - y_before / y_after (LLM 텍스트)
    """
    device = sae.decoder.device

    # 1) before steering: hidden, SAE forward
    x_before = get_activation(prompt, layer_idx, model, tokenizer)
    x_before = x_before.unsqueeze(0).to(device)
    out_before = sae(x_before)
    if isinstance(out_before, tuple) and len(out_before) == 3:
        _, f_before, g_before = out_before
    else:
        _, f_before = out_before
        g_before = None

    # 2) after steering: hook -> hidden -> SAE forward
    hook = model.model.layers[layer_idx].mlp.register_forward_hook(
        get_steering_hook(steering_vector, strength=strength)
    )
    try:
        x_after = get_activation(prompt, layer_idx, model, tokenizer)
    finally:
        hook.remove()

    x_after = x_after.unsqueeze(0).to(device)
    out_after = sae(x_after)
    if isinstance(out_after, tuple) and len(out_after) == 3:
        _, f_after, g_after = out_after
    else:
        _, f_after = out_after
        g_after = None

    # 3) LLM 텍스트 출력(전/후)
    y_before = generate_text(model, tokenizer, prompt)
    hook2 = model.model.layers[layer_idx].mlp.register_forward_hook(
        get_steering_hook(steering_vector, strength=strength)
    )
    try:
        y_after = generate_text(model, tokenizer, prompt)
    finally:
        hook2.remove()

    return {
        "f_before": f_before.detach().cpu(),
        "f_after": f_after.detach().cpu(),
        "g_before": g_before.detach().cpu() if g_before is not None else None,
        "g_after": g_after.detach().cpu() if g_after is not None else None,
        "y_before": y_before,
        "y_after": y_after,
    }
