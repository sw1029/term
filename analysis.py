import torch

from utils import generate_text, get_activation, get_steering_hook


def _run_steering_analysis_core(
    model,
    tokenizer,
    sae,
    prompt: str,
    layer_idx: int,
    steering_vector: torch.Tensor,
    strength: float,
    hook_layer_idx: int,
):
    """
    내부용 코어 함수:
    지정한 hook_layer_idx에 steering hook을 걸어
    steering 전/후 SAE feature, GNN 출력, LLM 텍스트를 비교한다.
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
    hook = model.model.layers[hook_layer_idx].mlp.register_forward_hook(
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
    hook2 = model.model.layers[hook_layer_idx].mlp.register_forward_hook(
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
        "hook_layer_idx": int(hook_layer_idx),
    }


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
    # 기본 실행 경로: layer_idx-1 에 hook을 거는 현재 정책을 유지
    hook_layer_idx = max(layer_idx - 1, 0)
    return _run_steering_analysis_core(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        prompt=prompt,
        layer_idx=layer_idx,
        steering_vector=steering_vector,
        strength=strength,
        hook_layer_idx=hook_layer_idx,
    )


def run_steering_analysis_compare_hooks(
    model,
    tokenizer,
    sae,
    prompt: str,
    layer_idx: int,
    steering_vector: torch.Tensor,
    strength: float,
):
    """
    동일한 설정에서
      - hook을 layer_idx-1에 걸었을 때
      - hook을 layer_idx에 걸었을 때
    두 경우 모두에 대해 steering 전/후 feature 활성 변화를 비교한다.

    반환 dict 구조:
      {
        "hook_layer_idx_minus1": {
            ... 기존 run_steering_analysis 결과 필드 ...,
            "delta_f": f_after - f_before,
            "delta_g": g_after - g_before 또는 None,
            "hook_layer_idx": <int>,
        },
        "hook_layer_idx_same": {
            ...,
        },
      }
    """
    # case 1: hook at layer_idx-1 (최소 0)
    hook_idx_minus1 = max(layer_idx - 1, 0)
    res_minus1 = _run_steering_analysis_core(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        prompt=prompt,
        layer_idx=layer_idx,
        steering_vector=steering_vector,
        strength=strength,
        hook_layer_idx=hook_idx_minus1,
    )

    # case 2: hook at layer_idx
    hook_idx_same = layer_idx
    res_same = _run_steering_analysis_core(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        prompt=prompt,
        layer_idx=layer_idx,
        steering_vector=steering_vector,
        strength=strength,
        hook_layer_idx=hook_idx_same,
    )

    # feature 활성 변화(Δf, Δg) 계산
    df_minus1 = res_minus1["f_after"] - res_minus1["f_before"]
    df_same = res_same["f_after"] - res_same["f_before"]

    dg_minus1 = None
    if res_minus1["g_before"] is not None and res_minus1["g_after"] is not None:
        dg_minus1 = res_minus1["g_after"] - res_minus1["g_before"]

    dg_same = None
    if res_same["g_before"] is not None and res_same["g_after"] is not None:
        dg_same = res_same["g_after"] - res_same["g_before"]

    return {
        "hook_layer_idx_minus1": {
            **res_minus1,
            "delta_f": df_minus1,
            "delta_g": dg_minus1,
            "hook_layer_idx": hook_idx_minus1,
        },
        "hook_layer_idx_same": {
            **res_same,
            "delta_f": df_same,
            "delta_g": dg_same,
            "hook_layer_idx": hook_idx_same,
        },
    }
