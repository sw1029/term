import json
import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

import plotly.express as px

"""
유틸 모듈
    - JumpReLU
    - LLM 텍스트 생성 및 steering hook
    - LLM activation 추출
    - 실험 로그 저장(JSON)
    - feature 중심 plot 생성
"""


class JumpReLU(nn.Module):  # 적용 시 의문점: b의 초기값을 휴리스틱하게 들어가야 하나? 아니라면 또 하나의 파라미터로 학습 가능하게 두어야 하나?
    def __init__(self, init_b=0.0):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(init_b, dtype=torch.float32))

    def forward(self, x):
        # JumpReLU(x) = ReLU(x + b)
        return F.relu(x + self.b)


def generate_text(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# --- Steering Hook 함수 정의 ---
def get_steering_hook(steering_vector, strength=10.0):
    def hook(module, input, output):
        # output이 튜플일 경우 (hidden_states, ...) 첫 번째 요소만 가져옴
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # [핵심] 0번 피처 벡터에 강도(Strength)를 곱해서 더해줌
        # Broadcasting: (Batch, Seq, Dim) + (Dim)
        hidden = hidden + (steering_vector * strength)

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    return hook


def get_batch_activations(model, tokenizer, text_batch, layer_idx, device):
    inputs = tokenizer(
        text_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states[layer_idx]

    # [Batch, Seq_len, Dim] -> [Batch * Seq_len, Dim] (Flatten)
    activations = hidden_states.reshape(-1, hidden_states.shape[-1])

    return activations


def get_activation(text, layer, model, tokenizer):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        # Ensure inputs are on the same device as the model.
        inputs_on_model_device = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs_on_model_device, output_hidden_states=True)
        activation_tensor = outputs.hidden_states[layer].squeeze(0).mean(dim=0)
    return activation_tensor


def concept_vector(p_v, n_v):
    conect_v = p_v - n_v
    conect_v = conect_v / conect_v.norm()  # 정규화
    return conect_v


# --------------------
# Logging / Plotting
# --------------------

def save_experiment_log(
    log_dict,
    log_dir: str = "experimentLog",
    filename: str | None = None,
) -> str:
    """
    실험 결과를 JSON 파일로 저장한다.

    Args:
        log_dict: 직렬화할 dict.
        log_dir: 파일이 저장될 상위 디렉토리.
        filename: 명시적인 파일 이름 (예: "L12_S15.json").
                  None 이면 실행 시점 기준 timestamp 이름을 사용한다.
    """
    os.makedirs(log_dir, exist_ok=True)
    if filename is None:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"run_{ts}.json"
    path = os.path.join(log_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log_dict, f, ensure_ascii=False, indent=2)
    return path


def plot_feature_changes(
    indices,
    f_before,
    f_after,
    g_before,
    g_after,
    save_path: str,
):
    """
    steered feature 및 그 주변 일부 feature에 대해서만
    grouped bar plot을 생성한다.

    Args:
        indices: feature 인덱스 1차원 배열
        f_before/f_after/g_before/g_after: 동일 길이의 텐서 또는 리스트
        save_path: HTML 파일 경로
    """
    # convert to plain Python types
    indices = [int(i) for i in indices]
    f_before = [float(v) for v in f_before]
    f_after = [float(v) for v in f_after]
    g_before = [float(v) for v in g_before]
    g_after = [float(v) for v in g_after]

    rows = []
    for idx, fb, fa, gb, ga in zip(indices, f_before, f_after, g_before, g_after):
        rows.append({"feature": idx, "type": "f_before", "value": fb})
        rows.append({"feature": idx, "type": "f_after", "value": fa})
        rows.append({"feature": idx, "type": "g_before", "value": gb})
        rows.append({"feature": idx, "type": "g_after", "value": ga})

    fig = px.bar(rows, x="feature", y="value", color="type", barmode="group")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)


def compute_feature_stats(
    model,
    tokenizer,
    sae,
    data_loader,
    layer_idx: int,
    device,
    max_batches: int = 10,
):
    """
    데이터셋 일부 구간에 대해 SAE feature 활성값의 평균/표준편차를 계산한다.

    Args:
        model, tokenizer: activation 추출에 사용할 LLM과 토크나이저.
        sae: 학습이 완료된 H_SAE 인스턴스.
        data_loader: 텍스트 샘플이 들어 있는 DataLoader (batch['text'] 사용).
        layer_idx: hook을 걸 LLM 레이어 인덱스.
        device: torch.device.
        max_batches: 통계를 계산할 때 사용할 최대 배치 수.

    Returns:
        mean: (d_sae,) 텐서
        std: (d_sae,) 텐서
    """
    sae.eval()
    all_feats = []
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            text_batch = batch["text"]
            if not text_batch:
                continue

            x = get_batch_activations(model, tokenizer, text_batch, layer_idx, device)
            if x.shape[0] == 0:
                continue

            out = sae(x)
            if isinstance(out, tuple):
                _, f = out[0], out[1]
            else:
                _, f = out, None

            if f is None:
                continue

            all_feats.append(f.detach().cpu())
            num_batches += 1

            if num_batches >= max_batches:
                break

    if not all_feats:
        return None, None

    feats = torch.cat(all_feats, dim=0)  # (N, d_sae)
    mean = feats.mean(dim=0)
    std = feats.std(dim=0)
    return mean, std


def compute_llm_hidden_stats(
    model,
    tokenizer,
    data_loader,
    layer_idx: int,
    device,
    max_batches: int = 10,
):
    """
    데이터셋 일부 구간에 대해 LLM hidden (특정 레이어)의
    각 차원별 평균/표준편차를 계산한다.

    Args:
        model, tokenizer: activation 추출에 사용할 LLM과 토크나이저.
        data_loader: 텍스트 샘플이 들어 있는 DataLoader (batch['text'] 사용).
        layer_idx: hidden을 추출할 LLM 레이어 인덱스.
        device: torch.device.
        max_batches: 통계를 계산할 때 사용할 최대 배치 수.

    Returns:
        mean: (hidden_dim,) 텐서
        std: (hidden_dim,) 텐서
    """
    model.eval()
    all_hidden = []
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            text_batch = batch["text"]
            if not text_batch:
                continue

            x = get_batch_activations(model, tokenizer, text_batch, layer_idx, device)
            if x.shape[0] == 0:
                continue

            all_hidden.append(x.detach().cpu())
            num_batches += 1

            if num_batches >= max_batches:
                break

    if not all_hidden:
        return None, None

    hidden = torch.cat(all_hidden, dim=0)  # (N, hidden_dim)
    mean = hidden.mean(dim=0)
    std = hidden.std(dim=0)
    return mean, std


def plot_feature_std_curve(std_tensor: torch.Tensor, save_path: str):
    """
    모든 SAE feature에 대해 표준편차를 계산한 뒤,
    내림차순으로 정렬한 곡선을 그려 분포를 시각화한다.

    - 많은 feature를 한 번에 보기 위해
      std 값을 내림차순으로 정렬한 line plot을 사용한다.
    """
    if std_tensor is None:
        return

    std_sorted, _ = torch.sort(std_tensor, descending=True)
    xs = list(range(len(std_sorted)))
    ys = [float(v) for v in std_sorted]

    rows = [{"rank": i, "std": v} for i, v in zip(xs, ys)]
    fig = px.line(rows, x="rank", y="std", title="SAE Feature STD (sorted)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)


def plot_llm_std_curve(std_tensor: torch.Tensor, save_path: str):
    """
    LLM hidden 차원별 표준편차를 내림차순으로 정렬한 곡선을 그려,
    모델 자체의 표현 분포를 시각화한다.
    """
    if std_tensor is None:
        return

    std_sorted, _ = torch.sort(std_tensor, descending=True)
    xs = list(range(len(std_sorted)))
    ys = [float(v) for v in std_sorted]

    rows = [{"rank": i, "std": v} for i, v in zip(xs, ys)]
    fig = px.line(rows, x="rank", y="std", title="LLM Hidden STD (sorted)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)
