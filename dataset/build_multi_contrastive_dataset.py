import os
import json
import random
import re

from datasets import load_dataset
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path


def is_code_like(text: str) -> bool:
    """
    Python 코드 여부를 휴리스틱하게 판별.
    """
    if "```python" in text or "```" in text:
        return True
    if re.search(r"\b(def|class|import|for|while|try|except)\b", text):
        return True
    if re.search(r"\bprint\(", text):
        return True
    return False


def is_structured(text: str) -> bool:
    """
    응답이 구조화(steps, bullet, heading 등)되어 있는지 휴리스틱하게 판별.
    """
    if re.search(r"^\s*[-*]\s+", text, flags=re.MULTILINE):
        return True
    if re.search(r"^\s*\d+\.\s+", text, flags=re.MULTILINE):
        return True
    if re.search(r"Step\s+\d+", text, flags=re.IGNORECASE):
        return True
    if re.search(r"(###|####|\bSummary\b|\bConclusion\b)", text):
        return True
    return False


def initial_harm_label_from_tox(sample) -> int:
    """
    RealToxicityPrompts 샘플에서 유해성 라벨을 초기 추정.
    """
    tox_score = sample.get("toxicity", None)
    if tox_score is not None and tox_score > 0.7:
        return 1

    text = sample.get("prompt", {}).get("text", "")
    if not text:
        return 0

    if re.search(r"(fuck|shit|idiot|bitch|hate)", text.lower()):
        return 1
    return 0


def sample_stack_python(n):
    ds = load_dataset(
        "bigcode/the-stack",
        data_dir="data/python",
        streaming=True,
        split="train",
    )
    out = []
    it = iter(ds)
    while len(out) < n:
        s = next(it)
        code = s.get("content") or s.get("text") or ""
        if not code.strip():
            continue
        out.append({
            "text": code,
            "label_code": 1,
            "label_harm": 0,
            "label_struct": 0,
            "source": "stack",
        })
    return out


def sample_alpaca_code(n):
    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    out = []
    it = iter(ds)
    while len(out) < n:
        s = next(it)
        combined = s["instruction"] + "\n\n" + s["output"]
        if is_code_like(combined):
            out.append({
                "text": combined,
                "label_code": 1,
                "label_harm": 0,
                "label_struct": int(is_structured(combined)),
                "source": "alpaca_code",
            })
    return out


def sample_alpaca_noncode(n):
    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    out = []
    it = iter(ds)
    while len(out) < n:
        s = next(it)
        combined = s["instruction"] + "\n\n" + s["output"]
        if not is_code_like(combined):
            out.append({
                "text": combined,
                "label_code": 0,
                "label_harm": 0,
                "label_struct": int(is_structured(combined)),
                "source": "alpaca_text",
            })
    return out


def sample_toxic(n):
    ds = load_dataset("allenai/real-toxicity-prompts", split="train", streaming=True)
    out = []
    it = iter(ds)
    while len(out) < n:
        s = next(it)
        text = s.get("prompt", {}).get("text", "")
        if not text.strip():
            continue
        harm = initial_harm_label_from_tox(s)
        out.append({
            "text": text,
            "label_code": 0,
            "label_harm": harm,
            "label_struct": int(is_structured(text)),
            "source": "tox_prompt",
        })
    return out


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    여러 공개 데이터셋을 기반으로 multi-label contrastive dataset을 구축한다.
    출력 경로 및 샘플 개수는 Hydra config(config/dataset, config/paths)를 통해 제어한다.
    """
    mc_cfg = cfg.dataset.multi_contrast
    n_pos_stack = mc_cfg.n_pos_stack
    n_pos_alpaca = mc_cfg.n_pos_alpaca
    n_neg_alpaca = mc_cfg.n_neg_alpaca
    n_tox = mc_cfg.n_tox

    print("Sampling Stack Python...")
    pos_stack = sample_stack_python(n_pos_stack)
    print("Sampling Alpaca code...")
    pos_alpaca = sample_alpaca_code(n_pos_alpaca)
    print("Sampling Alpaca non-code...")
    neg_alpaca = sample_alpaca_noncode(n_neg_alpaca)
    print("Sampling Toxic prompts...")
    tox_samples = sample_toxic(n_tox)

    data = pos_stack + pos_alpaca + neg_alpaca + tox_samples
    random.shuffle(data)

    raw_path_cfg = cfg.paths.datasets.multi_contrast_raw
    out_path = to_absolute_path(raw_path_cfg)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved raw multi-contrast dataset to {out_path}")


if __name__ == "__main__":
    main()

