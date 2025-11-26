import os
import json
import random
import re
from typing import List, Dict, Any

from datasets import load_dataset
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path


def is_code_like(text: str) -> bool:
    """
    Python 코드 여부를 휴리스틱하게 판별.

    - 명시적인 코드 블록( ``` )이 있는 경우
    - def/class/import/from 등 파이썬 구문이 줄의 시작에 오는 경우
    - 제어문(for/while/if/try/except/with)이 코드 형태로 사용되는 경우
    - print(...) 형태의 호출이 포함된 경우
    """
    if "```" in text:
        return True

    # 함수/클래스 정의
    if re.search(r"^\s*(def|class)\s+\w+", text, flags=re.MULTILINE):
        return True

    # import 구문
    if re.search(r"^\s*import\s+\w+", text, flags=re.MULTILINE):
        return True
    if re.search(r"^\s*from\s+\w+\s+import\b", text, flags=re.MULTILINE):
        return True

    # 제어문/블록 시작 (자연어 'for'와 구분하기 위해 줄 시작 + ':' 패턴만 사용)
    if re.search(
        r"^\s*(for|while|if|elif|else|try|except|with)\s+.+:",
        text,
        flags=re.MULTILINE,
    ):
        return True

    # 대표적인 함수 호출 패턴
    if "print(" in text:
        return True

    return False


def is_structured(text: str) -> bool:
    """
    응답이 구조화(steps, bullet, heading 등)되어 있는지 휴리스틱하게 판별.

    - bullet/numbered list
    - 'Step 1', '1)' 등 단계 표시
    - Markdown heading / Summary / Conclusion
    - First/Next/Then/Finally 등의 단계 표현
    """
    # bullet list
    if re.search(r"^\s*[-*]\s+", text, flags=re.MULTILINE):
        return True

    # numbered list: 1. foo / 1) foo
    if re.search(r"^\s*\d+\.\s+", text, flags=re.MULTILINE):
        return True
    if re.search(r"^\s*\d+\)\s+", text, flags=re.MULTILINE):
        return True

    # explicit step/heading
    if re.search(r"Step\s+\d+", text, flags=re.IGNORECASE):
        return True
    if re.search(r"(###|####|\bSummary\b|\bConclusion\b)", text, flags=re.IGNORECASE):
        return True

    # 서술형 단계 표현
    if re.search(
        r"\b(First|Firstly|Second|Secondly|Third|Next|Then|Finally|In conclusion)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True

    return False


def sample_stack_python(n: int) -> List[Dict]:
    """
    The Stack (Python)에서 코드 양성 샘플을 스트리밍으로 수집.

    - label_code = 1 (코드)
    - label_harm = 0 (기본적으로 무해한 코드로 가정)
    - label_struct = 0 (구조성은 자연어 기준으로만 보되, 필요시 확장 가능)
    """
    ds = load_dataset(
        "bigcode/the-stack",
        data_dir="data/python",
        streaming=True,
        split="train",
    )
    out: List[Dict] = []
    it = iter(ds)
    while len(out) < n:
        try:
            s = next(it)
        except StopIteration:
            break

        code = s.get("content") or s.get("text") or ""
        if not code or not code.strip():
            continue

        out.append(
            {
                "text": code,
                "label_code": 1,
                "label_harm": 0,
                "label_struct": 0,
                "source": "stack",
            }
        )
    return out


def _combine_alpaca_fields(sample: Dict) -> str:
    """
    Alpaca 샘플의 instruction/input/output을 하나의 텍스트로 합친다.
    """
    instruction = sample.get("instruction") or ""
    input_text = sample.get("input") or ""
    output = sample.get("output") or ""

    parts = [instruction]
    if input_text:
        parts.append(input_text)
    parts.append(output)
    return "\n\n".join(p for p in parts if p)


def sample_alpaca_code(n: int) -> List[Dict]:
    """
    Alpaca에서 코드성 응답을 가진 샘플을 수집.

    - dataset 특성을 우선적으로 활용:
        · Alpaca는 instruction-following이지만, 코드 예제도 다수 포함
    - is_code_like로 코드성을 필터링하여 label_code=1 부여
    """
    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    out: List[Dict] = []
    it = iter(ds)
    while len(out) < n:
        try:
            s = next(it)
        except StopIteration:
            break

        combined = _combine_alpaca_fields(s)
        if not combined.strip():
            continue
        if not is_code_like(combined):
            continue

        out.append(
            {
                "text": combined,
                "label_code": 1,
                "label_harm": 0,
                "label_struct": int(is_structured(combined)),
                "source": "alpaca_code",
            }
        )
    return out


def sample_alpaca_noncode(n: int) -> List[Dict]:
    """
    Alpaca에서 코드성이 뚜렷하지 않은 일반 텍스트 샘플을 수집.

    - label_code = 0
    - label_harm = 0 (기본적으로 안전한 텍스트로 가정)
    - label_struct = 구조화 여부 휴리스틱 판별
    """
    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    out: List[Dict] = []
    it = iter(ds)
    while len(out) < n:
        try:
            s = next(it)
        except StopIteration:
            break

        combined = _combine_alpaca_fields(s)
        if not combined.strip():
            continue
        if is_code_like(combined):
            continue

        out.append(
            {
                "text": combined,
                "label_code": 0,
                "label_harm": 0,
                "label_struct": int(is_structured(combined)),
                "source": "alpaca_text",
            }
        )
    return out


def sample_lmsys_chat(
    n: int,
    dataset_name: str = "lmsys/lmsys-chat-1m",
    split: str = "train",
) -> List[Dict]:
    """
    LMSYS 대화 데이터셋에서 일반 대화 샘플을 수집.

    - label_code = 0 (일반 대화로 간주)
    - label_harm = OpenAI moderation 'flagged' 기준으로 0/1
    - label_struct = 0 (구조성은 LLM judge에서 보완)
    """
    if n <= 0:
        return []

    ds = load_dataset(dataset_name, split=split, streaming=True)
    out: List[Dict] = []
    it = iter(ds)

    while len(out) < n:
        try:
            s: Dict[str, Any] = next(it)
        except StopIteration:
            break

        conv = s.get("conversation") or []
        if not conv:
            continue

        parts: List[str] = []
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role") or "unknown"
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            parts.append(f"{role}: {content}")

        text = "\n".join(parts).strip()
        if not text:
            continue

        mods = s.get("openai_moderation") or []
        flagged_any = False
        for m in mods:
            if isinstance(m, dict) and m.get("flagged", False):
                flagged_any = True
                break

        label_harm = 1 if flagged_any else 0

        out.append(
            {
                "text": text,
                "label_code": 0,
                "label_harm": label_harm,
                "label_struct": 0,
                "source": "lmsys_chat",
            }
        )

    return out


def sample_toxic(n: int, pos_fraction: float = 0.5) -> List[Dict]:
    """
    RealToxicityPrompts에서 유해/비유해 텍스트를 스트리밍으로 수집.

    - toxicity score를 1차 라벨 근거로 사용
      · 높은 점수: harm=1
      · 낮은 점수: harm=0
    - pos_fraction 비율로 harm=1 / harm=0를 균형 있게 샘플링
    """
    if n <= 0:
        return []

    target_pos = int(n * pos_fraction)
    target_neg = n - target_pos

    ds = load_dataset("allenai/real-toxicity-prompts", split="train", streaming=True)
    it = iter(ds)

    pos_samples: List[Dict] = []
    neg_samples: List[Dict] = []

    while (len(pos_samples) < target_pos or len(neg_samples) < target_neg):
        try:
            s = next(it)
        except StopIteration:
            break

        prompt = s.get("prompt") or {}
        text = prompt.get("text") or ""
        if not text.strip():
            continue

        tox_score = s.get("toxicity", None)
        if tox_score is None:
            continue

        # 유해/비유해를 toxicity score 기반으로 구분
        if tox_score >= 0.7 and len(pos_samples) < target_pos:
            pos_samples.append(
                {
                    "text": text,
                    "label_code": 0,
                    "label_harm": 1,
                    "label_struct": int(is_structured(text)),
                    "source": "tox_prompt",
                }
            )
        elif tox_score <= 0.3 and len(neg_samples) < target_neg:
            neg_samples.append(
                {
                    "text": text,
                    "label_code": 0,
                    "label_harm": 0,
                    "label_struct": int(is_structured(text)),
                    "source": "tox_prompt",
                }
            )

    # 실제 수집된 샘플 수가 목표보다 적을 수 있으므로 그대로 결합
    return pos_samples + neg_samples


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    여러 공개 데이터셋을 기반으로 multi-label contrastive dataset을 구축한다.

    설계 원칙:
      - datasetSample.py에서 사용하는 각 원천 데이터셋의 특성을
        1차 라벨 근거로 삼는다.
      - 각 축(code / harm / struct)에 대해
        충분히 의미 있는 1/0 라벨이 나오도록 휴리스틱을 보완한다.

    출력 경로 및 샘플 개수는 Hydra config(config/dataset, config/paths)를 통해 제어한다.
    """
    mc_cfg = cfg.dataset.multi_contrast
    n_pos_stack = mc_cfg.n_pos_stack
    n_pos_alpaca = mc_cfg.n_pos_alpaca
    n_neg_alpaca = mc_cfg.n_neg_alpaca
    n_tox = mc_cfg.n_tox
    n_lmsys = getattr(mc_cfg, "n_lmsys_chat", 0)

    print("Sampling Stack Python (code=1)...")
    pos_stack = sample_stack_python(n_pos_stack)

    print("Sampling Alpaca code-like (code=1)...")
    pos_alpaca = sample_alpaca_code(n_pos_alpaca)

    print("Sampling Alpaca non-code (code=0)...")
    neg_alpaca = sample_alpaca_noncode(n_neg_alpaca)

    print("Sampling RealToxicityPrompts (harm=0/1)...")
    tox_samples = sample_toxic(n_tox, pos_fraction=0.5)

    if n_lmsys > 0:
        print("Sampling LMSYS chat (general dialogue)...")
        lmsys_samples = sample_lmsys_chat(
            n_lmsys,
            dataset_name=getattr(mc_cfg, "lmsys_chat_name", "lmsys/lmsys-chat-1m"),
            split=getattr(mc_cfg, "lmsys_chat_split", "train"),
        )
    else:
        lmsys_samples = []

    data = pos_stack + pos_alpaca + neg_alpaca + tox_samples + lmsys_samples
    random.shuffle(data)

    raw_path_cfg = cfg.paths.datasets.multi_contrast_raw
    out_path = to_absolute_path(raw_path_cfg)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved raw multi-contrast dataset to {out_path}")
    print(f"Total samples: {len(data)}")


if __name__ == "__main__":
    main()
