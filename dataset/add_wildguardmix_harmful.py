import json
import os
import sys
from pathlib import Path
from typing import Tuple

from datasets import load_dataset
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

# 프로젝트 루트(/home/ysw/term)를 import 경로에 추가해서
# `python dataset/add_wildguardmix_harmful.py`로 실행해도 llm_judge를 찾을 수 있게 함.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_judge import aggregate_labels_with_openai


def build_wildguardmix_harmful_raw(raw_path: str) -> Tuple[str, int]:
    """
    allenai/wildguardmix에서 harmful 응답만 추려서
    multi_contrast 포맷의 raw JSONL 파일로 저장한다.
    """
    abs_raw = to_absolute_path(raw_path)
    os.makedirs(os.path.dirname(abs_raw), exist_ok=True)

    train = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    test = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")

    def iter_harmful(split_ds, split_name: str):
        for row in split_ds:
            if row.get("response_harm_label") != "harmful":
                continue
            text = row.get("response") or ""
            if not text.strip():
                continue

            yield {
                "text": text,
                "label_code": 0,
                "label_harm": 1,
                "label_struct": 0,
                "source": f"wildguardmix_{split_name}",
                "wildguardmeta": {
                    "prompt": row.get("prompt"),
                    "prompt_harm_label": row.get("prompt_harm_label"),
                    "response_refusal_label": row.get("response_refusal_label"),
                    "response_harm_label": row.get("response_harm_label"),
                    "subcategory": row.get("subcategory"),
                    "adversarial": row.get("adversarial"),
                },
            }

    count = 0
    with open(abs_raw, "w", encoding="utf-8") as f:
        for split_ds, split_name in ((train, "train"), (test, "test")):
            for obj in iter_harmful(split_ds, split_name):
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1

    print(f"[WILDGUARD] Saved {count} harmful responses to {abs_raw}")
    return abs_raw, count


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    allenai/wildguardmix 데이터셋에서 harmful 응답만 추출하고,
    기존 multi_contrast 최종 데이터셋(multi_contrast_train_final)에
    llm_judge.py의 OpenAI 기반 라벨링 로직으로 라벨을 추가한다.
    """
    raw_rel = "dataset/wildguardmix_harmful_raw.jsonl"
    raw_abs, num_rows = build_wildguardmix_harmful_raw(raw_rel)

    if num_rows == 0:
        print("[WILDGUARD] No harmful rows found, nothing to label.")
        return

    api_key = os.environ.get("OPENAI_API_KEY") or getattr(
        cfg.judge, "openai_api_key", None
    )
    if not api_key:
        raise ValueError(
            "OpenAI API 키가 필요합니다. "
            "환경변수 OPENAI_API_KEY 또는 config.judge.openai_api_key를 설정하세요."
        )

    model_name = getattr(cfg.judge, "openai_model", cfg.judge.primary_model)
    judge_logs_root = cfg.paths.judge_logs_root
    max_chars = getattr(cfg.judge, "openai_max_chars", None)
    max_qps = getattr(cfg.judge, "openai_max_qps", None)

    output_path = cfg.paths.datasets.multi_contrast_final

    print(
        f"[WILDGUARD] Running LLM judge for {num_rows} harmful rows "
        f"and appending to {to_absolute_path(output_path)}"
    )

    aggregate_labels_with_openai(
        raw_path=raw_abs,
        output_path=output_path,
        api_key=api_key,
        model_name=model_name,
        judge_logs_root=judge_logs_root,
        max_chars=max_chars,
        resume=True,
        max_qps=max_qps,
    )

    print("[WILDGUARD] Labeling finished.")


if __name__ == "__main__":
    main()
