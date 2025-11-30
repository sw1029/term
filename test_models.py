"""
간단한 모델 호출 가능 여부 테스트 스크립트.

Hydra config의 judge 섹션에 정의된:
  - primary_model
  - code_models
  - harm_models
  - struct_models
에 대해 각각 한 번씩 로드 및 짧은 inference를 수행하여
정상 동작 여부를 확인한다.
"""

import hydra
from omegaconf import DictConfig
import torch

from llm_judge import load_judge_model


def quick_infer(model, tokenizer, device: str):
    """
    아주 짧은 프롬프트로 1~2 토큰만 생성해보는 간단한 체크.
    """
    prompt = "Test"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=2)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = cfg.judge.device if torch.cuda.is_available() else "cpu"

    model_names = [cfg.judge.primary_model]
    model_names += cfg.judge.code_models
    model_names += cfg.judge.harm_models
    model_names += cfg.judge.struct_models

    # 중복 제거
    seen = set()
    unique_models = []
    for m in model_names:
        if m not in seen:
            seen.add(m)
            unique_models.append(m)

    print(f"[TEST] Checking {len(unique_models)} models on device={device}")

    loader_cfg = getattr(cfg.judge, "hf_loader", None)

    for name in unique_models:
        try:
            print(f"[LOAD] {name}")
            mdl, tok = load_judge_model(
                name,
                device=device,
                loader_cfg=loader_cfg,
            )
            quick_infer(mdl, tok, device)
            print(f"[OK] {name}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")


if __name__ == "__main__":
    main()
