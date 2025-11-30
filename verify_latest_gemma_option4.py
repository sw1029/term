import os
import json
from glob import glob
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import OmegaConf

from SAE import H_SAE
from analysis import run_steering_analysis
from llm_judge import evaluate_experiment_run


LOG_ROOT = "experimentLog"


def find_latest_gemma_option4_run(
    log_root: str = LOG_ROOT,
) -> Tuple[str, str, str]:
    """
    experimentLog 이하에서
    - sae_layer_*_lossopt_4.pt 체크포인트가 존재하고
    - Gemma(goggle/gemma-2-2b-it)를 사용한 run
    들 중 가장 최근(timestamp 기준) run을 찾는다.

    Returns:
        (timestamp, run_dir, ckpt_path)
    """
    candidates = []

    for root, _, files in os.walk(log_root):
        # option4 SAE 체크포인트들
        ckpts: list[str] = []
        for f in files:
            if not (f.startswith("sae_layer_") and "lossopt_4" in f and f.endswith(".pt")):
                continue
            # 파일명에서 layer 인덱스 추출: sae_layer_{L}_lossopt_4.pt
            try:
                rest = f[len("sae_layer_") :]
                layer_str = rest.split("_", 1)[0]
                layer_val = int(layer_str)
            except Exception:
                continue
            # 이 스크립트에서는 layer_idx == 12 인 실험만 대상으로 삼는다.
            if layer_val == 12:
                ckpts.append(f)
        if not ckpts:
            continue

        # run 디렉토리 안의 대표 JSON 로그 하나를 읽어 모델 이름 확인
        json_files = sorted(
            f
            for f in files
            if f.startswith("L")
            and f.endswith(".json")
            and not f.endswith("_eval.json")
            and f != "eval_summary.json"
        )
        if not json_files:
            continue

        try:
            with open(os.path.join(root, json_files[0]), "r", encoding="utf-8") as jf:
                log = json.load(jf)
        except json.JSONDecodeError:
            continue

        if log.get("model_name") != "google/gemma-2-2b-it":
            continue

        run_dir = root
        ts = os.path.basename(run_dir)
        ckpt_paths = [os.path.join(root, c) for c in ckpts]
        candidates.append((ts, run_dir, ckpt_paths))

    if not candidates:
        raise RuntimeError(
            "Gemma + option4(lossopt_4) 체크포인트가 포함된 run 디렉토리를 찾지 못했습니다."
        )

    # timestamp 문자열 기준으로 최신 run 선택
    candidates.sort(key=lambda x: x[0], reverse=True)
    ts, run_dir, ckpt_paths = candidates[0]

    # 하나의 run 안에 여러 layer ckpt가 있을 수 있으므로, layer 인덱스가 가장 큰 것을 선택
    ckpt_paths.sort()
    ckpt_path = ckpt_paths[-1]

    return ts, run_dir, ckpt_path


def load_log_meta(run_dir: str) -> dict:
    """
    run 디렉토리에서 대표 L*.json 로그를 하나 로드해
    모델 이름 / layer_idx / steering 정보 / SAE 설정 등을 가져온다.
    """
    json_candidates = sorted(
        glob(os.path.join(run_dir, "L*.json"))
    )
    json_candidates = [
        p
        for p in json_candidates
        if not p.endswith("_eval.json") and not p.endswith("eval_summary.json")
    ]
    if not json_candidates:
        raise RuntimeError(f"run 디렉토리에서 L*.json 로그를 찾지 못했습니다: {run_dir}")

    with open(json_candidates[0], "r", encoding="utf-8") as f:
        log = json.load(f)
    return log


def build_sae_from_log(log: dict, ckpt_path: str, device: torch.device) -> H_SAE:
    """
    로그의 SAE 설정과 체크포인트(.pt)를 사용해 H_SAE 인스턴스를 복원한다.
    """
    sae_cfg = log.get("sae", {})
    input_dim = int(sae_cfg.get("input_dim"))
    sae_dim = int(sae_cfg.get("sae_dim"))
    jumprelu_bandwidth = float(sae_cfg.get("jumprelu_bandwidth", 0.001))
    jumprelu_init_threshold = float(sae_cfg.get("jumprelu_init_threshold", 0.001))

    sae = H_SAE(
        input_dim=input_dim,
        sae_dim=sae_dim,
        c_vectors=None,
        jumprelu_bandwidth=jumprelu_bandwidth,
        jumprelu_init_threshold=jumprelu_init_threshold,
    )

    # fixed_v_cnt는 state_dict에 저장되지 않으므로 로그에서 복구
    fixed_v_cnt = int(sae_cfg.get("fixed_v_cnt", 0))
    sae.fixed_v_cnt = fixed_v_cnt

    state = torch.load(ckpt_path, map_location=device)
    # feature_gnn 등 부가 모듈이 포함된 state_dict도 허용
    sae.load_state_dict(state, strict=False)

    sae.to(device)
    sae.eval()
    return sae


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ts, run_dir, ckpt_path = find_latest_gemma_option4_run()
    print(f"[INFO] 최신 Gemma + option4 run: {run_dir} (timestamp={ts})")
    print(f"[INFO] 사용 체크포인트: {ckpt_path}")

    log = load_log_meta(run_dir)

    model_name = log.get("model_name", "google/gemma-2-2b-it")
    layer_idx = int(log.get("layer_idx", 12))
    steering_info = log.get("steering", {})
    feature_idx = int(steering_info.get("feature_idx", 0))
    strength = float(steering_info.get("strength", 15.0))
    prompt = log.get("prompt", "Explain what this SAE is doing.")

    print(f"[INFO] model_name = {model_name}")
    print(f"[INFO] layer_idx = {layer_idx}, feature_idx = {feature_idx}, strength = {strength}")

    # LLM 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # SAE 복원
    sae = build_sae_from_log(log, ckpt_path, device=device)

    # steering vector: 해당 SAE feature의 decoder 행
    with torch.no_grad():
        steering_vector = sae.decoder.data[feature_idx].to(device)

    # steering 전/후 SAE feature 및 텍스트 비교
    result = run_steering_analysis(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        prompt=prompt,
        layer_idx=layer_idx,
        steering_vector=steering_vector,
        strength=strength,
    )

    f_before = result["f_before"][0, feature_idx].item()
    f_after = result["f_after"][0, feature_idx].item()

    print("\n[VERIFY] SAE steered feature activation")
    print(f"  f_before[{feature_idx}] = {f_before:.6f}")
    print(f"  f_after [{feature_idx}] = {f_after:.6f}")
    print(f"  Δf = {f_after - f_before:.6f}")

    print("\n[VERIFY] LLM output (truncated)")
    y_before = result["y_before"]
    y_after = result["y_after"]
    print("  --- before ---")
    print(" ", y_before[:400].replace("\n", " ") + ("..." if len(y_before) > 400 else ""))
    print("  --- after ---")
    print(" ", y_after[:400].replace("\n", " ") + ("..." if len(y_after) > 400 else ""))

    # OpenAI judge를 이용한 전체 run eval (원래 파이프라인과 동일한 방식)
    # 우선 config/judge/default.yaml 의 openai_api_key를 사용하고,
    # 없으면 환경변수 OPENAI_API_KEY를 fallback으로 사용한다.
    judge_model = "gpt-4o-mini"
    max_chars = 8000
    max_qps = 2.0

    # OpenAI API 키 로드: config 우선, 그 다음 환경변수
    openai_api_key = None
    judge_cfg_path = os.path.join("config", "judge", "default.yaml")
    if os.path.isfile(judge_cfg_path):
        try:
            judge_cfg = OmegaConf.load(judge_cfg_path)
            openai_api_key = judge_cfg.get("openai_api_key", None)
        except Exception:
            openai_api_key = None

    if not openai_api_key:
        openai_api_key = os.environ.get("OPENAI_API_KEY")

    print("\n[JUDGE] OpenAI 기반 eval 수행 시작")
    eval_result = evaluate_experiment_run(
        log_dir=run_dir,
        judge_model_name=judge_model,
        device=device.type,
        backend="openai",
        api_key=openai_api_key,
        openai_model_name=judge_model,
        max_chars=max_chars,
        max_qps=max_qps,
        cfg=None,
    )
    print(f"[JUDGE] eval_summary.json saved to: {eval_result['summary_path']}")
    print(f"[JUDGE] delta_mean: {eval_result['summary'].get('delta_mean')}")


if __name__ == "__main__":
    main()
