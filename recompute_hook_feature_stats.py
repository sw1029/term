import argparse
import gc
import json
import os
from glob import glob
from multiprocessing import Process
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from SAE import H_SAE
from analysis import run_steering_analysis_compare_hooks


LOG_ROOT = "experimentLog"


def find_log_files(run_dir: str) -> list[str]:
    """
    run 디렉토리 내 steering 로그 파일 목록(L*.json)을 반환한다.
    _eval.json / eval_summary.json 은 제외한다.
    """
    pattern = os.path.join(run_dir, "L*.json")
    files = sorted(glob(pattern))
    return [
        f
        for f in files
        if not f.endswith("_eval.json") and not f.endswith("eval_summary.json")
    ]


def load_log(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_log(path: str, log: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def find_ckpt_for_log(run_dir: str, log: Dict[str, Any]) -> str | None:
    """
    로그 정보(layer_idx, loss_option)를 사용해 해당 run 디렉토리 안의
    SAE 체크포인트 경로를 찾는다.
    """
    layer_idx = int(log.get("layer_idx"))
    sae_cfg = log.get("sae", {})
    loss_option = int(sae_cfg.get("loss_option", 1))

    expected = os.path.join(
        run_dir,
        f"sae_layer_{layer_idx}_lossopt_{loss_option}.pt",
    )
    if os.path.isfile(expected):
        return expected

    # fallback: 같은 layer_idx 를 가진 어떤 체크포인트든 사용
    pattern = os.path.join(run_dir, f"sae_layer_{layer_idx}_lossopt_*.pt")
    candidates = sorted(glob(pattern))
    if candidates:
        return candidates[0]
    return None


def build_sae_from_log(log: Dict[str, Any], ckpt_path: str, device: torch.device) -> H_SAE:
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

    fixed_v_cnt = int(sae_cfg.get("fixed_v_cnt", 0))
    sae.fixed_v_cnt = fixed_v_cnt

    state = torch.load(ckpt_path, map_location=device)
    sae.load_state_dict(state, strict=False)

    sae.to(device)
    sae.eval()
    return sae


def recompute_for_run(
    run_dir: str,
    device: torch.device,
    dry_run: bool = False,
    top_n: int = 10,
    max_prompt_tokens: int = 128,
) -> None:
    """
    하나의 run 디렉토리에 대해:
      - L*.json steering 로그를 읽고
      - SAE 체크포인트를 로드한 뒤
      - hook_layer_idx_minus1 / hook_layer_idx_same 기준으로
        feature_stats 및 hook_comparison 를 재계산하여 덮어쓴다.
    """
    log_files = find_log_files(run_dir)
    if not log_files:
        return

    # 대표 로그 하나로 모델 이름 / layer_idx 등을 확인
    sample_log = load_log(log_files[0])
    model_name = sample_log.get("model_name", None)
    if model_name is None:
        print(f"[WARN] {run_dir}: model_name 이 없어 스킵합니다.")
        return

    print(f"[INFO] Processing run: {run_dir} (model={model_name})")

    tokenizer = None
    model = None
    sae_cache: Dict[Tuple[int, int], H_SAE] = {}

    try:
        # LLM 로드 (run 단위로 한 번만)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        model.eval()

        # (layer_idx, loss_option) -> SAE 캐시
        sae_cache = {}

        for path in log_files:
            log = load_log(path)

            layer_idx = int(log.get("layer_idx", 0))
            sae_cfg = log.get("sae", {})
            loss_option = int(sae_cfg.get("loss_option", 1))

            ckpt_path = find_ckpt_for_log(run_dir, log)
            if ckpt_path is None:
                print(
                    f"[WARN] {path}: SAE 체크포인트를 찾지 못했습니다. "
                    f"(layer_idx={layer_idx}, loss_option={loss_option})"
                )
                continue

            cache_key = (layer_idx, loss_option)
            if cache_key not in sae_cache:
                sae_model = build_sae_from_log(log, ckpt_path, device=device)
                sae_cache[cache_key] = sae_model
            else:
                sae_model = sae_cache[cache_key]

            steering_info = log.get("steering", {})
            feature_idx = int(steering_info.get("feature_idx", 0))
            strength = float(steering_info.get("strength", 0.0))
            prompt = log.get("prompt", "")

            # 너무 긴 프롬프트로 인한 OOM 방지를 위해 토큰 길이 제한을 적용
            def _truncate_prompt(text: str, max_tokens: int) -> str:
                if max_tokens is None or max_tokens <= 0 or not text:
                    return text
                tokens = tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_tokens,
                )
                ids = tokens.get("input_ids", [])
                if not ids:
                    return text
                return tokenizer.decode(ids, skip_special_tokens=True)

            truncated_prompt = _truncate_prompt(prompt, max_prompt_tokens)
            if truncated_prompt != prompt:
                # 재계산된 feature_stats와 일관성을 위해 로그의 prompt도 갱신
                log["prompt"] = truncated_prompt
                prompt = truncated_prompt

            with torch.no_grad():
                steering_vector = sae_model.decoder.data[feature_idx].to(device)

            # hook_layer_idx_minus1 / same 두 경우 모두 재계산
            hook_results = run_steering_analysis_compare_hooks(
                model=model,
                tokenizer=tokenizer,
                sae=sae_model,
                prompt=prompt,
                layer_idx=layer_idx,
                steering_vector=steering_vector,
                strength=strength,
            )

            res_minus1 = hook_results.get("hook_layer_idx_minus1")
            res_same = hook_results.get("hook_layer_idx_same")
            if not isinstance(res_minus1, dict) or not isinstance(res_same, dict):
                print(f"[WARN] {path}: hook_results 형식이 예상과 다릅니다. 스킵.")
                continue

            # 기존 feature_stats.indices 를 우선 사용 (없으면 steered feature만)
            old_feat = log.get("feature_stats", {})
            indices = old_feat.get("indices", None)
            if not indices:
                indices = [feature_idx]

            indices_tensor = torch.tensor(indices, dtype=torch.long, device=res_minus1["f_before"].device)

            def _extract_stats(res: Dict[str, Any]) -> Dict[str, Any]:
                f_before_t = res["f_before"][0]  # (d_sae,)
                f_after_t = res["f_after"][0]
                g_before_t = res.get("g_before", None)
                g_after_t = res.get("g_after", None)

                f_before_sel = f_before_t[indices_tensor].tolist()
                f_after_sel = f_after_t[indices_tensor].tolist()

                if g_before_t is not None and g_after_t is not None:
                    g_before_sel = g_before_t[0, indices_tensor].tolist()
                    g_after_sel = g_after_t[0, indices_tensor].tolist()
                else:
                    g_before_sel = [0.0 for _ in indices]
                    g_after_sel = [0.0 for _ in indices]

                return {
                    "indices": [int(i) for i in indices],
                    "f_before": f_before_sel,
                    "f_after": f_after_sel,
                    "g_before": g_before_sel,
                    "g_after": g_after_sel,
                }

            def _extract_topn(res: Dict[str, Any], k: int) -> Dict[str, Any]:
                """
                steering 전/후 Δf 기준 상위 k개 feature를 요약한다.
                전체 SAE 차원(d_sae)에서 top-k를 고르며,
                각 feature에 대해 index / f_before / f_after / delta_f 를 기록한다.
                """
                f_before_t = res["f_before"][0]  # (d_sae,)
                f_after_t = res["f_after"][0]
                delta = f_after_t - f_before_t
                k = min(int(k), delta.numel())
                if k <= 0:
                    return {"features": []}

                # Δf 절댓값이 큰 순으로 상위 k개 선택
                _, top_idx = torch.topk(delta.abs(), k=k, largest=True)

                feats = []
                for idx_val in top_idx.tolist():
                    fb = float(f_before_t[idx_val].item())
                    fa = float(f_after_t[idx_val].item())
                    df = fa - fb
                    feats.append(
                        {
                            "index": int(idx_val),
                            "f_before": fb,
                            "f_after": fa,
                            "delta_f": df,
                        }
                    )
                return {"features": feats, "k": k}

            # feature_stats (hook_layer_idx_minus1 기준) 재기입
            new_feature_stats = _extract_stats(res_minus1)
            new_feature_stats["num_batches_for_stats"] = int(
                old_feat.get("num_batches_for_stats", 0)
            )
            log["feature_stats"] = new_feature_stats

            # steering.hook_layer_idx 를 layer_idx-1 기준으로 갱신
            hook_layer_idx_minus1 = int(res_minus1.get("hook_layer_idx", max(layer_idx - 1, 0)))
            steering_info["hook_layer_idx"] = hook_layer_idx_minus1
            log["steering"] = steering_info

            # hook_comparison 전체 갱신
            hook_comparison = {
                "hook_layer_idx_minus1": {
                    "hook_layer_idx": hook_layer_idx_minus1,
                    "feature_stats": _extract_stats(res_minus1),
                    "top_activated_features": _extract_topn(res_minus1, top_n),
                    "output": {
                        "before": res_minus1.get("y_before", ""),
                        "after": res_minus1.get("y_after", ""),
                    },
                },
                "hook_layer_idx_same": {
                    "hook_layer_idx": int(res_same.get("hook_layer_idx", layer_idx)),
                    "feature_stats": _extract_stats(res_same),
                    "top_activated_features": _extract_topn(res_same, top_n),
                    "output": {
                        "before": res_same.get("y_before", ""),
                        "after": res_same.get("y_after", ""),
                    },
                },
            }
            log["hook_comparison"] = hook_comparison

            if dry_run:
                print(f"[DRY-RUN] {path} 업데이트 예정 (indices={indices})")
            else:
                save_log(path, log)
                print(f"[UPDATE] {path} 를 재기입했습니다.")

    finally:
        # run 단위로 LLM/SAE 캐시를 명시적으로 정리해 메모리 사용량을 줄인다.
        for _, sae_model in sae_cache.items():
            try:
                sae_model.to("cpu")
            except Exception:
                pass
        sae_cache.clear()

        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
            del model
        if tokenizer is not None:
            del tokenizer

        gc.collect()
        if device.type == "cuda":  # type: ignore[attr-defined]
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "experimentLog 이하를 재귀적으로 순회하며, "
            "각 run 디렉토리의 SAE 체크포인트(.pt)와 steering 로그(L*.json)를 이용해 "
            "hook(layer_idx-1) 기준 feature_stats 및 hook_comparison 를 재계산해 덮어씁니다."
        )
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default=LOG_ROOT,
        help=f"실험 로그 루트 디렉토리 (기본: {LOG_ROOT})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda 또는 cpu (지정하지 않으면 각 프로세스에서 자동 선택)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="실제 파일을 덮어쓰지 않고 어떤 파일이 업데이트될지만 출력",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="steering 전후 Δf 기준 상위 N개 feature를 요약에 포함 (기본: 10)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="병렬 처리에 사용할 프로세스 수 (기본: 4)",
    )
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=128,
        help="프롬프트 토큰 최대 길이 (0 이하면 제한 없음, 기본: 128)",
    )

    args = parser.parse_args()

    log_root = args.log_root
    if not os.path.isdir(log_root):
        print(f"[ERROR] 로그 루트 디렉토리를 찾을 수 없습니다: {log_root}")
        return

    device_str = args.device  # 각 워커에서 torch.device(...)로 변환
    top_n = max(int(args.top_n), 1)
    num_workers = max(int(args.num_workers), 1)
    max_prompt_tokens = int(args.max_prompt_tokens)

    # 1) experimentLog 이하를 재귀적으로 탐색해 run 디렉토리 목록 수집
    run_dirs: list[str] = []
    for root, dirs, files in os.walk(log_root):
        if any(fname.startswith("L") and fname.endswith(".json") for fname in files):
            run_dirs.append(root)

    if not run_dirs:
        print("[INFO] 처리할 run 디렉토리가 없습니다.")
        return

    print(f"[INFO] Found {len(run_dirs)} run directories under '{log_root}'")

    # 2) 4분할(또는 num_workers 분할)하여 병렬 처리
    num_workers = min(num_workers, len(run_dirs))
    if num_workers <= 1:
        print("[INFO] Single-process 모드로 실행합니다.")
        recompute_worker(
            run_dirs,
            device_str=device_str,
            dry_run=args.dry_run,
            top_n=top_n,
            max_prompt_tokens=max_prompt_tokens,
        )
        return

    print(f"[INFO] Using {num_workers} worker processes")

    # round-robin 방식으로 분할
    chunks: list[list[str]] = [[] for _ in range(num_workers)]
    for i, rd in enumerate(run_dirs):
        chunks[i % num_workers].append(rd)

    procs: list[Process] = []
    for wid, chunk in enumerate(chunks):
        if not chunk:
            continue
        p = Process(
            target=recompute_worker,
            args=(chunk,),
            kwargs={
                "device_str": device_str,
                "dry_run": args.dry_run,
                "top_n": top_n,
                "max_prompt_tokens": max_prompt_tokens,
            },
        )
        p.start()
        procs.append(p)
        print(f"[INFO] Started worker {wid} with {len(chunk)} runs")

    for p in procs:
        p.join()


def recompute_worker(
    run_dirs: list[str],
    device_str: str | None,
    dry_run: bool,
    top_n: int,
    max_prompt_tokens: int,
) -> None:
    """
    주어진 run 디렉토리 목록에 대해 recompute_for_run 을 순차 실행하는 워커 함수.

    각 프로세스는 자신만의 torch.device 를 구성한다.
    """
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[WORKER] PID={os.getpid()}, device={device}, runs={len(run_dirs)}")

    for run_dir in run_dirs:
        recompute_for_run(
            run_dir,
            device=device,
            dry_run=dry_run,
            top_n=top_n,
            max_prompt_tokens=max_prompt_tokens,
        )


if __name__ == "__main__":
    main()
