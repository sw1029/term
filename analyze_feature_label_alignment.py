import argparse
import json
import os
from glob import glob
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from SAE import H_SAE
from utils import get_batch_activations


LOG_ROOT = "experimentLog"


def find_run_dir(explicit: str | None = None) -> str:
    """
    분석 대상 run 디렉토리를 결정한다.

    - explicit 이 주어지면 그대로 사용.
    - 없으면 experimentLog 아래 timestamp 디렉토리 중 가장 최신 것을 선택.
    """
    if explicit is not None:
        if not os.path.isdir(explicit):
            raise FileNotFoundError(f"지정한 run 디렉토리를 찾을 수 없습니다: {explicit}")
        return explicit

    if not os.path.isdir(LOG_ROOT):
        raise FileNotFoundError(f"로그 루트 디렉토리를 찾을 수 없습니다: {LOG_ROOT}")

    candidates = [
        d
        for d in os.listdir(LOG_ROOT)
        if os.path.isdir(os.path.join(LOG_ROOT, d))
    ]
    if not candidates:
        raise RuntimeError(f"{LOG_ROOT} 아래에 run 디렉토리가 없습니다.")

    candidates.sort()
    ts = candidates[-1]
    return os.path.join(LOG_ROOT, ts)


def load_log_meta(run_dir: str) -> Dict[str, Any]:
    """
    run 디렉토리에서 대표 L*.json 로그를 하나 로드해
    모델 / 레이어 / SAE 설정 등을 가져온다.
    """
    json_candidates = sorted(glob(os.path.join(run_dir, "L*.json")))
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


def build_sae_from_log(log: Dict[str, Any], ckpt_path: str, device: torch.device) -> H_SAE:
    """
    로그의 SAE 설정과 체크포인트(.pt)를 사용해 H_SAE 인스턴스를 복원한다.
    (verify_latest_gemma_option4.py 의 로직을 재사용)
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


def find_ckpt_for_log(run_dir: str, log: Dict[str, Any]) -> str:
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
    patterns = glob(os.path.join(run_dir, f"sae_layer_{layer_idx}_lossopt_*.pt"))
    if not patterns:
        raise FileNotFoundError(
            f"run 디렉토리에서 layer={layer_idx} SAE 체크포인트를 찾지 못했습니다: {run_dir}"
        )
    patterns.sort()
    return patterns[0]


def load_paths_cfg() -> Dict[str, Any]:
    """
    config/paths/default.yaml 을 읽어 dataset 경로 등을 가져온다.
    """
    path_cfg_path = os.path.join("config", "paths", "default.yaml")
    if not os.path.isfile(path_cfg_path):
        raise FileNotFoundError(f"paths 설정 파일을 찾을 수 없습니다: {path_cfg_path}")

    cfg = OmegaConf.load(path_cfg_path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[no-any-return]


def compute_basic_stats(values: List[float]) -> Tuple[float, float]:
    """
    단순 평균/표준편차 계산.
    """
    if not values:
        return 0.0, 0.0
    n = float(len(values))
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / max(n - 1.0, 1.0)
    std = var ** 0.5
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "multi_contrast_final 데이터의 label(code/harm/struct)과 "
            "SAE concept feature 활성(f)을 정량적으로 정렬 여부를 분석합니다."
        )
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="분석할 run 디렉토리 (기본: experimentLog 아래 최신 timestamp)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="분석에 사용할 최대 샘플 수 (기본: 5000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="LLM/SAE forward 시 사용할 배치 크기 (기본: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda 또는 cpu (기본: 사용 가능하면 cuda, 아니면 cpu)",
    )

    args = parser.parse_args()

    run_dir = find_run_dir(args.run_dir)
    ts = os.path.basename(run_dir)
    print(f"[INFO] 분석 대상 run 디렉토리: {run_dir} (timestamp={ts})")

    log = load_log_meta(run_dir)
    model_name = log.get("model_name", "google/gemma-2-2b-it")
    layer_idx = int(log.get("layer_idx", 12))
    sae_cfg = log.get("sae", {})
    concept_indices: Dict[str, Any] = sae_cfg.get("concept_feature_indices", {})

    if not concept_indices:
        raise RuntimeError("로그에서 concept_feature_indices 를 찾지 못했습니다.")

    exp_cfg = log.get("experiment", {})
    use_multi_contrast = bool(exp_cfg.get("use_multi_contrast", False))
    if not use_multi_contrast:
        print(
            "[WARN] 이 run 은 use_multi_contrast=false 로 실행되었습니다. "
            "multi_contrast_final 라벨과의 정렬 분석에는 덜 적합할 수 있습니다."
        )

    # device 선택
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # LLM 로드
    print(f"[INFO] LLM 로드: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # SAE 로드
    ckpt_path = find_ckpt_for_log(run_dir, log)
    print(f"[INFO] SAE 체크포인트 로드: {ckpt_path}")
    sae = build_sae_from_log(log, ckpt_path, device=device)

    # multi_contrast_final 데이터 로드
    paths_cfg = load_paths_cfg()
    datasets_cfg = paths_cfg.get("datasets", {})
    multi_contrast_path = datasets_cfg.get("multi_contrast_final")
    if multi_contrast_path is None:
        raise RuntimeError("paths 설정에서 datasets.multi_contrast_final 을 찾지 못했습니다.")

    print(f"[INFO] multi_contrast_final 데이터 로드: {multi_contrast_path}")
    ds = load_dataset("json", data_files={"train": multi_contrast_path})["train"]

    # concept feature별 / label별 활성값 수집
    # stats[name]["pos_vals"], ["neg_vals"]
    stats: Dict[str, Dict[str, List[float]]] = {}
    for name in concept_indices.keys():
        stats[name] = {
            "pos_vals": [],
            "neg_vals": [],
        }

    total_seen = 0
    idx = 0
    n_dataset = len(ds)

    print(
        f"[INFO] 데이터셋 크기: {n_dataset} 샘플, "
        f"분석 최대 샘플 수: {args.max_samples}"
    )

    while idx < n_dataset and total_seen < args.max_samples:
        end = min(idx + args.batch_size, n_dataset)
        rows = [ds[i] for i in range(idx, end)]
        idx = end

        # text / label 추출
        texts: List[str] = []
        label_rows: List[Dict[str, int]] = []
        for row in rows:
            text = str(row.get("text", "")).strip()
            if not text:
                continue

            labels_for_row: Dict[str, int] = {}
            missing_label = False
            for cname in concept_indices.keys():
                key = f"label_{cname}"
                if key not in row:
                    missing_label = True
                    break
                try:
                    labels_for_row[cname] = int(row.get(key, 0))
                except Exception:
                    missing_label = True
                    break

            if missing_label:
                continue

            texts.append(text)
            label_rows.append(labels_for_row)

        if not texts:
            continue

        # LLM hidden → SAE feature f 계산
        with torch.no_grad():
            x = get_batch_activations(model, tokenizer, texts, layer_idx, device)
            out = sae(x)
            if isinstance(out, tuple):
                _, f = out[0], out[1]
            else:
                _, f = out, None

        if f is None:
            continue

        f_cpu = f.detach().cpu()
        batch_size_actual = f_cpu.size(0)

        for i in range(batch_size_actual):
            if total_seen >= args.max_samples:
                break

            row_labels = label_rows[i]
            for cname, idx_feature in concept_indices.items():
                f_idx = int(idx_feature)
                if f_idx < 0 or f_idx >= f_cpu.size(1):
                    continue

                val = float(f_cpu[i, f_idx].item())
                lbl = int(row_labels.get(cname, 0))
                if lbl == 1:
                    stats[cname]["pos_vals"].append(val)
                else:
                    stats[cname]["neg_vals"].append(val)

            total_seen += 1

    print(f"[INFO] 실제 사용된 샘플 수: {total_seen}")

    # 개념별로 평균/표준편차 및 간단한 효과 크기 계산
    alignment_summary: Dict[str, Any] = {}

    for cname, buf in stats.items():
        pos_vals = buf["pos_vals"]
        neg_vals = buf["neg_vals"]
        n_pos = len(pos_vals)
        n_neg = len(neg_vals)

        mean_pos, std_pos = compute_basic_stats(pos_vals)
        mean_neg, std_neg = compute_basic_stats(neg_vals)
        diff = mean_pos - mean_neg

        # pooled std 기반 Cohen's d
        if n_pos > 1 and n_neg > 1:
            num = ((n_pos - 1) * (std_pos ** 2)) + ((n_neg - 1) * (std_neg ** 2))
            denom = max(n_pos + n_neg - 2, 1)
            pooled_var = num / denom
            pooled_std = pooled_var ** 0.5
            d = diff / pooled_std if pooled_std > 0 else 0.0
        else:
            d = 0.0

        alignment_summary[cname] = {
            "feature_index": int(concept_indices[cname]),
            "num_pos": n_pos,
            "num_neg": n_neg,
            "mean_pos": mean_pos,
            "mean_neg": mean_neg,
            "mean_diff": diff,
            "std_pos": std_pos,
            "std_neg": std_neg,
            "cohens_d": d,
        }

    # 결과 출력
    print("\n======== SAE concept feature vs multi_contrast label 정렬 분석 ========")
    print(f"[RUN] {ts} — model={model_name}, layer_idx={layer_idx}")
    for cname, info in alignment_summary.items():
        print(
            f"- concept '{cname}' (feature_idx={info['feature_index']}): "
            f"pos_mean={info['mean_pos']:.4f} (n={info['num_pos']}), "
            f"neg_mean={info['mean_neg']:.4f} (n={info['num_neg']}), "
            f"Δ={info['mean_diff']:.4f}, d={info['cohens_d']:.3f}"
        )

    # run 디렉토리에 JSON으로도 저장
    out_path = os.path.join(run_dir, "feature_label_alignment.json")
    payload = {
        "run_dir": ts,
        "model_name": model_name,
        "layer_idx": layer_idx,
        "total_samples": total_seen,
        "alignment": alignment_summary,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] alignment 요약을 저장했습니다: {out_path}")


if __name__ == "__main__":
    main()

