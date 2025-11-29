import json
import os
from glob import glob
from typing import Any, Dict, List, Optional, Tuple


LOG_ROOT = "experimentLog"


def load_run_summary(run_dir: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    한 run 디렉토리(<timestamp>)에서 eval_summary와 메타 정보를 로드한다.

    Returns:
        (summary, meta) 또는 eval_summary.json 이 없으면 None.
    """
    eval_path = os.path.join(run_dir, "eval_summary.json")
    if not os.path.isfile(eval_path):
        return None

    with open(eval_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    meta: Dict[str, Any] = {"run_dir": os.path.basename(run_dir)}

    # 대표 steering 로그 하나를 골라 모델/SAE 설정을 가져온다.
    json_pattern = os.path.join(run_dir, "L*.json")
    json_files = sorted(
        p
        for p in glob(json_pattern)
        if not p.endswith("_eval.json") and not p.endswith("eval_summary.json")
    )

    if json_files:
        with open(json_files[0], "r", encoding="utf-8") as lf:
            log = json.load(lf)

        meta["model_name"] = log.get("model_name")
        meta["command"] = log.get("command")

        sae_info = log.get("sae", {})
        meta["loss_option"] = sae_info.get("loss_option")
        meta["loss_module"] = sae_info.get("loss_module")
        meta["use_jump_relu"] = sae_info.get("use_jump_relu")
        meta["l0_coeff"] = sae_info.get("l0_coeff")
        meta["guidance_method"] = sae_info.get("guidance_method")
        meta["role_sep_coeff"] = sae_info.get("role_sep_coeff")
        meta["alpha_concept"] = sae_info.get("alpha_concept")

        steer_info = log.get("steering", {})
        meta["steered_label"] = steer_info.get("label")
        meta["steering_strength"] = steer_info.get("strength")

    return summary, meta


def format_mean_triplet(mean_dict: Dict[str, float]) -> str:
    """
    {"code": x, "harm": y, "struct": z} 형태를 간단한 문자열로 포맷.
    """
    code = mean_dict.get("code", 0.0)
    harm = mean_dict.get("harm", 0.0)
    struct = mean_dict.get("struct", 0.0)
    return f"code={code:.3f}, harm={harm:.3f}, struct={struct:.3f}"


def compute_feature_separation(summary: Dict[str, Any]) -> Tuple[float, float]:
    """
    steering label 별 delta_mean을 기반으로,
    - self_delta_mean: 각 label이 자신의 축에서 올리는 평균 Δ
    - cross_delta_mean: 다른 축들에서의 |Δ| 평균
    을 계산해 feature 간 분리 정도를 간단히 요약한다.
    """
    by_label = summary.get("by_steer_label", {})
    if not by_label:
        return 0.0, 0.0

    diag_vals: List[float] = []
    offdiag_vals: List[float] = []

    for label_name, stats in by_label.items():
        delta = stats.get("delta_mean", {})
        for key in ("code", "harm", "struct"):
            v = float(delta.get(key, 0.0))
            if key == label_name:
                diag_vals.append(v)
            else:
                offdiag_vals.append(abs(v))

    if not diag_vals:
        return 0.0, float(sum(offdiag_vals) / max(len(offdiag_vals), 1)) if offdiag_vals else 0.0

    self_mean = float(sum(diag_vals) / len(diag_vals))
    cross_mean = float(sum(offdiag_vals) / max(len(offdiag_vals), 1)) if offdiag_vals else 0.0
    return self_mean, cross_mean


def summarize_run(summary: Dict[str, Any], meta: Dict[str, Any]) -> str:
    """
    단일 run(eval_summary.json + meta)에 대한 자연어 요약을 생성한다.
    """
    lines: List[str] = []
    run_dir = meta.get("run_dir", "?")
    model_name = meta.get("model_name", "unknown_model")

    loss_option = meta.get("loss_option")
    loss_module = meta.get("loss_module")
    guidance_method = meta.get("guidance_method")
    use_jump_relu = meta.get("use_jump_relu")
    l0_coeff = meta.get("l0_coeff")
    role_sep_coeff = meta.get("role_sep_coeff")
    alpha_concept = meta.get("alpha_concept")

    lines.append(f"[RUN] {run_dir} — 모델: {model_name}")

    config_desc_parts: List[str] = []
    if loss_option is not None:
        config_desc_parts.append(f"loss_option={loss_option}")
    if loss_module:
        config_desc_parts.append(f"loss_module={loss_module}")
    if guidance_method:
        config_desc_parts.append(f"guidance_method={guidance_method}")
    if use_jump_relu is not None:
        config_desc_parts.append(f"JumpReLU={bool(use_jump_relu)}")
    if l0_coeff is not None:
        config_desc_parts.append(f"L0_coeff={l0_coeff}")
    if role_sep_coeff is not None:
        config_desc_parts.append(f"role_sep_coeff={role_sep_coeff}")
    if alpha_concept:
        config_desc_parts.append(f"alpha_concept={alpha_concept}")

    if config_desc_parts:
        lines.append("  설정: " + ", ".join(str(p) for p in config_desc_parts))

    num_logs = summary.get("num_logs", 0)
    lines.append(f"  평가한 steering 로그 수: {num_logs}")

    before_mean = summary.get("before_mean", {})
    after_mean = summary.get("after_mean", {})
    delta_mean = summary.get("delta_mean", {})

    lines.append(
        "  전체 평균 라벨 (before → after, Δ): "
        f"{format_mean_triplet(before_mean)} → "
        f"{format_mean_triplet(after_mean)}, "
        f"Δ={format_mean_triplet(delta_mean)}"
    )

    by_label = summary.get("by_steer_label", {})
    if by_label:
        lines.append("  steering label 별 효과:")
        for label_name, stats in by_label.items():
            lbl_before = stats.get("before_mean", {})
            lbl_after = stats.get("after_mean", {})
            lbl_delta = stats.get("delta_mean", {})
            count = stats.get("num_logs", 0)
            lines.append(
                f"    - {label_name} (로그 {count}개): "
                f"{format_mean_triplet(lbl_before)} → "
                f"{format_mean_triplet(lbl_after)}, "
                f"Δ={format_mean_triplet(lbl_delta)}"
            )

        self_mean, cross_mean = compute_feature_separation(summary)
        lines.append(
            "  feature 분리 지표 (라벨 self Δ vs cross Δ 평균): "
            f"self={self_mean:.3f}, cross={cross_mean:.3f}"
        )

    return "\n".join(lines)


def main() -> None:
    if not os.path.isdir(LOG_ROOT):
        print(f"[ERROR] 로그 루트 디렉토리 '{LOG_ROOT}' 를 찾을 수 없습니다.")
        return

    run_dirs = sorted(
        os.path.join(LOG_ROOT, d)
        for d in os.listdir(LOG_ROOT)
        if os.path.isdir(os.path.join(LOG_ROOT, d))
    )

    collected: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    for run_dir in run_dirs:
        loaded = load_run_summary(run_dir)
        if loaded is None:
            continue
        summary, meta = loaded
        collected.append((summary, meta))

    if not collected:
        print("[INFO] eval_summary.json 이 있는 run 디렉토리가 없습니다.")
        return

    print("======== 실험별 요약 ========")
    for summary, meta in collected:
        print(summarize_run(summary, meta))
        print()

    # run 간 비교: code/harm/struct delta_mean 기준으로 간단한 best run 리포트
    def _get_delta(s: Dict[str, Any], key: str) -> float:
        return float(s.get("delta_mean", {}).get(key, 0.0))

    best_code = max(collected, key=lambda sm: _get_delta(sm[0], "code"))
    best_harm = max(collected, key=lambda sm: _get_delta(sm[0], "harm"))
    best_struct = max(collected, key=lambda sm: _get_delta(sm[0], "struct"))

    print("======== 전체 비교 (delta_mean 기준) ========")
    for metric_name, best in [
        ("code", best_code),
        ("harm", best_harm),
        ("struct", best_struct),
    ]:
        summary, meta = best
        run_dir = meta.get("run_dir", "?")
        delta_val = _get_delta(summary, metric_name)
        loss_option = meta.get("loss_option")
        guidance_method = meta.get("guidance_method")
        print(
            f"- {metric_name} Δ가 가장 큰 run: {run_dir} "
            f"(loss_option={loss_option}, guidance_method={guidance_method}, Δ={delta_val:.3f})"
        )


if __name__ == "__main__":
    main()
