import json
import os
from glob import glob
from typing import Any, Dict, Tuple


LOG_ROOT = "experimentLog"


def load_log_files(run_dir: str) -> list[str]:
    """
    run 디렉토리 내 steering 로그 파일 목록(L*.json)을 반환한다.
    _eval.json / eval_summary.json 은 제외한다.
    """
    files = sorted(glob(os.path.join(run_dir, "L*.json")))
    return [
        f
        for f in files
        if not f.endswith("_eval.json") and not f.endswith("eval_summary.json")
    ]


def summarize_run(run_dir: str) -> Dict[str, Any] | None:
    """
    hook_comparison 필드를 사용해
    - layer_idx-1 에 hook을 건 경우
    - layer_idx 에 hook을 건 경우
    각각에 대해 steered feature(steering.feature_idx)의 Δf 평균을 계산한다.
    """
    log_files = load_log_files(run_dir)
    if not log_files:
        return None

    # variant: "minus1" (hook_layer_idx_minus1), "same" (hook_layer_idx_same)
    accum: Dict[str, Dict[str, float]] = {
        "minus1": {"sum_df": 0.0, "count": 0.0},
        "same": {"sum_df": 0.0, "count": 0.0},
    }

    for path in log_files:
        with open(path, "r", encoding="utf-8") as f:
            log = json.load(f)

        hook_comp = log.get("hook_comparison")
        if not isinstance(hook_comp, dict):
            continue

        steering_info = log.get("steering", {})
        feature_idx = int(steering_info.get("feature_idx", 0))

        for key, variant in [
            ("hook_layer_idx_minus1", "minus1"),
            ("hook_layer_idx_same", "same"),
        ]:
            res = hook_comp.get(key)
            if not isinstance(res, dict):
                continue

            feat_stats = res.get("feature_stats", {})
            indices = feat_stats.get("indices", [])
            f_before = feat_stats.get("f_before", [])
            f_after = feat_stats.get("f_after", [])

            if not indices or not f_before or not f_after:
                continue

            try:
                pos = indices.index(feature_idx)
            except ValueError:
                # steered feature가 indices에 포함되지 않은 경우 스킵
                continue

            try:
                fb = float(f_before[pos])
                fa = float(f_after[pos])
            except Exception:
                continue

            df = fa - fb
            accum[variant]["sum_df"] += df
            accum[variant]["count"] += 1.0

    def _mean(sum_val: float, cnt: float) -> float:
        return float(sum_val / cnt) if cnt > 0 else 0.0

    minus1_mean = _mean(accum["minus1"]["sum_df"], accum["minus1"]["count"])
    same_mean = _mean(accum["same"]["sum_df"], accum["same"]["count"])

    if accum["minus1"]["count"] == 0.0 and accum["same"]["count"] == 0.0:
        return None

    return {
        "minus1": {
            "mean_df": minus1_mean,
            "count": int(accum["minus1"]["count"]),
        },
        "same": {
            "mean_df": same_mean,
            "count": int(accum["same"]["count"]),
        },
    }


def main() -> None:
    if not os.path.isdir(LOG_ROOT):
        print(f"[ERROR] 로그 루트 디렉토리 '{LOG_ROOT}' 를 찾을 수 없습니다.")
        return

    run_dirs = sorted(
        os.path.join(LOG_ROOT, d)
        for d in os.listdir(LOG_ROOT)
        if os.path.isdir(os.path.join(LOG_ROOT, d))
    )

    if not run_dirs:
        print("[INFO] run 디렉토리가 없습니다.")
        return

    print("======== hook layer 위치별 steered feature Δf 요약 ========")

    for run_dir in run_dirs:
        summary = summarize_run(run_dir)
        if summary is None:
            continue

        ts = os.path.basename(run_dir)
        minus1 = summary["minus1"]
        same = summary["same"]

        print(f"[RUN] {ts}")
        print(
            f"  hook at layer_idx-1: Δf_mean={minus1['mean_df']:.4f} "
            f"(logs={minus1['count']})"
        )
        print(
            f"  hook at layer_idx  : Δf_mean={same['mean_df']:.4f} "
            f"(logs={same['count']})"
        )


if __name__ == "__main__":
    main()

