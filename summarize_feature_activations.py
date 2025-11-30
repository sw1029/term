import json
import os
from typing import Any, Dict, Tuple


LOG_ROOT = "experimentLog"


def load_feature_activation_summary(run_dir: str) -> Dict[str, Any] | None:
    """
    run 디렉토리 내 feature_activation_by_input_type.json 을 로드한다.

    반환:
        dict 또는 파일이 없으면 None.
    """
    path = os.path.join(run_dir, "feature_activation_by_input_type.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_self_cross_delta(summary: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    feature_activation_by_input_type.json 의 내용을 기반으로
    - self Δf: steering label == concept 이름일 때 mean_f_after - mean_f_before 의 평균
    - cross Δf: steering label != concept 이름일 때 |Δf| 의 평균
    - self Δg / cross Δg: 위와 동일하되 g 기준
    을 계산한다.
    """
    diag_df = []
    off_df = []
    diag_dg = []
    off_dg = []

    for steer_label, concepts in summary.items():
        if not isinstance(concepts, dict):
            continue
        for concept_name, stats in concepts.items():
            try:
                mf_before = float(stats.get("mean_f_before", 0.0))
                mf_after = float(stats.get("mean_f_after", 0.0))
                mg_before = float(stats.get("mean_g_before", 0.0))
                mg_after = float(stats.get("mean_g_after", 0.0))
            except Exception:
                continue

            df = mf_after - mf_before
            dg = mg_after - mg_before

            if steer_label == concept_name:
                diag_df.append(df)
                diag_dg.append(dg)
            else:
                off_df.append(abs(df))
                off_dg.append(abs(dg))

    def _mean(vals: list[float]) -> float:
        return float(sum(vals) / len(vals)) if vals else 0.0

    return _mean(diag_df), _mean(off_df), _mean(diag_dg), _mean(off_dg)


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

    print("======== feature_activation_by_input_type 요약 (self vs cross Δf/Δg) ========")

    for run_dir in run_dirs:
        summary = load_feature_activation_summary(run_dir)
        if summary is None:
            continue

        ts = os.path.basename(run_dir)
        self_df, cross_df, self_dg, cross_dg = compute_self_cross_delta(summary)

        print(f"[RUN] {ts}")
        print(
            f"  Δf: self={self_df:.4f}, cross={cross_df:.4f}"
        )
        print(
            f"  Δg: self={self_dg:.4f}, cross={cross_dg:.4f}"
        )


if __name__ == "__main__":
    main()

