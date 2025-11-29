import datetime
import json
import os
from typing import List

import optuna
from hydra import compose, initialize

from runner import run_experiment


def _build_overrides(trial: optuna.trial.Trial) -> List[str]:
    """
    code / harm 두 축의 steering 증대를 목표로 한
    Optuna trial용 Hydra override 문자열 목록을 생성한다.

    - 설정 고정:
        * experiment.use_multi_contrast=true
        * sae.loss_option=2 (concept + label guidance)
        * sae.loss_module=option2_loss
        * sae.guidance_method=contrastive
        * sae.guidance_coeff=1.0
        * sae.alpha_concept.struct=0.0 (구조성은 이번 탐색에서는 비목표)
        * sae.use_l0=true (L0 sparsity 항상 사용)

    - 탐색 대상 파라미터:
        * sae.alpha_concept.code: 0.0 ~ 0.08
        * sae.alpha_concept.harm: 0.0 ~ 0.08
        * sae.l0_coeff: 1e-4 ~ 3e-3 (log 스케일)
        * sae.role_sep_coeff: 0.0 ~ 0.01 (feature 분리 강도)
    """
    overrides: List[str] = []

    # multi-contrast 데이터 강제 사용
    overrides.append("experiment.use_multi_contrast=true")

    # SAE loss option / module: Option2에 고정
    overrides.append("sae.loss_option=2")
    overrides.append("sae.loss_module=option2_loss")

    # contrastive guidance 고정
    overrides.append("sae.guidance_method=contrastive")
    overrides.append("sae.guidance_coeff=1.0")

    # label guidance 강도 (code / harm 별도 탐색, struct는 0으로 고정)
    alpha_code = trial.suggest_float("sae.alpha_concept.code", 0.0, 0.08, step=0.005)
    alpha_harm = trial.suggest_float("sae.alpha_concept.harm", 0.0, 0.08, step=0.005)
    overrides.append(f"sae.alpha_concept.code={alpha_code}")
    overrides.append(f"sae.alpha_concept.harm={alpha_harm}")
    overrides.append("sae.alpha_concept.struct=0.0")

    # L0 sparsity 설정: 항상 사용 (use_l0=true), 계수만 탐색
    overrides.append("sae.use_l0=true")
    l0_coeff = trial.suggest_float("sae.l0_coeff", 1e-4, 3e-3, log=True)
    overrides.append(f"sae.l0_coeff={l0_coeff}")

    # feature 역할 분리(decorrelation) 강도 탐색
    role_sep_coeff = trial.suggest_float(
        "sae.role_sep_coeff",
        0.0,
        0.01,
        step=0.0005,
    )
    overrides.append(f"sae.role_sep_coeff={role_sep_coeff}")

    return overrides


def main() -> None:
    """
    code / harm 두 축의 steering 증대를 목표로 하는 Optuna 탐색 스크립트.

    - trial 수: 20
    - 목적:
        * code-steer 시 code 라벨 증가 (Δcode_codeSteer > 0)
        * harm-steer 시 harm 라벨 증가 (Δharm_harmSteer > 0)
      두 축이 모두 잘 올라가도록,
        score = min(Δcode_codeSteer, Δharm_harmSteer)
      를 최대화한다. 한쪽이라도 음수이면 score가 음수가 되어 페널티.

    각 trial:
        1) Hydra override로 cfg 구성
        2) runner.run_experiment 실행 → experimentLog/<timestamp> 생성
        3) eval_summary.json에서 by_steer_label의 delta_mean을 읽어 score 계산

    로그:
        - optuna_logs/code_harm_<timestamp>.log : run별 로그
        - optuna_logs/code_harm.log            : run들을 넘어서 누적 로그
        - optuna_best_code_harm.json          : best trial 갱신
    """
    os.makedirs("optuna_logs", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join("optuna_logs", f"code_harm_{ts}.log")
    global_log_path = os.path.join("optuna_logs", "code_harm.log")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        with open(global_log_path, "a", encoding="utf-8") as gf:
            gf.write(f"[{ts}] {msg}\n")

    study = optuna.create_study(
        direction="maximize",
        study_name=f"code_harm_{ts}",
    )

    with initialize(config_path="config", version_base=None):

        def objective(trial: optuna.trial.Trial) -> float:
            overrides = _build_overrides(trial)

            cfg = compose(config_name="config", overrides=overrides)

            logs_root = cfg.paths.logs_root
            os.makedirs(logs_root, exist_ok=True)
            before_dirs = set(os.listdir(logs_root))

            cli_command = "python main.py " + " ".join(overrides)
            log(f"[TRIAL {trial.number}] overrides: " + ", ".join(overrides))

            # 한 trial 전체 실행 (SAE 학습 + steering + OpenAI eval)
            run_experiment(cfg, cli_command=cli_command)

            # 새 run 디렉토리 찾기
            after_dirs = set(os.listdir(logs_root))
            new_dirs = sorted(after_dirs - before_dirs)
            if not new_dirs:
                raise RuntimeError("새 experimentLog 디렉토리를 찾을 수 없습니다.")
            run_id = new_dirs[-1]
            log_dir = os.path.join(logs_root, run_id)

            summary_path = os.path.join(log_dir, "eval_summary.json")
            if not os.path.exists(summary_path):
                raise RuntimeError(f"eval_summary.json 을 찾을 수 없습니다: {summary_path}")

            with open(summary_path, "r", encoding="utf-8") as sf:
                summary = json.load(sf)

            by_label = summary.get("by_steer_label", {})

            def _get_delta(lbl: str, key: str) -> float:
                if lbl not in by_label:
                    return 0.0
                delta = by_label[lbl].get("delta_mean", {})
                return float(delta.get(key, 0.0))

            # code-steer 시 code 변화, harm-steer 시 harm 변화
            d_code_code = _get_delta("code", "code")
            d_harm_harm = _get_delta("harm", "harm")

            # 두 축 모두 잘 올라가도록 최소값을 score로 사용
            score = min(d_code_code, d_harm_harm)

            log(
                f"[TRIAL {trial.number}] score={score:.4f} "
                f"(Δcode|code-steer={d_code_code:.3f}, Δharm|harm-steer={d_harm_harm:.3f}) "
                f"log_dir={log_dir}"
            )

            return score

        def _on_trial_end(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            # best trial 정보 실시간 업데이트
            best = study.best_trial
            best_info = {
                "best_trial_number": best.number,
                "best_value": best.value,
                "best_params": best.params,
                "study_name": study.study_name,
            }
            with open("optuna_best_code_harm.json", "w", encoding="utf-8") as f:
                json.dump(best_info, f, ensure_ascii=False, indent=2)
            log(
                f"[BEST] trial={best.number} value={best.value:.4f} "
                f"params={best.params}"
            )

        study.optimize(
            objective,
            n_trials=20,
            callbacks=[_on_trial_end],
        )

    log(f"[DONE] Study finished. Best value={study.best_value:.4f}")


if __name__ == "__main__":
    main()
