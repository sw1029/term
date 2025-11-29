import datetime
import json
import os
from typing import List

import optuna
from hydra import compose, initialize

from runner import run_experiment


def _build_overrides(trial: optuna.trial.Trial) -> List[str]:
    """
    sae.alpha_concept.{code,harm,struct} 세 개의 가중치를 Optuna로 탐색하는 스크립트용
    Hydra override 문자열 목록을 생성한다.

    고정 설정:
        - experiment.use_multi_contrast=true
        - sae.loss_option=2, sae.loss_module=option2_loss
        - sae.guidance_method=contrastive, sae.guidance_coeff=1.0
        - sae.use_l0=true, sae.l0_coeff=0.0009132649020605174
        - sae.role_sep_coeff=0.0035

    탐색 대상:
        - sae.alpha_concept.code   : 0.03 ~ 0.09 (step=0.005)
        - sae.alpha_concept.harm   : 0.0  ~ 0.04 (step=0.005)
        - sae.alpha_concept.struct : 0.0  ~ 0.04 (step=0.005)
    """
    overrides: List[str] = []

    # multi-contrast 데이터 강제 사용
    overrides.append("experiment.use_multi_contrast=true")

    # SAE loss option / module: Option2 고정
    overrides.append("sae.loss_option=2")
    overrides.append("sae.loss_module=option2_loss")

    # contrastive label guidance 고정
    overrides.append("sae.guidance_method=contrastive")
    overrides.append("sae.guidance_coeff=1.0")

    # L0 sparsity 및 role separation 계수는 현재 best 근처 값으로 고정
    overrides.append("sae.use_l0=true")
    overrides.append("sae.l0_coeff=0.0009132649020605174")
    overrides.append("sae.role_sep_coeff=0.0035")

    # alpha_concept.{code,harm,struct} 탐색
    alpha_code = trial.suggest_float(
        "sae.alpha_concept.code",
        0.03,
        0.09,
        step=0.005,
    )
    alpha_harm = trial.suggest_float(
        "sae.alpha_concept.harm",
        0.0,
        0.04,
        step=0.005,
    )
    alpha_struct = trial.suggest_float(
        "sae.alpha_concept.struct",
        0.0,
        0.04,
        step=0.005,
    )

    overrides.append(f"sae.alpha_concept.code={alpha_code}")
    overrides.append(f"sae.alpha_concept.harm={alpha_harm}")
    overrides.append(f"sae.alpha_concept.struct={alpha_struct}")

    return overrides


def main() -> None:
    """
    sae.alpha_concept.{code,harm,struct} 세 개의 가중치를 튜닝하여
    steering 결과(Δcode|code, Δharm|harm, Δstruct|struct)를 동시에 키우는
    Optuna 탐색 스크립트.

    목적 함수:
        - eval_summary.json 의 by_steer_label에서
          * d_code_code   = Δcode (code-steer)
          * d_harm_harm   = Δharm (harm-steer)
          * d_struct_struct = Δstruct (struct-steer)
        - 위 세 값의 최소값을 score로 사용:
              score = min(d_code_code, d_harm_harm, d_struct_struct)
          → 세 축 모두가 의미 있게 양수로 올라가도록 유도.

    trial 수:
        - n_trials = 30

    로그/결과물:
        - optuna_logs/alpha_concept_<timestamp>.log : run별 로그
        - optuna_logs/alpha_concept.log            : 누적 로그
        - optuna_best_alpha_concept.json           : best trial 정보
    """
    os.makedirs("optuna_logs", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join("optuna_logs", f"alpha_concept_{ts}.log")
    global_log_path = os.path.join("optuna_logs", "alpha_concept.log")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        with open(global_log_path, "a", encoding="utf-8") as gf:
            gf.write(f"[{ts}] {msg}\n")

    study = optuna.create_study(
        direction="maximize",
        study_name=f"alpha_concept_{ts}",
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

            # 여러 디렉토리가 새로 생긴 경우(동시 실행 등),
            # eval_summary.json 이 실제 존재하는 최신 디렉토리를 선택한다.
            run_id: str | None = None
            summary_path: str | None = None
            for candidate in reversed(new_dirs):  # 문자열 정렬 기준 최신부터 확인
                cand_log_dir = os.path.join(logs_root, candidate)
                cand_summary = os.path.join(cand_log_dir, "eval_summary.json")
                if os.path.exists(cand_summary):
                    run_id = candidate
                    summary_path = cand_summary
                    break

            if run_id is None or summary_path is None:
                raise RuntimeError(
                    f"새 experimentLog 디렉토리들에서 eval_summary.json 을 찾을 수 없습니다: {new_dirs}"
                )

            log_dir = os.path.join(logs_root, run_id)

            with open(summary_path, "r", encoding="utf-8") as sf:
                summary = json.load(sf)

            by_label = summary.get("by_steer_label", {})

            def _get_delta(lbl: str, key: str) -> float:
                if lbl not in by_label:
                    return 0.0
                delta = by_label[lbl].get("delta_mean", {})
                return float(delta.get(key, 0.0))

            # code-steer 시 code 변화, harm-steer 시 harm 변화, struct-steer 시 struct 변화
            d_code_code = _get_delta("code", "code")
            d_harm_harm = _get_delta("harm", "harm")
            d_struct_struct = _get_delta("struct", "struct")

            # 세 축 모두 잘 올라가도록 최소값을 score로 사용
            score = min(d_code_code, d_harm_harm, d_struct_struct)

            log(
                f"[TRIAL {trial.number}] score={score:.4f} "
                f"(Δcode|code-steer={d_code_code:.3f}, "
                f"Δharm|harm-steer={d_harm_harm:.3f}, "
                f"Δstruct|struct-steer={d_struct_struct:.3f}) "
                f"log_dir={log_dir}"
            )

            return score

        def _on_trial_end(
            study: optuna.Study, trial: optuna.trial.FrozenTrial
        ) -> None:
            # best trial 정보 실시간 업데이트
            best = study.best_trial
            best_info = {
                "best_trial_number": best.number,
                "best_value": best.value,
                "best_params": best.params,
                "study_name": study.study_name,
            }
            with open(
                "optuna_best_alpha_concept.json", "w", encoding="utf-8"
            ) as f:
                json.dump(best_info, f, ensure_ascii=False, indent=2)
            log(
                f"[BEST] trial={best.number} value={best.value:.4f} "
                f"params={best.params}"
            )

        study.optimize(
            objective,
            n_trials=30,
            callbacks=[_on_trial_end],
        )

    log(f"[DONE] Study finished. Best value={study.best_value:.4f}")


if __name__ == "__main__":
    main()
