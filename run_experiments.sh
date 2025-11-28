#!/usr/bin/env bash

# Hybrid-SAE 실험 일괄 실행 스크립트.
# - Hydra config 기반으로 여러 옵션/하이퍼파라미터 조합을 순차 실행
# - 각 실험은 실패(OOM 포함)해도 다음 실험으로 계속 진행
# - 전체 스크립트는 nohup으로 한 번만 감싸서 실행하는 것을 권장:
#     nohup bash run_experiments.sh > run_experiments.out 2>&1 &

set -u  # 정의되지 않은 변수 사용 방지 (에러 발생 시점만 제어)

# conda 환경 'term'에서 실행되도록 기본 Python을 지정
# 필요시 실행 시점에 PYTHON_BIN을 override 할 수 있음.
PYTHON_BIN=${PYTHON_BIN:-conda run -n term python}
LOG_ROOT="nohup_logs"
mkdir -p "${LOG_ROOT}"

# 공통 옵션: multi_contrast 라벨 사용
COMMON_ARGS="experiment.use_multi_contrast=true"

# 실험 조합 정의
# 각 항목은 Hydra override 문자열 한 줄
EXPERIMENTS=(
  # 1. Option1: concept only, L0 OFF
  "sae.loss_option=1 sae.loss_module=option1_loss ${COMMON_ARGS} sae.use_l0=false"

  # 2. Option1: concept only, L0 ON (약한 sparsity)
  "sae.loss_option=1 sae.loss_module=option1_loss ${COMMON_ARGS} sae.use_l0=true sae.l0_coeff=0.001"

  # 3. Option2: concept+label, α=0.01, contrastive, L0 OFF
  "sae.loss_option=2 sae.loss_module=option2_loss ${COMMON_ARGS} \
   sae.alpha_concept.code=0.01 sae.alpha_concept.harm=0.01 sae.alpha_concept.struct=0.01 \
   sae.guidance_method=contrastive sae.use_l0=false"

  # 4. Option2: concept+label, α=0.01, contrastive, L0 ON
  "sae.loss_option=2 sae.loss_module=option2_loss ${COMMON_ARGS} \
   sae.alpha_concept.code=0.01 sae.alpha_concept.harm=0.01 sae.alpha_concept.struct=0.01 \
   sae.guidance_method=contrastive sae.use_l0=true sae.l0_coeff=0.001"

  # 5. Option2: concept+label, α=0.01, margin-based guidance
  "sae.loss_option=2 sae.loss_module=option2_loss ${COMMON_ARGS} \
   sae.alpha_concept.code=0.01 sae.alpha_concept.harm=0.01 sae.alpha_concept.struct=0.01 \
   sae.guidance_method=margin sae.guidance_margin=1.0 sae.guidance_coeff=1.0 \
   sae.use_l0=false"

  # 6. Option2: concept+label, α=0.01, MSE-based guidance
  "sae.loss_option=2 sae.loss_module=option2_loss ${COMMON_ARGS} \
   sae.alpha_concept.code=0.01 sae.alpha_concept.harm=0.01 sae.alpha_concept.struct=0.01 \
   sae.guidance_method=mse sae.guidance_mse_pos_value=1.0 sae.guidance_mse_neg_value=0.0 \
   sae.guidance_coeff=1.0 sae.use_l0=false"

  # 7. Option2: concept+label, α=0.05, contrastive, L0 OFF
  "sae.loss_option=2 sae.loss_module=option2_loss ${COMMON_ARGS} \
   sae.alpha_concept.code=0.05 sae.alpha_concept.harm=0.05 sae.alpha_concept.struct=0.05 \
   sae.guidance_method=contrastive sae.use_l0=false"

  # 8. Option3: label-only, contrastive, L0 OFF
  "sae.loss_option=3 sae.loss_module=option3_loss ${COMMON_ARGS} \
   sae.guidance_method=contrastive sae.use_l0=false"

  # 9. Option3: label-only, contrastive, L0 ON
  "sae.loss_option=3 sae.loss_module=option3_loss ${COMMON_ARGS} \
   sae.guidance_method=contrastive sae.use_l0=true sae.l0_coeff=0.001"
)

SUMMARY_LOG="${LOG_ROOT}/summary_$(date +%Y%m%d-%H%M%S).log"
echo "[$(date)] Starting experiment batch." | tee -a "${SUMMARY_LOG}"

idx=0
for EXP in "${EXPERIMENTS[@]}"; do
  idx=$((idx + 1))
  TS=$(date +%Y%m%d-%H%M%S)
  LOG_FILE="${LOG_ROOT}/exp_${idx}_${TS}.log"

  echo "------------------------------------------------------------" | tee -a "${SUMMARY_LOG}" | tee -a "${LOG_FILE}"
  echo "[$(date)] Experiment #${idx}" | tee -a "${SUMMARY_LOG}" | tee -a "${LOG_FILE}"
  echo "Command overrides: ${EXP}" | tee -a "${SUMMARY_LOG}" | tee -a "${LOG_FILE}"

  # 실제 실행 (OOM 포함 어떤 에러가 나도 exit code로만 판단)
  ${PYTHON_BIN} main.py ${EXP} >> "${LOG_FILE}" 2>&1
  STATUS=$?

  if [ ${STATUS} -ne 0 ]; then
    echo "[$(date)] Experiment #${idx} FAILED (exit code=${STATUS}). Continuing to next." | tee -a "${SUMMARY_LOG}" | tee -a "${LOG_FILE}"
  else
    echo "[$(date)] Experiment #${idx} COMPLETED successfully." | tee -a "${SUMMARY_LOG}" | tee -a "${LOG_FILE}"
  fi
done

echo "[$(date)] All experiments finished." | tee -a "${SUMMARY_LOG}"
