# `experimentLog` 로그 구조 및 필드 설명

이 문서는 `experimentLog` 디렉터리 아래에 생성되는 JSON 로그와 보조 파일들의 구조와,
각 필드(항목)가 의미하는 바를 간략하게 정리한 것입니다.

## 1. 디렉터리 / 파일 개요

- `experimentLog/{timestamp}/`
  - 한 번의 실험 러닝(run)에 해당하는 기본 디렉터리입니다.
  - 안에는 layer·strength·label·샘플별 steering 로그와 평가 결과, 요약 파일이 들어갑니다.
- `experimentLog/layer_sweep/*`, `experimentLog/qwen_layer_sweep/*`,
  `experimentLog/llamaguard_layer_sweep/*`
  - 여러 layer/모델 조합을 한 번에 돌릴 때 생성되는 하위 실험 디렉터리입니다.
  - 안의 JSON 구조는 기본 `experimentLog/{timestamp}`와 동일합니다.
- 주요 파일 종류
  - `L{layer}_S{strength}_{label}_N{idx}.json`  
    개별 steering 로그(모델 출력 및 feature 통계).
  - `L{layer}_S{strength}_{label}_N{idx}_eval.json`  
    위 로그를 LLM judge로 평가한 결과.
  - `eval_summary.json`  
    한 run 디렉터리 안의 `_eval.json`들을 집계한 요약 통계.
  - `feature_activation_by_input_type.json`  
    입력 프롬프트 라벨별 · concept feature별 활성도 요약.
  - `feature_activation_by_input_type_eval.json`  
    위 요약에 대해 LLM judge를 한 번 더 적용한 결과(현재는 placeholder에 가까움).

아래에서는 각 파일 타입별로 포함되는 필드의 의미를 설명합니다.

---

## 2. 개별 steering 로그  
### 파일명: `L{layer}_S{strength}_{label}_N{idx}.json`

한 layer·strength·steering label·샘플 조합에 대해,
SAE 학습 설정, steering 전/후 출력, feature 통계, loss 요약을 담는 기본 로그입니다.

### 최상위 필드

- `model_name`  
  사용한 LLM 모델 이름 (예: `google/gemma-2-2b-it`, `Qwen/Qwen1.5-1.8B-Chat`).
- `layer_idx`  
  steering hook / SAE를 연결한 Transformer 레이어 인덱스(정수).
- `command`  
  이 로그를 생성할 때 사용한 `main.py` CLI 전체 명령 문자열 (Hydra override 포함).

### `sae` 블록 (SAE 설정)

- `input_dim`  
  SAE가 입력으로 받는 LLM hidden 차원 수.
- `sae_dim`  
  SAE latent feature 수(encoder 출력 차원).
- `fixed_v_cnt`  
  decoder에서 gradient를 막고 고정하는 concept feature 개수.
- `batch_size`  
  SAE 학습 시 사용한 배치 크기.
- `epochs`  
  (옵션) 에포크 수. `max_steps`가 우선인 경우도 있음.
- `max_steps`  
  SAE 학습의 최대 step 수.
- `l1_coeff`  
  L1 sparsity 정규화 계수.
- `l0_coeff`  
  L0 regularization(approx) 계수.
- `use_l0`  
  L0 계정을 실제로 사용하는지 여부.
- `concept_samples_per_label`  
  concept 벡터 계산에 사용한 라벨당 샘플 수.
- `loss_option`  
  사용한 loss 설계 옵션 번호 (1, 2, 3, 4 등).
- `loss_module`  
  실제 loss 구현이 들어있는 모듈 이름 (`option1_loss`, `option4_loss` 등).
- `concept_feature_indices`  
  `{ "code": 0, "harm": 1, "struct": 2 }` 와 같이  
  각 concept가 SAE feature 어느 인덱스에 박혀 있는지 나타냄.
- `alpha_concept`  
  `{concept 이름: alpha 값}` 형태.  
  concept 벡터를 얼마나 강하게 guidance에 사용할지에 대한 가중치.
- `positive_targets`  
  `{ "code": 1, "harm": 1, "struct": 1 }` 처럼  
  각 concept에 대해 positive로 밀고자 하는 방향(대부분 1/0).
- `guidance_method`  
  concept guidance 방식 이름 (예: `"contrastive"`).
- `guidance_margin`  
  contrastive loss에서 사용하는 margin 값.
- `guidance_coeff`  
  guidance term 전체에 곱해지는 계수.
- `guidance_mse_pos_value`, `guidance_mse_neg_value`  
  MSE 기반 guidance를 사용할 때 positive/negative 타깃 값.
- `jumprelu_bandwidth`  
  JumpReLU 활성화에서 사용하는 band 폭 설정.
- `jumprelu_init_threshold`  
  JumpReLU 임계값의 초기값.
- `role_sep_coeff`  
  role separation(특징 분리) 관련 정규화 계수.

### `gnn` 블록 (Feature GNN 설정)

- `use_gnn`  
  SAE 위에 feature GNN을 얹어 해석을 확장할지 여부.
- `top_k`  
  decoder 기반 feature 그래프에서 각 노드가 유지하는 이웃 수(top-k).

### `experiment` 블록 (실험 전역 설정)

- `device`  
  학습 및 분석에 사용한 디바이스 (`"cuda"`, `"cpu"` 등).
- `dataset_name`  
  Hugging Face 데이터셋 이름 (`"wikitext"` 등).
- `dataset_config`  
  데이터셋의 세부 설정/버전 (`"wikitext-2-raw-v1"` 등).
- `use_multi_contrast`  
  LLM judge로 라벨링된 multi-contrast 데이터셋을 사용하는지 여부.
- `num_steering_samples_per_label`  
  steering 분석에 사용할 라벨당 샘플 수.
- `top_k_for_plot`  
  steered feature 주변에서 plot/로그에 포함할 이웃 feature 수.
- `layer_sweep`  
  한 run에서 실험한 layer 인덱스 리스트.
- `strength_sweep`  
  한 run에서 실험한 steering strength 리스트.

### `steering` 블록 (이번 샘플에 대한 steering 설정)

- `label`  
  이 샘플에서 steering한 concept 이름 (`"code"`, `"harm"`, `"struct"` 등).
- `feature_idx`  
  steering에 사용한 SAE feature 인덱스 (대개 `concept_feature_indices[label]`).
- `strength`  
  steering 강도 (activation을 얼마나 밀어 올릴지).

### 프롬프트 / 출력

- `prompt`  
  steering 분석에 사용한 입력 프롬프트 전체 텍스트.
- `output`
  - `before`  
    steering 적용 **전** 모델의 출력 텍스트.
  - `after`  
    steering 적용 **후** 모델의 출력 텍스트.

### `feature_stats` 블록 (선택된 feature들에 대한 활성값 통계)

- `indices`  
  통계를 기록한 SAE feature 인덱스 리스트  
  (중심 steered feature와 GNN 이웃 feature 위주).
- `f_before`, `f_after`  
  steering 전/후 SAE latent `f`의 값 (각 인덱스별).
- `g_before`, `g_after`  
  steering 전/후 GNN 출력 `g`의 값 (각 인덱스별).  
  GNN을 사용하지 않으면 0.0으로 채워짐.
- `num_batches_for_stats`  
  feature 통계를 계산할 때 사용한 데이터 배치 수.

### `loss_summary` 블록 (SAE 학습 loss 요약)

- `train_mean_loss`  
  전체 학습 동안의 평균 total loss.
- `train_mean_l2`  
  reconstruction L2 loss의 평균.
- `train_mean_l1`  
  L1 sparsity term의 평균.
- `num_steps`  
  실제로 수행된 학습 step 수.
- `num_batches`  
  학습 과정에서 처리한 배치 수.

### (선택) `feature_std_summary` 블록

SAE feature 전체에 대한 표준편차 곡선(plot)을 함께 저장할 때 포함됩니다.

- `plot_path`  
  SAE feature std 곡선이 저장된 HTML 파일 경로.
- `mean_std`  
  std 벡터 자체의 표준편차(분산의 분포가 얼마나 넓은지).
- `max_std`  
  SAE feature들 중 가장 큰 표준편차 값.

---

## 3. per-log judge 평가 로그  
### 파일명: `L{layer}_S{strength}_{label}_N{idx}_eval.json`

각 steering 로그에 대해 LLM judge(또는 OpenAI)를 이용해  
출력 텍스트의 `code / harm / struct` 라벨 변화를 평가한 결과입니다.

### 최상위 필드

- `layer_idx`  
  평가 대상 로그의 layer 인덱스.
- `strength`  
  평가 대상 로그의 steering 강도.
- `sae`  
  원본 로그에서 가져온 SAE 설정 전체 (`build_experiment_log`의 `sae`와 동일 구조).
- `judge`
  - `model_name`  
    평가에 사용한 judge 모델 이름 (예: `"gpt-4o-mini"`).
  - `backend`  
    judge 실행 방식 (`"hf"` 또는 `"openai"`).
- `labels`
  - `before`  
    steering 전 응답에 대한 라벨 3개.
  - `after`  
    steering 후 응답에 대한 라벨 3개.
  - `delta`  
    `after - before` 값 (각 축별 변화량).

### `labels` 내부 구조

- 키: `"code"`, `"harm"`, `"struct"`
  - 값: 정수 0 또는 1
    - `code`: 응답이 **도움 되는 코드(특히 프로그래밍 코드)** 를 포함하는지 여부.
    - `harm`: 응답이 **유해/독성/공격적** 인지 여부.
    - `struct`: 응답이 **구조화(단계·리스트·섹션 등)** 가 잘 되어 있는지 여부.

예를 들어,

- `before["code"] = 0`, `after["code"] = 1`이면  
  steering 후 코드 관련성이 올라간 것으로 해석할 수 있습니다.

---

## 4. run 단위 평가 요약  
### 파일명: `eval_summary.json`

한 run 디렉터리 안의 모든 `_eval.json` 파일을 집계해,
전체/옵션별/steering label별 성능 변화를 요약한 파일입니다.

### 최상위 필드

- `num_logs`  
  집계에 사용된 steering 로그 개수.
- `before_mean`  
  전체 로그에 대해 라벨별 `before` 평균값.  
  `{ "code": float, "harm": float, "struct": float }`
- `after_mean`  
  전체 로그에 대해 라벨별 `after` 평균값.
- `delta_mean`  
  전체 로그에 대해 라벨별 `(after - before)` 평균값.

### `by_loss_option`

- 키: `loss_option` 번호 (예: `"1"`, `"4"`). 문자열 또는 정수로 직렬화됨.
- 값: 각 옵션별 요약
  - `num_logs`  
    해당 loss 옵션으로 학습된 steering 로그 개수.
  - `before_mean`, `after_mean`, `delta_mean`  
    해당 옵션에 속한 로그들만 대상으로 한 평균값 (구조는 위와 동일).

### `by_steer_label`

- 키: steering label 이름 (예: `"code"`, `"harm"`, `"struct"`).
- 값:
  - `num_logs`  
    해당 label을 steering한 로그 개수.
  - `before_mean`, `after_mean`, `delta_mean`  
    해당 label에 대한 평균 효과.

### (선택) `vs_option1_delta`

`loss_option=1` (baseline) 이 존재할 때만 포함됩니다.

- 키: 다른 `loss_option` 번호.
- 값:  
  각 옵션의 `delta_mean`과 Option1의 `delta_mean` 차이  
  `{ "code": float, "harm": float, "struct": float }`.  
  양수이면 Option1 대비 더 많이 올렸음을 의미합니다.

---

## 5. 입력 타입별 feature 활성 요약  
### 파일명: `feature_activation_by_input_type.json`

한 run 동안 수집한 steering 결과를 기반으로,  
“입력 프롬프트 라벨(steering label) × concept feature” 조합별 평균 활성 값을 정리한 파일입니다.

### 구조 개요

- 최상위 키: steering label 이름
  - 예: `"code"`, `"harm"`, `"struct"`
- 두 번째 계층 키: concept 이름
  - 예: `"code"`, `"harm"`, `"struct"` (SAE `concept_feature_indices`와 일치)
- 값: 통계 값 객체

### 통계 값 필드

각 `(steering_label, concept_name)` 쌍에 대해 다음 필드를 가집니다.

- `mean_f_before`  
  steering 전 SAE latent `f`의 평균 활성값.
- `mean_f_after`  
  steering 후 SAE latent `f`의 평균 활성값.
- `mean_g_before`  
  steering 전 GNN 출력 `g`의 평균 활성값 (GNN 미사용 시 0에 가까움).
- `mean_g_after`  
  steering 후 GNN 출력 `g`의 평균 활성값.
- `count`  
  이 평균을 계산할 때 사용한 샘플 수.

예) `["code" 입력일 때, "code" concept feature]의 평균 활성`  
대 `["harm" 입력일 때, "code" concept feature]의 평균 활성` 을 비교해  
feature의 selectivity를 확인할 수 있습니다.

---

## 6. 입력 타입별 요약에 대한 judge 평가  
### 파일명: `feature_activation_by_input_type_eval.json`

`feature_activation_by_input_type.json`에 대응하는 judge 평가 결과를 저장하기 위한 형식입니다.  
현재 예시에서는 대부분 placeholder 수준(값이 0 또는 null)으로 채워져 있습니다.

### 필드

- `layer_idx`  
  관련된 layer 인덱스 (현재 예시에서는 `null`인 경우가 많음).
- `strength`  
  관련된 steering 강도 (마찬가지로 `null`인 경우가 많음).
- `sae`  
  SAE 설정(필요 시 확장 가능, 예시에서는 빈 객체 `{}`).
- `judge`  
  - `model_name`  
    평가에 사용한 judge 모델 이름 (예: `"gpt-4o-mini"`).
  - `backend`  
    judge backend 종류 (`"openai"` 등).
- `labels`  
  구조는 개별 `_eval.json`의 `labels`와 동일:
  - `before` / `after` / `delta`  
    각 항목은 `{ "code": int, "harm": int, "struct": int }` 형식.

---

이 문서를 참고하면 `experimentLog` 아래의 JSON 로그를 직접 열어보지 않고도  
각 필드가 무엇을 의미하는지 빠르게 이해할 수 있습니다.***
