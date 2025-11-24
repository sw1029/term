# Contrastive Learning 기반 Steering Feature 정렬 가설

본 문서는 현재 H-SAE + GNN-XAI 실험 파이프라인에서 관찰된 **실험 로그**를 바탕으로,
단순 재구성 기반 SAE를 넘어 **contrastive learning(대조 학습)** 을 도입해
특정 feature index를 명시적인 semantic concept(예: “Python 코드성”)으로 정렬시키려는
개선 방향을 정리합니다.

여기서 초점은 **contrastive 분석(사후 통계 분석)** 이 아니라,
학습 목표(loss)에 대비 정보(positive/negative)를 직접 포함시키는
**contrastive learning 설계**에 있습니다.

---

## 1. 현재 실험 로그로 본 H-SAE의 동작 특성

예시 로그:

```text
Step 1000, Loss: 5.9903 (L2: 2.0339, L1: 3.9564)
Training Finished.
[LOG] Experiment saved to: experimentLog/20251124-230018/L12_S15.0.json (layer=12, strength=15.0)
[PLOT] Feature change plot saved to: outputs/plots/20251124-230018/feature_0_L12_S15.0.html
```

해당 실행의 JSON (`experimentLog/20251124-230018/L12_S15.0.json`) 요약:

- 모델: `google/gemma-2-2b-it`
- 타깃 레이어: `layer_idx = 12`
- SAE:
  - `input_dim = 2304`
  - `sae_dim = 9216`
  - `fixed_v_cnt = 1` → 0번 feature의 decoder row에 concept vector 고정
- GNN:
  - `use_gnn = true`, `top_k = 10`
- Steering 설정:
  - `feature_idx = 0`
  - `strength = 15.0`
- 프롬프트:
  - `"How to sort a list?"`

### 1.1 LLM 출력(before/after)의 변화

`output` 필드:

- before:
  - 질문을 그대로 반복하고,
  - 파이썬 리스트 정렬 예제 코드(`my_list.sort(); print(my_list)`)를 제시.
- after:
  - `"**Solution:**"` 헤더를 붙이고,
  - 비슷한 파이썬 코드지만 인라인 주석 등으로 설명성이 약간 강화된 형태.

→ 프롬프트 자체가 Python 코드 답변을 자연스럽게 유도하는 형태이므로,
steering 전/후 모두 Python 코드 중심 응답이며,
steering 후에 형식/톤이 “튜토리얼스러운 방향”으로 약간 이동한 정도만 관찰됩니다.

### 1.2 SAE feature / GNN 출력 관측치

`feature_stats`:

- `indices`:
  - `[0, 3972, 7471, 2253, 4940, 3690, 8548, 8777, 5926, 3714, 8432]`
  - 0번 steered feature + GNN adjacency 기반 top-k 이웃 feature들.
- `f_before`, `f_after`:
  - 위 11개 feature에 대해 모두 `0.0`.
  - → 이 프롬프트에서 0번 feature 및 이웃 feature들이 ReLU 이후 전혀 활성되지 않았음을 의미.
- `g_before`, `g_after`:
  - 몇몇 이웃 feature에서 0이 아닌 값이 있으나,
  - `g_before = g_after` 로 값이 완전히 동일.
  - → `f_before = f_after` 이므로, GNN 전파 결과도 동일하게 나오는 상태.

결론적으로, 이 로그는 다음을 보여줍니다.

- 현재 설정에서:
  - steered feature(0번)와 그 주변 이웃 feature 집합은
    - 이 프롬프트에서 **실제로는 거의 사용되지 않고 있으며**,  
    - steering 전/후에도 값이 변하지 않습니다.
- 그럼에도 LLM 출력 텍스트는 약간 달라졌으므로,
  - steering 벡터가 LLM 내부 표현에는 영향을 주지만,
  - 우리가 “concept를 심어두었다고 가정한 index(0번)”와 그 이웃에는  
    **그 영향이 거의 투영되지 않고 있음**을 시사합니다.

### 1.3 feature 분산(표준편차) 분포

`feature_std_summary`:

- `mean_std ≈ 0.326`
- `max_std ≈ 34.608`

→ 대부분의 SAE feature는 표준편차가 0.3대인 작은 변동만 가지지만,  
극소수 feature는 30 이상까지 크게 변동하는 전형적인 “sparse tail” 분포를 보입니다.

이는 다음을 의미합니다.

- SAE는 소수의 feature를 자주 크게 활용하고,
- 나머지 feature 대부분은 거의 꺼져 있는 상태로 유지합니다.
- 하지만,
  - 그 “소수의 강하게 쓰이는 feature들”이 우리가 의도한 concept(예: Python)과
    얼마나 잘 align 되어 있는지는 **현재 loss 설계만으로는 보장되지 않습니다.**

---

## 2. 현 구조에서 SAE가 “재구성 중심”에 머무르는 이유

현재 H-SAE 학습에서 사용하는 손실:

```text
L_recon = ||x_hat - x||^2
L_sparse = λ * |f|_1
L_total  = L_recon + L_sparse
```

- `x`: LLM layer 12의 hidden (flatten 후)
- `f`: encoder(x) → ReLU 이후의 SAE feature
- `x_hat`: `f @ decoder + dec_bias`
- `λ`: 희소성을 유도하는 L1 계수

이때:

- **손실이 보는 것은 오직 두 가지입니다.**
  1. 입력 `x`를 얼마나 정확히 복원하는가? (L2 재구성)
  2. feature를 얼마나 적게(희소하게) 쓰는가? (L1)
- “이 feature가 Python concept인지, 영어 문장 concept인지”와 같은 semantic 정보는
  - decoder row의 방향 벡터 안에 간접적으로만 존재할 뿐,
  - encoder가 특정 index를 **반드시** 써야 할 이유가 없습니다.

컨셉 벡터를 0번 decoder row에 고정해도:

- encoder(x)가 0번 feature를 활성화하면,
  - 그 방향으로 x_hat에 기여하면서 L2를 줄이는 데 도움을 줄 수 있습니다.
- 하지만:
  - 이미 다른 feature 몇 개만으로도 x를 충분히 잘 복원할 수 있다면,
  - encoder 입장에서는 굳이 0번 feature를 자주 쓸 필요가 없습니다.
  - 오히려 0번을 안 쓰고, 다른 feature에 부담을 몰아주는 편이
    L1 관점에서 더 유리할 수도 있습니다.

따라서, **현재 loss 설계만 놓고 보면**:

- SAE는 “일반적인 토큰 분포를 재구성하는 것”에만 주로 관심을 갖고,
- 0번 feature가 Python concept을 표현해야 한다는 요구는
  - loss 함수 입장에서는 사실상 “부가적인 장식”에 가까운 상태입니다.
- 이 구조가 바로,
  - 로그에서 관찰된 “0번 feature 및 이웃이 아예 켜지지 않는” 현상을 낳는 근본적인 이유입니다.

---

## 3. Contrastive Learning의 목표: feature index에 semantic concept 심기

우리가 contrastive learning을 도입해 해결하고자 하는 핵심 목표는 다음과 같습니다.

> “재구성 + sparsity”만 최적화하는 SAE에  
> **pos/neg 대비 정보를 직접 학습 목표로 주어**,  
> 특정 feature index(또는 소수의 feature cluster)가
> 명시적인 semantic concept(예: Python vs non-Python)을 표현하도록 정렬(alignment)한다.

### 3.1 개념적 정식화

1. 데이터 분할
   - positive 집합 `D_pos`:
     - Python 코드, programming 관련 질문/답변 등 concept에 해당하는 텍스트들.
   - negative 집합 `D_neg`:
     - 일반 자연어 문장, 코드와 무관한 텍스트 등.

2. 타깃 feature index
   - 예: `k = 0` (현재 concept vector를 고정해둔 index)
   - 목표:
     - `E_{x∈D_pos}[f_k(x)]` 는 **크게**,
     - `E_{x∈D_neg}[f_k(x)]` 는 **작게** 만들고 싶다.

3. 직관적 요구:

```text
E_pos[f_k]  >>  E_neg[f_k]
```

이 조건을 학습 과정의 명시적인 제약으로 추가하는 것이
contrastive learning 설계의 중심입니다.

---

## 4. 제안하는 Contrastive Loss 설계

아래 손실들은 모두 기존 `L_recon + L_sparse`에 **추가로 더하는 형태**로 설계됩니다.

### 4.1 Feature-level Contrastive Supervision (단일 index용)

가장 단순한 형태는 “0번 feature 평균이 pos에서 크고 neg에서 작도록 만드는” 손실입니다.

1. 배치 구성을 pos/neg로 나눈다고 가정:

```text
batch_pos  = {x_i | 레이블 = positive}
batch_neg  = {x_j | 레이블 = negative}
```

2. 배치 내 평균 feature:

```text
μ_pos = E_batch_pos[f_k(x)]
μ_neg = E_batch_neg[f_k(x)]
```

3. contrastive 손실 예시:

```text
L_concept = - (μ_pos - μ_neg)
L_total   = L_recon + λ L_sparse + α L_concept
```

- `μ_pos - μ_neg` 가 커질수록,
  - 즉, pos에서 f_k가 커지고 neg에서 f_k가 작아질수록
  - `L_concept`가 작아지므로,
- α > 0 이면 학습이 이 방향으로 feature를 밀어주게 됩니다.

이 손실의 기대 효과:

- 현재 로그처럼 0번 feature가 거의 0에 머무르는 상황에서:
  - `D_pos`에 대해 0번이 **강하게 활성**되도록 유도하고,
  - `D_neg`에 대해서는 여전히 거의 0에 가깝게 유지하도록 유도합니다.
- 결과적으로:
  - 재구성에만 의존하던 encoder가
  - “Python vs non-Python 대비”라는 **semantic 신호**를
    0번 feature에 억지로라도 실어 보내게 됩니다.

### 4.2 Representation-level Contrastive Loss (InfoNCE 스타일)

단일 index가 아니라,
문장/토큰의 전체 SAE feature 표현을 대상으로 contrastive learning을 적용할 수도 있습니다.

예시 아이디어:

1. 각 샘플 x에 대해 SAE feature를 문장 단위로 pooling:

```text
z(x) = Pool(f(x))   # 예: 토큰 평균, max-pooling 등
```

2. positive pair / negative pair 정의:
   - 예: 동일한 concept(“Python”)에 속하는 두 문장은 positive pair,
   - concept가 다른 문장쌍은 negative pair.

3. InfoNCE/NT-Xent 스타일 loss:

```text
L_contrastive = - log exp(sim(z_i, z_i⁺) / τ)
                --------------------------------
                sum_j exp(sim(z_i, z_j) / τ)
```

- 여기서 `sim`은 cosine similarity, τ는 온도(temperature).
- 이 손실은:
  - 같은 concept의 z 표현들은 가깝게,
  - 다른 concept의 z 표현들은 멀어지도록 유도합니다.

SAE 관점에서:

- z가 f의 함수이므로,
  - encoder와 decoder(특히 concept 관련 feature들)에 gradient가 흘러 들어가며,
  - 특정 feature subset이 concept 구분에 중요한 방향으로 학습됩니다.
- 이때, 우리가 0번 feature 또는 GNN을 통해 선택한 이웃 feature들에
  - 가중치를 더 주는 형태로 `z`를 정의하면,
  - 해당 feature들이 semantic axis 역할을 하도록 더욱 집중시킬 수 있습니다.

### 4.3 GNN 기반 Contrast Regularization (선택 사항)

GNN-XAI 브랜치를 활용하면,
concept index와 그 GNN 이웃들을 **같은 cluster**로 묶는 정규화도 가능합니다.

예시:

1. steering index `k`와 그 이웃 집합 `N(k)`을 사용:

```text
cluster_k = { k } ∪ N(k)
```

2. pos/neg 배치에 대해 cluster 평균 활성값 계산:

```text
μ_pos_cluster = E_pos[ mean_{i∈cluster_k} f_i(x) ]
μ_neg_cluster = E_neg[ mean_{i∈cluster_k} f_i(x) ]
```

3. loss:

```text
L_cluster = - (μ_pos_cluster - μ_neg_cluster)
```

- 이 손실을 `L_concept`와 함께 사용하면,
  - 0번 feature 뿐 아니라,
  - GNN 상에서 그와 강하게 연결된 feature들까지
  - “Python vs non-Python 대비”에 관여하는 cluster로 정렬시킬 수 있습니다.

---

## 5. 기존 재구성 loss와의 통합 및 기대되는 로그 변화

최종 loss 예시:

```text
L_total = L_recon + λ L_sparse
          + α L_concept
          + β L_contrastive
          + γ L_cluster
```

- α, β, γ는 contrastive 항들의 기여를 조절하는 계수.
- 실험 과정에서 다음과 같은 trade-off 를 튜닝하게 됩니다.

### 5.1 기대되는 변화 (현재 로그와 대비)

현 로그에서:

- `f_before`, `f_after` (선택된 index들) = 모두 0
- steering 전/후 `g_before`, `g_after`도 동일
- STD:
  - mean_std는 작고, max_std는 아주 큰 tail

contrastive learning을 적용한 후 기대되는 변화:

1. 특정 concept index(예: 0번)의 활성 패턴
   - `D_pos`에 대한 `E[f_0]`는 0 근처가 아니라,
     **명확하게 양의 값으로 올라가야** 합니다.
   - `D_neg`에 대한 `E[f_0]`는 여전히 0에 가깝게 유지되도록 조절.
   - steering 전/후 `Δf_0`도 지금처럼 0이 아니라,
     concept과 관련된 방향으로 의미 있게 변해야 합니다.

2. GNN 이웃 feature들의 활성
   - `cluster_k` 내 feature들의 `Δf`, `Δg`가
     - concept에 따라 일관된 방향으로 변하는 패턴이 나타나야 합니다.

3. STD 분포
   - 전체적으로 sparse 구조는 유지하되,
   - concept index 및 그 cluster의 표준편차가
     - “concept과 무관한 tail feature들”과는 다른 패턴을 보이게 됩니다.
   - 예: Python concept와 관련된 layer에서는
     - 해당 feature cluster의 std가 pos/neg 대비 차이를 명확히 드러냄.

4. steering 전/후 텍스트 변화
   - 현재처럼 “이미 Python 코드를 잘 내던 질문에서 약간 포맷만 바뀌는” 정도를 넘어,
   - concept index에 해당하는 steering을 걸었을 때:
     - Python 코드 관련 질문에서는 더 강한 코드/설명 성향,
     - 반대로 비관련 질문에서 steering을 줄이면 영향이 제한적이어야 합니다.

---

## 6. 요약: 왜 contrastive **learning**이 필요한가

현재 구조에서:

- H-SAE는 재구성(L2) + sparsity(L1)만 최적화하며,
  - “0번 feature가 Python concept여야 한다”는 요구는 loss에 거의 반영되지 않는다.
- 그 결과,
  - concept vector를 decoder 0번에 고정해도,
  - encoder는 0번 feature를 거의 쓰지 않아도 재구성이 잘 되면 그대로 두게 된다.
- 실제 로그에서,
  - steered feature 및 GNN 이웃이 전혀 활성되지 않는 상태로 steering 실험이 수행되고 있음을 확인했다.

이 상황을 바꾸려면:

- 단순 분석(사후 probing)만으로는 부족하고,
- 학습 단계에서 pos/neg 대비 정보를 **loss에 직접 주입**해야 한다.

즉, contrastive learning의 역할은:

1. “일반적인 토큰 분포 재구성”이라는 기본 역할은 유지하면서,
2. 특정 feature index 또는 feature cluster가
   - 우리가 정의한 concept (예: Python vs non-Python)에 대해
   - 일관된 대비 패턴을 보이도록 **명시적으로 정렬(alignment)시키는 것**입니다.

향후 실험에서는:

- 위에서 제안한 `L_concept`, `L_contrastive`, `L_cluster` 형태의 손실을
  점진적으로 추가/튜닝하면서,
- 현재와 같은 로그(예: `L12_S15.0.json`) 구조를 그대로 찍되,
  - `f_before`, `f_after`, `feature_std_summary` 등 지표가
    contrastive learning 도입 이후 어떻게 달라지는지 비교함으로써
  - 실제 개선 여부를 검증할 수 있을 것입니다.

