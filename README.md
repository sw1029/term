# Project H-SAE(가칭)

## 문제 정의(The Problem)
LLM의 추상적인 내부구조는 블랙박스와 같아서
특정 동작을 유도하거나 분석을 하려면
수많은 뉴런을 분석해야 하는 막대한 비용이 듭니다.

While the high-level abstraction of Large Language Models (LLMs) provides convenience for general users, it imposes a significant burden on engineers and researchers aiming to interpret and steer the models. This is because reverse engineering is essential to understanding the internal representations employed by these models.


## 솔루션(Solution)

본 프로젝트는 **H-SAE(가칭)** 를 통해 원하는 특징을 SAE 의 고정된 위치(Fixed position) 에 심는 방법을 제시합니다.
>Semi supervised : 대다수의 특징은 모델이 스스로 학습하게 두고 우리가 제어하고 싶은 일부 특징만 지도 학습을 통해 특정 위치에 고정합니다.

>Architecture : Transformer 레이어 출력에 SAE를 결합해 밀집된 정보를  해석 가능한 정보로 변환합니다.
 
- 즉각적인 분석 가능
-  탐색없는 조정

This project propose the **H-SAE** , 
which applies the 'Peature Freezing' techique to control these issus. By attaching a SAE to the
output layer if a specific layer in a standard Transformer model, we supervised learn only a very small subset of entire features as target concept vectors. This forces specific indices with in the SAE to represent those particiular features.
The remaining features restain the existing 
unsupervised approach, preserving the 
model's expressive power. 


## GNN 기반 설명(XAI) 확장

SAE.py 에서 제안된 아이디어를 실제 구현으로 확장하여,
SAE feature 간 상관관계를 1-layer GNN 으로 설명하는 브랜치를 추가했습니다.

- SAE 경로는 그대로 유지
  - encoder → ReLU(JumpReLU) → decoder → reconstruction loss
  - fixed concept vector 가 걸린 decoder row 는 gradient blocking 으로 보호
- GNN 브랜치 (설명 전용)
  - 노드: SAE feature index
  - 간선: decoder row 간 cosine similarity 기반 그래프 (top-k sparsity)
  - 연산: g = ReLU(A_norm @ f)
  - 학습 loss에는 포함하지 않으며, feature 중요도/상호작용 분석용으로만 사용


## 실행 방법 (Hydra 기반)

main.py 는 Hydra 를 사용해 config/config.yaml 를 로드합니다.

```bash
python main.py
```

기본 설정은 다음을 포함합니다:

- 모델: google/gemma-2-2b-it
- 대상 레이어: layer_idx=12
- SAE 차원: input_dim=2304, sae_dim=9216
- GNN:
  - decoder 기반 feature 그래프
  - top_k 이웃만 유지
- 실험:
  - positive/negative 프롬프트 (Python vs 일반 문장) 로 concept vector 추출
  - steered_feature_idx=0 에 concept 를 고정
  - steering_strength=15.0 으로 steering hook 주입


## 실험 산출물

실행 시 다음 산출물이 자동으로 생성됩니다.

- JSON 로그: `experimentLog/{timestamp}/L{layer}_S{strength}.json`
  - 모델/설정 정보
  - steering 전/후 텍스트 출력
  - steered feature 및 이웃 feature 의 f/g 활성값 변화
  - SAE 학습 loss 요약(mean L2/L1)
- Plot: `outputs/plots/{timestamp}/feature_{idx}_L{layer}_S{strength}.html`
  - steered feature + GNN 이웃들에 대한
    - f_before, f_after
    - g_before, g_after
  - grouped bar 로 시각화

이를 활용해 `가설/예상되는 문제점.md` 에 정리된 이슈들
(GT 벡터 품질, 특징 중복, 특징 소실 가능성 등)에 대해
steering 전후의 feature activation / graph 전파 결과를 비교 분석할 수 있습니다.


## 추가 실험: strength / layer sweep + STD plot

Hydra 설정(`config/experiment/default.yaml`)을 수정하면,
하나의 실행에서 여러 layer / strength 조합을 자동으로 실험할 수 있습니다.

- `layer_sweep`: `[12, 16, ...]` 처럼 여러 레이어를 지정
- `strength_sweep`: `[5.0, 10.0, 15.0]` 처럼 여러 steering 강도 지정

각 `(layer, strength)` 조합에 대해:

- 별도의 JSON 로그 (`experimentLog/{timestamp}/L{layer}_S{strength}.json`)
- steered feature 중심 bar plot (`outputs/plots/{timestamp}/feature_{idx}_L{layer}_S{strength}.html`)

이 생성됩니다.

또한, 각 layer 별로 SAE feature 전체에 대한 분산(표준편차) 분포를 보여주는
STD plot 이 추가로 생성됩니다.

- `outputs/plots/{timestamp}/feature_std_layer_{layer}.html`
- 모든 feature 의 std 값을 내림차순으로 정렬한 곡선을 그려,
  많은 feature 를 한 눈에 파악할 수 있게 디자인했습니다.
