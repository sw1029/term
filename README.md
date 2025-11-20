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
