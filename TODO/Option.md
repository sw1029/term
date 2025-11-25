## Option1 : concept 벡터 고정

```python
decoer_w[0] = concept_v
no_grad
loss = L_재구성 + λ·L_sparse
```


## Option2 : 1 + label guidance

```python
decoer_w[0] = concept_v # fixed

loss = L_재구성 + λ·L_sparse + L_gudiance (label)
```

## Option3 : label guidance

```python
loss = L_재구성 + λ·L_sparse + L_gudiance (label)
```

### concept_v 문제.

최적이 아닐 수 있음.
재구성 시 고정된 v 의 사용이 될것이라는 보장이 없음.

### label 만 사용시 문제.
학습하면서 가진 feature 벡터가 원하는 개념이 아닐 수 있음.
---> 실험하면서 제대로 가졌는 지 확인해야함.

### 둘다 사용시 문제
재구성 손실이 매우 높을 듯
---> loss 계산에서 파리미터 조절이 필요.

----
1.고정된 기저 벡터를 빼고 학습하면서 concept_vec 와 유사하게 학습되길 기대하거나.  
2.고정된 벡터를 사용하면서 적절한 하이퍼파라미터가 탐색되거나 GT를 매우 잘뽑기를 기대해야함. 
