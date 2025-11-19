'''
실제 호출부
왠만한건 다 여기서 호출할것

huggingface 로그인이 안된 경우라면 터미널에서 huggingface cli에 토큰 등록 필요
'''

# 별도로 정의한 기능들
from utils import *
from train import *
from SAE import *

# 기타 필요한 라이브러리(기존의 의존성 그대로 가져옴.)
import numpy
import os
import torch
from tqdm import tqdm
import plotly.express as px
import webbrowser
import http.server
import socketserver
import threading
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader





#모델 호출
model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval() # 학습시키지 x

# 데이터로더 준비
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)







'''
이 하단은 실제 작동 실행 검증을 위한 부분이며, 이후 조금 더 가시성 + 가용성 감안해서 기능 분리 예정입니다.
'''








pos = ["def my_func(x): return x", "import numpy as np", "print('hello')"]
nag = ["The weather is nice.", "I like apples.", "History of Rome"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE) # Move model to device

# get_activation 함수는 사용자 정의 함수 사용 (layer index 주의)
LAYER_IDX = 12 # 예: 12번 레이어
# get_activation now directly returns a (hidden_size,) tensor, so no [0] is needed.
p_acts = torch.stack([get_activation(t, LAYER_IDX, model, tokenizer) for t in pos]).mean(dim=0)
n_acts = torch.stack([get_activation(t, LAYER_IDX, model, tokenizer) for t in nag]).mean(dim=0)
c_vector = concept_vector(p_acts, n_acts)
c_vectors = c_vector.unsqueeze(0)

# 2. 설정값 정의
config = {
    'batch_size': 4,         # VRAM에 따라 조절
    'epochs': 1,
    'lr': 3e-4,
    'l1_coeff': 0.005,       # 희소성 조절 계수
    'layer_idx': LAYER_IDX,
    'max_steps': 1000,       # 테스트용으로 1000번
    'device': DEVICE,
    'c_vectors_norm': F.normalize(c_vectors.to(DEVICE), dim=1) # 재할당용, use the defined DEVICE
}

# 3. 모델 초기화
d_model = 2304 # Gemma-2B Hidden Size
d_sae = d_model * 4 # 확장 계수 (보통 16~32를 쓰지만 테스트용으론 작게)

sae = H_SAE(d_model, d_sae, c_vectors.to(config['device']))

# 4. 학습 시작
trained_sae = train_model(sae, model, tokenizer, data_loader, config)


# 1. 테스트할 프롬프트 (일반적인 질문)
test_prompt = "How to sort a list?"

# 2. Steering에 사용할 벡터 가져오기 (0번 피처 = Python Concept)
# 학습된 SAE의 디코더 0번 컬럼이 우리가 고정한 벡터입니다.
steering_vector = trained_sae.decoder.data[0].to(model.device)

print(f"=== Test Prompt: '{test_prompt}' ===\n")

# [Case A] 기존 모델 (No Steering)
print("--- [A] Original Model Output ---")
print(generate_text(model, tokenizer, test_prompt))

# [Case B] Steering 모델 (With Python Feature)
# Hook 등록
strength = 15.0 # 강도 조절 (너무 작으면 효과 X, 너무 크면 깨짐. 5~20 사이 추천)
layer_to_hook = config['layer_idx']

# Gemma 구조에 맞춰 Hook 위치 지정 (model.layers[i].mlp)
hook_handle = model.model.layers[layer_to_hook].mlp.register_forward_hook(
    get_steering_hook(steering_vector, strength=strength)
)

print(f"\n--- [B] Steered Model Output (Strength={strength}) ---")
try:
    print(generate_text(model, tokenizer, test_prompt))
finally:
    # [중요] 반드시 Hook을 제거해야 다음 실행에 영향이 없음
    hook_handle.remove()
    print("\n(Hook removed)")
