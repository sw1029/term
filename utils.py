import torch
import torch.nn as nn
import torch.nn.functional as F

'''
구현 TODO: 
    학습 결과 plotting 기능
    process checkpoint
    jump relu 구현
'''


class JumpReLU(nn.Module): # 적용 시 의문점: b의 초기값을 휴리스틱하게 들어가야 하나? 아니라면 또 하나의 파라미터로 학습 가능하게 두어야 하나?
    def __init__(self, init_b=0.0):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(init_b, dtype=torch.float32))

    def forward(self, x):
        # JumpReLU(x) = ReLU(x + b)
        return F.relu(x + self.b)
    








'''
이 아래로는 기존 구현들
'''


def generate_text(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Steering Hook 함수 정의 ---
def get_steering_hook(steering_vector, strength=10.0):
    def hook(module, input, output):
        # output이 튜플일 경우 (hidden_states, ...) 첫 번째 요소만 가져옴
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # [핵심] 0번 피처 벡터에 강도(Strength)를 곱해서 더해줌
        # Broadcasting: (Batch, Seq, Dim) + (Dim)
        hidden = hidden + (steering_vector * strength)

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return hook

def get_batch_activations(model, tokenizer, text_batch, layer_idx, device):
    inputs = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states[layer_idx]

    # [Batch, Seq_len, Dim] -> [Batch * Seq_len, Dim] (Flatten)
    activations = hidden_states.reshape(-1, hidden_states.shape[-1])

    return activations

def get_activation(text, layer, model, tokenizer):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        # Ensure inputs are on the same device as the model.
        # This assumes `model` has already been moved to the desired device (e.g., in the calling cell).
        inputs_on_model_device = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs_on_model_device, output_hidden_states = True)
        activation_tensor = outputs.hidden_states[layer].squeeze(0).mean(dim=0)
    return activation_tensor

def concept_vector(p_v, n_v):

    conect_v = p_v - n_v
    conect_v = conect_v / conect_v.norm() #정규화

    return conect_v