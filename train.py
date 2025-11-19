import torch.optim as optim
from tqdm import tqdm
from utils import *

def train_model(sae_model, llm_model, tokenizer, data_loader, config):
    device = config['device']
    sae_model.to(device)
    llm_model.to(device)

    optimizer = optim.Adam(sae_model.parameters(), lr=config['lr'])

    # Hyperparameters
    l1_coeff = config['l1_coeff']
    layer_idx = config['layer_idx']

    sae_model.train()

    print(f"Start Training Hybrid SAE on Layer {layer_idx}...")
    print(f"Fixed Features: {sae_model.fixed_v_cnt} / Trainable Features: {sae_model.sae_dim - sae_model.fixed_v_cnt}")

    step = 0
    total_loss_sum = 0

    for epoch in range(config['epochs']):
        for batch in tqdm(data_loader):
            text_batch = batch['text']
            if not text_batch: continue # 빈 데이터 스킵

            # 1. LLM에서 활성화 값(x) 추출
            try:
                x = get_batch_activations(llm_model, tokenizer, text_batch, layer_idx, device)
            except Exception as e:
                continue # 토큰화 오류 등 발생 시 스킵

            if x.shape[0] == 0: continue

            # 2. SAE Forward
            optimizer.zero_grad()
            x_reconst, f = sae_model(x)

            # 3. Loss 계산 (L2 Reconstruction + L1 Sparsity)
            l2_loss = F.mse_loss(x_reconst, x)
            l1_loss = l1_coeff * f.abs().sum(dim=1).mean()
            loss = l2_loss + l1_loss

            # 4. Backward
            loss.backward()

            # -----------------------------------------------------------
            # [핵심 로직] 컨셉 벡터로 지정된 Decoder 가중치의 Gradient를 0으로 만듦
            # -----------------------------------------------------------
            # sae_model.decoder.grad shape: (sae_dim, input_dim)
            # 0번부터 fixed_v_cnt번까지의 행(row)을 0으로 설정하여 업데이트 방지
            if sae_model.fixed_v_cnt > 0:
                sae_model.decoder.grad[:sae_model.fixed_v_cnt, :] = 0.0

            # 5. Optimizer Step
            optimizer.step()

            # 6. Decoder Unit Norm Constraint (SAE 표준 관행)
            # 학습 후 디코더 벡터의 크기를 1로 재조정
            with torch.no_grad():
                sae_model.decoder.data = F.normalize(sae_model.decoder.data, dim=1)

                # (선택) 혹시 모를 값 변형 방지를 위해 고정 벡터 다시 덮어쓰기 (엄격한 고정)
                # sae_model.decoder.data[:sae_model.fixed_v_cnt] = config['c_vectors_norm']

            total_loss_sum += loss.item()
            step += 1

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f} (L2: {l2_loss.item():.4f}, L1: {l1_loss.item():.4f})")

            if step >= config['max_steps']:
                break
        if step >= config['max_steps']:
            break

    print("Training Finished.")
    return sae_model