import torch.optim as optim
from tqdm import tqdm
from utils import *


def train_model(sae_model, llm_model, tokenizer, data_loader, config):
    """
    H-SAE를 학습시키는 함수.

    - SAE의 재구성 경로는 명시적으로 유지한다.
    - 디코더에 심어둔 컨셉 벡터는 학습 중에도 고정된다.
    - sae_model에 부착된 feature_gnn은 손실 계산에 사용되지 않으며,
      학습된 SAE feature를 설명(XAI)할 때에만 활용된다.
    """
    device = config["device"]
    sae_model.to(device)
    llm_model.to(device)

    optimizer = optim.Adam(sae_model.parameters(), lr=config["lr"])

    # 하이퍼파라미터 설정
    l1_coeff = config["l1_coeff"]
    layer_idx = config["layer_idx"]

    sae_model.train()

    print(f"Start Training Hybrid SAE on Layer {layer_idx}...")
    print(
        f"Fixed Features: {sae_model.fixed_v_cnt} / "
        f"Trainable Features: {sae_model.sae_dim - sae_model.fixed_v_cnt}"
    )

    step = 0
    total_loss_sum = 0.0
    total_l2_sum = 0.0
    total_l1_sum = 0.0
    total_batches = 0

    for epoch in range(config["epochs"]):
        for batch in tqdm(data_loader):
            text_batch = batch["text"]
            if not text_batch:
                continue  # 빈 데이터 스킵

            # 1. LLM에서 활성화 값(x) 추출
            try:
                x = get_batch_activations(llm_model, tokenizer, text_batch, layer_idx, device)
            except Exception:
                # 토큰화 오류 등 발생 시 스킵
                continue

            if x.shape[0] == 0:
                continue

            # 2. SAE Forward (설명용 GNN 브랜치 포함 가능)
            optimizer.zero_grad()

            out = sae_model(x)
            if isinstance(out, tuple):
                # 새로운 경로: (x_reconst, f, g) 또는 기존 형태 (x_reconst, f)
                x_reconst, f = out[0], out[1]
            else:
                # 단일 출력 모듈이 들어온 경우를 대비한 방어적 처리
                x_reconst, f = out, None

            # 3. Loss 계산 (L2 Reconstruction + L1 Sparsity)
            l2_loss = F.mse_loss(x_reconst, x)
            if f is not None:
                l1_loss = l1_coeff * f.abs().sum(dim=1).mean()
            else:
                l1_loss = 0.0 * l2_loss

            loss = l2_loss + l1_loss

            # 4. Backward
            loss.backward()

            # -----------------------------------------------------------
            # [핵심 로직] 컨셉 벡터로 지정된 Decoder 가중치의 Gradient를 0으로 만듦
            # -----------------------------------------------------------
            # sae_model.decoder.grad shape: (sae_dim, input_dim)
            # 0번부터 fixed_v_cnt번까지의 행(row)을 0으로 설정하여 업데이트 방지
            if sae_model.fixed_v_cnt > 0 and sae_model.decoder.grad is not None:
                sae_model.decoder.grad[: sae_model.fixed_v_cnt, :] = 0.0

            # 5. Optimizer Step
            optimizer.step()

            # 6. Decoder Unit Norm Constraint (SAE 표준 관행)
            # 학습 후 디코더 벡터의 크기를 1로 재조정
            with torch.no_grad():
                sae_model.decoder.data = F.normalize(sae_model.decoder.data, dim=1)
                # (선택) 혹시 모를 값 변형 방지를 위해 고정 벡터 다시 덮어쓰기 (엄격한 고정)
                # sae_model.decoder.data[:sae_model.fixed_v_cnt] = config['c_vectors_norm']

            total_loss_sum += float(loss.item())
            total_l2_sum += float(l2_loss.item())
            total_l1_sum += float(l1_loss.item())
            total_batches += 1
            step += 1

            if step % 100 == 0:
                print(
                    f"Step {step}, Loss: {loss.item():.4f} "
                    f"(L2: {l2_loss.item():.4f}, L1: {l1_loss.item():.4f})"
                )

            if step >= config["max_steps"]:
                break
        if step >= config["max_steps"]:
            break

    print("Training Finished.")

    train_stats = {
        "num_steps": step,
        "num_batches": total_batches,
        "mean_loss": total_loss_sum / max(total_batches, 1),
        "mean_l2": total_l2_sum / max(total_batches, 1),
        "mean_l1": total_l1_sum / max(total_batches, 1),
    }

    return sae_model, train_stats
