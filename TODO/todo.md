todo markdown 문서

시작안함/미완/완료 단계를 명확히 구분하여 기술.


https://www.canva.com/design/DAG5wwVrofA/p791vl7h_VvAoIZHg13L_w/edit?utm_content=DAG5wwVrofA&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
## 지도 손실의 추가
> $$L_{total} = L_{recon} + \lambda L_{L0} + \beta L_{guidance}$$

```python
def compute_guidance_loss(f, target_idx, labels, method="mse", margin=1.0):
    """
    L_guidance 계산 함수
    
    Args:
        f: SAE의 인코더 출력 (Batch, sae_dim)
        target_idx: 강제하고 싶은 뉴런의 인덱스 (예: 0)
        labels: 해당 배치의 라벨 (Batch,) -> 1.0 (특징 있음) or 0.0 (특징 없음)
        method: 'mse' 또는 'contrastive'
    """
    # f shape: (Batch, Features) -> target_activations shape: (Batch,)
    target_activations = f[:, target_idx]
    
    if method == "mse":
        # 옵션 A: MSE 
        return F.mse_loss(target_activations, labels)
        
    elif method == "contrastive":
        # 옵션 B: Contrastive Margin
        pos_mask = (labels == 1.0)
        neg_mask = (labels == 0.0)
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=f.device) # 비교 대상 없으면 0 반환
            
        pos_acts = target_activations[pos_mask]
        neg_acts = target_activations[neg_mask]
        
        # (평균 차이) 방식: Margin - (Pos평균 - Neg평균) > 0 이면 Loss 발생
        loss = F.relu(margin - (pos_acts.mean() - neg_acts.mean()))
        return loss
    
    return torch.tensor(0.0, device=f.device)
```
간소버전
```python
def compute_guidance_loss(f, target_idx, labels, margin=1.0):
    target_activations = f[:, target_idx]
    
    pos_mask = (labels == 1.0)
    neg_mask = (labels == 0.0)
    
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return torch.tensor(0.0, device=f.device)
        
    pos_acts = target_activations[pos_mask]
    neg_acts = target_activations[neg_mask]

    loss = F.relu(margin - (pos_acts.mean() - neg_acts.mean()))
    return loss
```

## Jump Relu
JumpReLU: x * H(x - θ) // H 계단함수   
band width : 정확한 근사, 학습 불안전 <-----> 부드러운 학습, 불정확한 근사   
init_threshold : 일반 Relu 와 유사 <-----> 대부분의 뉴런이 비활성화   
```python
class JumpReLUFunction(torch.autograd.Function):
    def forward(ctx, x, threshold, bandwidth):
        """
        x: 입력 활성화 값 (Pre-activations)
        threshold: 임계값 (Theta)
        bandwidth: 커널의 너비 (Epsilon), 기울기 추정용
        """
        # 역전파를 위해 저장
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        
        # Forward: x가 threshold보다 크면 x, 아니면 0 
        # JumpReLU(z) = z * H(z - theta)
        # 각 feature 마다 다른 임계값을 가질 수 있게됨.
        mask = (x > threshold).float()
        return x * mask
        
    def backward(ctx, grad_output):
        # 계단함수는 미분이 불가 -> 이를 미분 가능하게 하기 위해
        # 2에서 근사를 진행함.
        # bandwidth 내의 값만 전달.
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        
        # 1. 입력(x)에 대한 Gradient 
        # 활성화된 부분(x > threshold)은 그대로 흘려보냄 (Identity)
        grad_x = (x > threshold).float() * grad_output
        
        # 2. 임계값(threshold)에 대한 Gradient (핵심 STE)
        # 미분 불가능한 지점을 '직사각형 커널'로 근사하여 계산
        # grad_theta = -(theta / bandwidth) * rectangle((x - theta) / bandwidth) * grad_output
        
        # 커널 입력값 계산
        kernel_arg = (x - threshold) / bandwidth
        
        # 직사각형 커널 (Rectangle Kernel): -0.5 < z < 0.5 범위에서 1 [cite: 914]
        rectangle_kernel = ((kernel_arg > -0.5) & (kernel_arg < 0.5)).float()
        
 
        grad_threshold = -(threshold / bandwidth) * rectangle_kernel * grad_output
        
        # 배치(Batch) 차원에 대해 합산 (Threshold는 보통 피처 차원만 가지므로)
        if grad_threshold.ndim > threshold.ndim:
            grad_threshold = grad_threshold.sum(dim=0) # 배치 차원 합산
            
        return grad_x, grad_threshold, None

class JumpReLU(nn.Module):
    def __init__(self, feature_dim, bandwidth=0.001, init_threshold=0.001):
        super().__init__()
        self.feature_dim = feature_dim
        self.bandwidth = bandwidth
        self.log_threshold = nn.Parameter(torch.log(torch.full((feature_dim,), init_threshold)))

    def forward(self, x):
        threshold = self.get_threshold()
        x = torch.relu(x)
        
        return JumpReLUFunction.apply(x, threshold, self.bandwidth)
    
    def get_threshold(self):
        return torch.exp(self.log_threshold)

```

## Heavy side function
sparsity 제어를 위해 threshold만 조정
JumpReLU의 독립성
→ 재구성 품질에만 집중
→ z 값을 최적화

Heaviside의 독립성  
→ sparsity 제어에만 집중
→ threshold 값을 최적화

---> L0 loss

```python
class HeavisideStepFunction(torch.autograd.Function):
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return (x > threshold).float()
    
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        
        # x는 gradient 차단
        grad_x = torch.zeros_like(x)
        
        # threshold만 학습
        kernel_arg = (x - threshold) / bandwidth
        rectangle_kernel = ((kernel_arg > -0.5) & (kernel_arg < 0.5)).float()
        grad_threshold = -(1.0 / bandwidth) * rectangle_kernel * grad_output
        
        if grad_threshold.ndim > threshold.ndim:
            grad_threshold = grad_threshold.sum(dim=0)
            
        return grad_x, grad_threshold, None
```

## flow

```python
        ..........

        z = self.encoder(x)
        
        # 1. JumpReLU로 활성화 (실제 계산용)
        h = self.jumprelu(z)
        
        # 2. Heaviside로 L0 카운팅 (sparsity 측정용)
        threshold = self.jumprelu.get_threshold()
        bandwidth = self.jumprelu.bandwidth
        l0_count = HeavisideStepFunction.apply(z, threshold, bandwidth).sum(dim=-1)
        
        # Decode
        x_recon = self.decoder(h)

        ..........
```

