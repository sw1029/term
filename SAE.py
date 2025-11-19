import torch
import torch.nn as nn
import utils

'''
의문: feature간 상관관계를 attention 혹은 GNN으로 추가 설명가능성을 확보할 수는 없을까?
    이 경우 안쪽에 간단한 형식의 모델을 추가하는 것만으로 추가적인 시각화 가능성/해석가능성의 담보가 가능
    다만 이 경우 attention/GNN을 추가로 붙일 때 정말 간단하게만 붙여서 내부에 또 하나의 블랙박스가 생기지 않도록 주의가 필요할수도?
'''

class H_SAE(nn.Module):
    def __init__(self, input_dim, sae_dim, c_vectors):
        super().__init__()
        self.input_dim = input_dim
        self.sae_dim = sae_dim

        #encoder
        self.encoder = nn.Linear(input_dim , sae_dim)
        self.enc_bias = nn.Parameter(torch.zeros(sae_dim))
        #Decoder
        self.decoder = nn.Parameter(torch.randn(sae_dim, input_dim)) # output size = input
        self.dec_bias = nn.Parameter(torch.zeros(input_dim))

        self.fixed_v_cnt = len(c_vectors)
        #concept vectors
        with torch.no_grad():
            self.decoder.data[:self.fixed_v_cnt] = c_vectors.clone()

    def forward(self, x):
        x = self.encoder(x) - self.enc_bias
        f = torch.relu(x)  # <-- jump relu로 수정할것.
        #f = utils.JumpReLU(x, init_b=0.0) # b값을 고정한 형태의 초기적인 jump relu

        x = torch.matmul(f, self.decoder) + self.dec_bias 
        return x, f

