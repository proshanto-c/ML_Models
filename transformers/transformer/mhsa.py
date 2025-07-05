import torch
import torch.nn as nn
from transformers.transformer.sdpa import SDPA

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        if (d_model % num_heads != 0):
            raise ValueError ('d_model must be divisible by the number of heads for MHSA')

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(self.d_model, self.d_model)
        self.W_K = nn.Linear(self.d_model, self.d_model)
        self.W_V = nn.Linear(self.d_model, self.d_model)
        self.linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, v_k, q=None, mask=None):
        B, T, _ = v_k.size()

        # Projection linears
        V = self.W_V(v_k)
        K = self.W_K(v_k)
        if q is not None:
            Q = self.W_Q(q)
        else:
            Q = self.W_Q(v_k)
        
        # Split the input into heads
        tensors = [Q, K, V]
        for tensr in tensors:
            tensr = tensr.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn = SDPA(Q, K, V, mask)

        # Concat
        attn = attn.transpose(1,2).contiguous().view(B, T, self.d_model)

        # Final linear layer
        out = self.linear(attn)

        return out