import torch.nn as nn
from transformers.transformer.sdpa import SDPA

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, mask=None):
        super().__init__()

        # Check divisibility
        if (d_model % num_heads != 0):
            raise ValueError ('d_model must be divisible by the number of heads for MHSA')

        # Store model parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.mask = mask
        self.d_k = d_model // num_heads

        # Build model layers
        self.W_Q = nn.Linear(self.d_model, self.d_model)
        self.W_K = nn.Linear(self.d_model, self.d_model)
        self.W_V = nn.Linear(self.d_model, self.d_model)
        self.linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, v_k, q=None):
        # Input x = [batch, seq_len, d_model]
        B, T, check = v_k.size()

        # Check size of third dimension
        if check != self.d_model:
            raise ValueError('Third dimension of input v_k into MHSA does not equal d_model.')

        # Projection linears
        V = self.W_V(v_k)
        K = self.W_K(v_k)

        # If Q and V, K are from different x
        if q is not None:
            
            B_q, T_q, check = q.size()
            # Check size of third dimension
            if check != self.d_model:
                raise ValueError('Third dimension of input q into MHSA does not equal d_model.')
            
            # Check batch sizes match
            if B != B_q:
                raise ValueError('Mismatched batch size for inputs V_K and Q for MHSA.')
            
            # Check sequence lengths match
            if T != T_q:
                raise ValueError('Mismatched sequence lengths for inputs V_K and Q for MHSA.')

            Q = self.W_Q(q)
        else:
            Q = self.W_Q(v_k)
        
        # Split the input into heads
        tensors = [Q, K, V]
        for tensr in tensors:
            tensr = tensr.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn = SDPA(Q, K, V, self.mask)

        # Concat
        attn = attn.transpose(1,2).contiguous().view(B, T, self.d_model)

        # Final linear layer
        out = self.linear(attn)

        return out