import torch
import torch.nn as nn
import math

## Scaled Dot-Product Attention
def SDPA(Q:torch.Tensor, K: torch.Tensor, V:torch.Tensor, mask=None) -> torch.Tensor:
    # Type checks
    tensors = [Q, K, V]
    for tensr in tensors:
        if not isinstance(tensr, torch.Tensor):
            raise TypeError('All inputs to the SDPA must be torch.Tensor objects')
        if (tensr.type() != 'torch.FloatTensor' and tensr.type() != 'torch.cuda.FloatTensor'):
            raise TypeError(f'All input Tensors to the SDPA must be of FloatTensor - found {tensr.type}')
    # Device check
    if Q.device != K.device or K.device != V.device:
        raise TypeError('All input tensors to the SDPA must be on the same device')

    dim_k = Q.size(-1)
    sm = nn.Softmax(dim=-1)
    out = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim_k)

    if mask is not None:
        out = out.masked_fill(mask == 0, float('-inf'))

    out = torch.matmul(sm(out), V)
    return out