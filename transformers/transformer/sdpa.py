import torch
import torch.nn as nn

## Scaled Dot-Product Attention
def SDPA(Q:torch.Tensor, K: torch.Tensor, V:torch.Tensor) -> torch.Tensor:
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

    dim_k = K.shape[0]
    sm = nn.Softmax()
    out = torch.matmul(sm(torch.matmul(Q, K.T) / dim_k), V)
    return out