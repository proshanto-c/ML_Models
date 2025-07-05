import torch
import torch.nn as nn
from transformers.transformer.encoder import EncoderBlock
from transformers.transformer.decoder import DecoderBlock
from transformers.transformer.mhsa import MultiHeadSelfAttention
import numpy as np

class Transformer(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

    def forward(self, x):
        PE = get_positional_encoding(self.seq_len, self.d_model)
        x = x + PE.unsqueeze(0)




def get_positional_encoding(seq_len, d_model, device):
    if (d_model%2 == 1):
        raise ValueError('d_model must be divisible by 2')
    PE = np.zeros((seq_len, d_model))
    for pos in range (seq_len):
        for i in range (d_model//2):
            PE[pos, 2*i] = np.sin(pos/(10000**(2*i / d_model)))
            PE[pos, 2*i + 1] = np.cos(pos/(10000**(2*i / d_model)))
    PE = torch.tensor(PE, device=device)
    return PE
