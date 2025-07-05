import torch
import torch.nn as nn
from transformers.transformer.encoder import EncoderBlock
from transformers.transformer.decoder import DecoderBlock
from transformers.transformer.mhsa import MultiHeadSelfAttention
import numpy as np

class Transformer(nn.Module):
    def __init__(self, seq_len, d_model, n_enc_blocks, n_dec_blocks, n_out):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_out = n_out
        self.encoder_blocks = nn.ModuleList([EncoderBlock() for _ in range(n_enc_blocks)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock() for _ in range(n_dec_blocks)])
        self.final_linear = nn.Linear(d_model, n_out)
        self.sm = nn.Softmax()

    def forward(self, x):
        PE = get_positional_encoding(self.seq_len, self.d_model)
        x = x + PE.unsqueeze(0)
        enc_out = x
        for e in self.encoder_blocks:
            enc_out = e(enc_out)
        dec_out = x
        for d in self.decoder_blocks:
            dec_out = d(dec_out, enc_out)
        out = self.final_linear(out)
        out = self.sm(out)
        return out
        



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
