import torch
import torch.nn as nn
from transformers.transformer.encoder import EncoderBlock
from transformers.transformer.decoder import DecoderBlock
import numpy as np

class Transformer(nn.Module):
    def __init__(self, seq_len, d_model, n_enc_blocks, n_dec_blocks, n_out, mode='nlp', feature_size=None):
        super().__init__()

        # Create input projection layer
        if mode.lower() == 'nlp':
            if feature_size is None:
                raise ValueError('For NLP mode the feature_size parameter must be set to ' \
                'the vocabulary size. Currently set to None.')
            else:
                self.input_proj= nn.Embedding(feature_size, d_model)
        elif mode.lower() == 'ts' or mode.lower() == 'ci':
            if feature_size is None:
                raise ValueError('For time series or continuous input modes the feature_size ' \
                'parameter must be set to the feature_dim ([batch, seq_len, feature_dim]). ' \
                'Currently set to None.')
            else:
                self.input_proj = nn.Linear(feature_size, d_model)
        
        # Store variables and parameters
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_out = n_out

        # Build Transformer
        self.encoder_blocks = nn.ModuleList([EncoderBlock() for _ in range(n_enc_blocks)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock() for _ in range(n_dec_blocks)])
        self.final_linear = nn.Linear(d_model, n_out)
        self.sm = nn.Softmax()

    def forward(self, x):
        # Project input
        x = self.input_proj(x)

        # Positional Encoding
        PE = get_positional_encoding(self.seq_len, self.d_model)
        x = x + PE.unsqueeze(0)

        # Encode
        enc_out = x
        for e in self.encoder_blocks:
            enc_out = e(enc_out)
        
        # Additional positional encoding for output
        PE_2 = get_positional_encoding(self.seq_len, self.d_model)
        dec_out = enc_out + PE_2.unsqueeze(0)

        # Decode
        for d in self.decoder_blocks:
            dec_out = d(dec_out, enc_out)
        
        # Linear and Softmax
        out = self.final_linear(out)
        out = self.sm(out)
        return out
        
# Positional encoding function
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
