import torch.nn as nn
from transformers.transformer.mhsa import MultiHeadSelfAttention

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.MHA = MultiHeadSelfAttention(d_model, num_heads)
        self.FF = nn.ModuleList([nn.Linear(d_model, d_model),
                                 nn.ReLU(),
                                 nn.Linear(d_model, d_model)])

    def forward(self, e):
        out = self.MHA(e)
        out = nn.LayerNorm(e+out)

        out_ff = out
        for layer in self.FF:
            out_ff = layer(out_ff)
        out_ff = nn.LayerNorm(out+out_ff)
        
        return nn.Softmax(out_ff)