import torch.nn as nn
from transformers.transformer.mhsa import MultiHeadSelfAttention

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.MMHA = MultiHeadSelfAttention(d_model, num_heads, mask=True)
        self.MHA = MultiHeadSelfAttention(d_model, num_heads)
        self.FF = nn.ModuleList([nn.Linear(d_model, d_model),
                                 nn.ReLU(),
                                 nn.Linear(d_model, d_model)])

    def forward(self, d, e):

        out = self.MMHA(d)
        out = nn.LayerNorm(d+out)

        out2 = self.MHA(e, out)
        out2 = nn.LayerNorm(out+out2)

        out_ff = out2
        for layer in self.FF:
            out_ff = layer(out_ff)
        out_ff = nn.LayerNorm(out2+out_ff)
        
        return nn.Softmax(out_ff)