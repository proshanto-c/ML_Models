import torch
from transformers.transformer.mhsa import MultiHeadSelfAttention
import pytest

def test_shape(subtests):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d_model = 20
    num_heads = 5
    mhsa = MultiHeadSelfAttention(d_model, num_heads).to(device)
    V_K = torch.randn(1, 5, d_model, device=device)
    Q = torch.randn(1, 5, d_model, device=device)
    
    with subtests.test('Check output shape matches input shape'):
        assert mhsa(V_K, Q).size() == V_K.size()
    
    with subtests.test('Check that V_K and Q dimensions must match'):
        with pytest.raises(ValueError, match='Mismatched batch size for inputs V_K and Q for MHSA.'):
            Q_b = torch.randn(2, 5, d_model, device=device)
            mhsa(V_K, Q_b)
        with pytest.raises(ValueError, match='Mismatched sequence lengths for inputs V_K and Q for MHSA.'):
            Q_t = torch.randn(1, 6, d_model, device=device)
            mhsa(V_K, Q_t)

    with subtests.test('Check input d must match d_model'):
        with pytest.raises(ValueError, match='Third dimension of input q into MHSA does not equal d_model.'):
            Q_ii = torch.randn(1, 5, d_model+1, device=device)
            mhsa(V_K, Q_ii)
        with pytest.raises(ValueError, match='Third dimension of input v_k into MHSA does not equal d_model.'):
            V_K_ii = torch.randn(1, 5, d_model+1, device=device)
            mhsa(V_K_ii, Q)
    
def test_divisibility():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d_model = 20
    num_heads = 6
    with pytest.raises(ValueError, match='d_model must be divisible by the number of heads for MHSA'):
        _ = MultiHeadSelfAttention(d_model, num_heads).to(device)