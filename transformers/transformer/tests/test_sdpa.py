import torch
from transformers.transformer.sdpa import SDPA
import pytest

# Test the dimensions of the output from the SDPA
def test_dimensions(subtests):
    n = 5
    Q = torch.randn(n, n)
    K = torch.randn(n, n)
    V = torch.randn(n, n)
    
    with subtests.test("Check if first dimension is dim_k"): 
        assert SDPA(Q, K, V).shape[0] == 5

    with subtests.test("Check if second dimension is dim_k"): 
        assert SDPA(Q, K, V).shape[1] == 5

    with subtests.test("Check that output Tensor is 2-dimensional"):
        assert len(SDPA(Q, K, V).shape) == 2

# Test the type checking of the SDPA
def test_typing(subtests):
    Q = [[1, 2] ,[1, 2]]
    K = [[1, 2] ,[1, 2]]
    V = [[1, 2] ,[1, 2]]
    
    with subtests.test("Check that non-Tensors are not allowed"):
        with pytest.raises(TypeError, match='All inputs to the SDPA must be torch.Tensor objects'):
            SDPA(Q, K, V)
    
    Q = torch.Tensor(Q)
    Q = Q.type(dtype=torch.double)
    K = torch.Tensor(K)
    V = torch.Tensor(V)

    with subtests.test("Check that Tensors must be typed correctly"):
        with pytest.raises(TypeError, match='All input Tensors to the SDPA must be of FloatTensor'):
            SDPA(Q, K, V)

# Test the type checking when on the GPU
def test_device(subtests):
    n = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        pytest.skip('No GPU available to test GPU support.')
    Q = torch.randn(n, n, device=device)
    K = torch.randn(n, n, device=device)
    V = torch.randn(n, n)

    with subtests.test("Check that all Tensors must be on the same device"):
        with pytest.raises(TypeError, match='All input tensors to the SDPA must be on the same device'):
            SDPA(Q, K, V)
    
    V = V.to(device)

    with subtests.test("Check that GPU Tensors are allowed to pass correctly"):
        try:
            SDPA(Q, K, V)
        except TypeError:
            pytest.fail('Unexpected TypeError for GPU tensors...')

    with subtests.test("Check that GPU Tensors are type-checked correctly"):
        Q = Q.type(dtype=torch.double)
        with pytest.raises(TypeError, match='All input Tensors to the SDPA must be of FloatTensor'):
            SDPA(Q, K, V)