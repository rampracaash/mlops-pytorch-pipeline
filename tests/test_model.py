import torch
import pytest
from src.model import get_model

def test_model_initialization():
    model = get_model(architecture="resnet18", num_classes=10)
    assert model is not None
    
    # Check output shape
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (batch_size, 10)

def test_unsupported_architecture():
    with pytest.raises(ValueError):
        get_model(architecture="unsupported", num_classes=10)
