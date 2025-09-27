import torch
from torch import nn

from fisher_lora import FisherLoRAConfig, FisherLoRALinear, attach_fisher_lora


def test_forward_matches_base_when_adapter_zero():
    torch.manual_seed(0)
    base = nn.Linear(6, 4)
    x = torch.randn(3, 6)
    layer = FisherLoRALinear.from_linear(base, config=FisherLoRAConfig(rank=2))
    layer.eval()
    with torch.no_grad():
        expected = base(x)
        actual = layer(x)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_fisher_statistics_update_after_backward():
    torch.manual_seed(0)
    config = FisherLoRAConfig(rank=3, ema_decay=0.5, update_interval=1)
    layer = FisherLoRALinear(5, 4, config=config)
    x = torch.randn(8, 5)
    target = torch.randn(8, 4)
    out = layer(x)
    loss = (out - target).pow(2).mean()
    loss.backward()
    eye_a = torch.eye(layer.in_features, dtype=layer.A_ema.dtype)
    eye_b = torch.eye(layer.out_features, dtype=layer.B_ema.dtype)
    assert not torch.allclose(layer.A_ema, eye_a, atol=1e-6)
    assert not torch.allclose(layer.B_ema, eye_b, atol=1e-6)


def test_attach_fisher_lora_wraps_all_linear_layers():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    config = FisherLoRAConfig(rank=2)
    replaced = attach_fisher_lora(model, config=config)
    assert isinstance(model[0], FisherLoRALinear)
    assert isinstance(model[2], FisherLoRALinear)
    assert set(replaced.keys()) == {"0", "2"}


def test_trainable_u_v_receive_gradients():
    torch.manual_seed(0)
    config = FisherLoRAConfig(rank=2, train_U=True, train_V=True, train_S=True)
    layer = FisherLoRALinear(4, 4, config=config)
    out = layer(torch.randn(2, 4))
    loss = out.sum()
    loss.backward()
    assert layer.U.grad is not None
    assert layer.V.grad is not None
    assert layer.S.grad is not None
