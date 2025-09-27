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
    config = FisherLoRAConfig(rank=2, train_U=True, train_V=True)
    layer = FisherLoRALinear(4, 4, config=config)
    out = layer(torch.randn(2, 4))
    loss = out.sum()
    loss.backward()
    assert layer.U.grad is not None
    assert layer.V.grad is not None

def test_adapter_matches_materialized_deltaW():
    torch.manual_seed(0)
    cfg = FisherLoRAConfig(rank=3)
    layer = FisherLoRALinear(6, 5, config=cfg)
    x = torch.randn(7, 6)
    # Force a refresh so whiteners are non-identity but consistent
    layer.refresh()
    y1 = layer(x)
    # Materialize ΔW and use nn.Linear semantics (x @ W.T)
    L, R = layer._skinny_bases()
    deltaW = (L @ R.T)  # note: this is (B^{-1/2} U) (A^{-1/2} V)^T
    y2 = layer.base(x) + x @ deltaW.T
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-6)

def fro_norm(M): return (M - torch.eye(M.shape[0], dtype=M.dtype, device=M.device)).norm() / (M.shape[0] ** 0.5)

def test_whitener_whitens_ema():
    torch.manual_seed(0)
    cfg = FisherLoRAConfig(rank=2, ema_decay=0.8, update_interval=1, damping=1e-5)
    layer = FisherLoRALinear(16, 12, config=cfg)
    opt = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=1e-2)

    # collect some stats
    for _ in range(20):
        x = torch.randn(32, 16)
        (layer(x).pow(2).mean()).backward()
        opt.step(); opt.zero_grad()

    layer.refresh()
    A_damped = layer.A_ema + cfg.damping * torch.eye(16, dtype=layer.A_ema.dtype, device=layer.A_ema.device)
    B_damped = layer.B_ema + cfg.damping * torch.eye(12, dtype=layer.B_ema.dtype, device=layer.B_ema.device)

    Ax = layer.A_inv_sqrt @ A_damped @ layer.A_inv_sqrt
    Bg = layer.B_inv_sqrt @ B_damped @ layer.B_inv_sqrt

    def fro_gap(M): 
        I = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
        return (M - I).norm() / (M.shape[0] ** 0.5)
    assert fro_gap(Ax) < 1e-3
    assert fro_gap(Bg) < 1e-3

def test_whitening_reduces_anisotropy_on_skewed_inputs():
    torch.manual_seed(0)
    d_in, d_out = 16, 12
    cfg = FisherLoRAConfig(rank=2, ema_decay=0.9, update_interval=1, damping=1e-4)
    layer = FisherLoRALinear(d_in, d_out, config=cfg)
    opt = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=1e-2)

    # Create a fixed anisotropy for inputs: covariance = Q diag(scales) Q^T
    scales = torch.linspace(0.1, 3.0, d_in)
    Q, _ = torch.linalg.qr(torch.randn(d_in, d_in))
    C_half = (Q * scales.sqrt()) @ Q.T  # symmetric sqrt of covariance

    # Train while feeding anisotropic inputs; grads will also be anisotropic
    for _ in range(200):
        x_iso = torch.randn(64, d_in)
        x = x_iso @ C_half.T
        (layer(x).pow(2).mean()).backward()
        opt.step(); opt.zero_grad()

    layer.refresh()

    # Fresh anisotropic batch to evaluate whiteness
    x_eval = (torch.randn(1024, d_in) @ C_half.T)
    Ahat = (x_eval.T @ x_eval) / x_eval.shape[0]
    Ax_before = Ahat                             # identity whitener
    Ax_after  = layer.A_inv_sqrt @ Ahat @ layer.A_inv_sqrt

    def fro_gap(M): 
        I = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
        return (M - I).norm() / (M.shape[0] ** 0.5)

    assert fro_gap(Ax_after) < 0.7 * fro_gap(Ax_before)  # 30% reduction


def test_sequence_shape():
    cfg = FisherLoRAConfig(rank=2)
    layer = FisherLoRALinear(6, 4, config=cfg)
    x = torch.randn(3, 5, 6)  # (B, T, D)
    y = layer(x)
    assert y.shape == (3, 5, 4)
