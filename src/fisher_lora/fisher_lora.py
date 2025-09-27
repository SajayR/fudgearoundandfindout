"""Fisher LoRA adapters with full K-FAC statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
from torch import Tensor, nn


@dataclass
class FisherLoRAConfig:
    """Configuration options controlling Fisher-LoRA behavior."""

    rank: int
    ema_decay: float = 0.95
    update_interval: int = 32
    damping: float = 1.0e-5
    min_factor_eig: float = 1.0e-6
    freeze_base: bool = True
    train_U: bool = False
    train_V: bool = False
    train_S: bool = True
    init_scale: float = 1.0e-2
    factor_dtype: torch.dtype = torch.float32
    track_fisher: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.rank):
            raise ValueError("rank must be non-negative")
        if not (0.0 < self.ema_decay < 1.0):
            raise ValueError("ema_decay must be in (0, 1)")
        if self.update_interval <= 0:
            raise ValueError("update_interval must be positive")
        if self.damping <= 0.0:
            raise ValueError("damping must be positive")
        if self.min_factor_eig <= 0.0:
            raise ValueError("min_factor_eig must be positive")


class FisherLoRALinear(nn.Module):
    """Linear layer augmented with a Fisher-whitened low-rank adapter."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        config: Optional[FisherLoRAConfig] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = FisherLoRAConfig(rank=0)
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(config.rank, in_features, out_features)
        self.base = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
        if self.config.freeze_base:
            for param in self.base.parameters():
                param.requires_grad_(False)

        factor_dtype = self.config.factor_dtype
        self.register_buffer(
            "A_ema",
            torch.eye(in_features, dtype=factor_dtype, device=device),
        )
        self.register_buffer(
            "B_ema",
            torch.eye(out_features, dtype=factor_dtype, device=device),
        )
        self.register_buffer(
            "A_inv_sqrt",
            torch.eye(in_features, dtype=factor_dtype, device=device),
        )
        self.register_buffer(
            "B_inv_sqrt",
            torch.eye(out_features, dtype=factor_dtype, device=device),
        )
        self.register_buffer(
            "step_count",
            torch.zeros((), dtype=torch.long, device=device),
        )

        if self.rank > 0:
            init_scale = self.config.init_scale
            u = torch.randn(out_features, self.rank, device=device, dtype=dtype) * init_scale
            v = torch.randn(in_features, self.rank, device=device, dtype=dtype) * init_scale
            #s = torch.zeros(self.rank, self.rank, device=device, dtype=dtype)
            s = torch.eye(self.rank, device=device, dtype=dtype) * init_scale

            self.U = nn.Parameter(u, requires_grad=self.config.train_U)
            self.V = nn.Parameter(v, requires_grad=self.config.train_V)
            self.S = nn.Parameter(s, requires_grad=self.config.train_S)
            self.register_buffer(
                "L0_cached",
                torch.zeros(out_features, self.rank, device=device, dtype=dtype),
            )
            self.register_buffer(
                "R0_cached",
                torch.zeros(in_features, self.rank, device=device, dtype=dtype),
            )
            self._cache_l_valid = False
            self._cache_r_valid = False
        else:
            self.register_parameter("U", None)
            self.register_parameter("V", None)
            self.register_parameter("S", None)
            self.register_buffer("L0_cached", torch.zeros(0, 0))
            self.register_buffer("R0_cached", torch.zeros(0, 0))
            self._cache_l_valid = False
            self._cache_r_valid = False

        self._refresh_whiteners(cache_bases=True)

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        config: Optional[FisherLoRAConfig] = None,
    ) -> "FisherLoRALinear":
        """Clone weights from a plain Linear layer into a Fisher-LoRA layer."""

        new_module = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            config=config,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        with torch.no_grad():
            new_module.base.weight.copy_(module.weight)
            if module.bias is not None:
                new_module.base.bias.copy_(module.bias)
        return new_module

    def forward(self, input: Tensor) -> Tensor:
        output = self.base(input)
        if self.rank == 0:
            return output

        L, R = self._skinny_bases()
        right = torch.matmul(self.S, R.T)
        input_2d = input.reshape(-1, self.in_features)
        proj = torch.matmul(input_2d, right.T)
        adapter_2d = torch.matmul(proj, L.T)
        adapter = adapter_2d.reshape(*output.shape)
        result = output + adapter

        if self.training and self.config.track_fisher:
            self._register_fisher_hooks(input, result)
        return result

    def _register_fisher_hooks(self, input: Tensor, output: Tensor) -> None:
        activations = input.detach()

        def _capture_grad(grad_output: Tensor) -> Tensor:
            self._update_fisher_stats(activations, grad_output.detach())
            return grad_output

        output.register_hook(_capture_grad)

    def _update_fisher_stats(self, activations: Tensor, grad_output: Tensor) -> None:
        if self.rank == 0:
            return
        with torch.no_grad():
            a = activations.reshape(-1, self.in_features)
            g = grad_output.reshape(-1, self.out_features)
            if a.numel() == 0 or g.numel() == 0:
                return
            factor_dtype = self.config.factor_dtype
            a_t = a.to(factor_dtype)
            g_t = g.to(factor_dtype)
            a_factor = torch.matmul(a_t.T, a_t) / a.shape[0]
            g_factor = torch.matmul(g_t.T, g_t) / g.shape[0]
            decay = self.config.ema_decay
            self.A_ema.mul_(decay).add_(a_factor, alpha=1.0 - decay)
            self.B_ema.mul_(decay).add_(g_factor, alpha=1.0 - decay)
            self.step_count.add_(1)
            if int(self.step_count.item()) % self.config.update_interval == 0:
                self._refresh_whiteners(cache_bases=True)

    def _skinny_bases(self) -> tuple[Tensor, Tensor]:
        if self.rank == 0:
            raise RuntimeError("adapter rank is zero")
        factor_dtype = self.config.factor_dtype
        adapter_dtype = self.base.weight.dtype
        if self.config.train_U:
            B = self.B_inv_sqrt.to(self.U.dtype)
            L = torch.matmul(B, self.U)
            if L.dtype != adapter_dtype:
                L = L.to(adapter_dtype)
        else:
            if not self._cache_l_valid:
                with torch.no_grad():
                    tmp = torch.matmul(self.B_inv_sqrt, self.U.detach().to(factor_dtype))
                    self.L0_cached.copy_(tmp.to(adapter_dtype))
                    self._cache_l_valid = True
            L = self.L0_cached
        if self.config.train_V:
            A = self.A_inv_sqrt.to(self.V.dtype)
            R = torch.matmul(A, self.V)
            if R.dtype != adapter_dtype:
                R = R.to(adapter_dtype)
        else:
            if not self._cache_r_valid:
                with torch.no_grad():
                    tmp = torch.matmul(self.A_inv_sqrt, self.V.detach().to(factor_dtype))
                    self.R0_cached.copy_(tmp.to(adapter_dtype))
                    self._cache_r_valid = True
            R = self.R0_cached
        return L, R

    def _refresh_whiteners(self, *, cache_bases: bool) -> None:
        with torch.no_grad():
            eye_in = torch.eye(
                self.in_features,
                dtype=self.config.factor_dtype,
                device=self.A_ema.device,
            )
            eye_out = torch.eye(
                self.out_features,
                dtype=self.config.factor_dtype,
                device=self.B_ema.device,
            )
            a = self.A_ema + self.config.damping * eye_in
            b = self.B_ema + self.config.damping * eye_out
            min_eig = self.config.min_factor_eig
            self.A_inv_sqrt.copy_(self._matrix_inv_sqrt(a, min_eig))
            self.B_inv_sqrt.copy_(self._matrix_inv_sqrt(b, min_eig))
            if cache_bases and self.rank > 0:
                self._cache_l_valid = False
                self._cache_r_valid = False

    def refresh(self) -> None:
        """Force-refresh whitening factors from the current EMAs."""
        self._refresh_whiteners(cache_bases=True)

    def reset_fisher(self) -> None:
        """Reset Fisher statistics to their initial isotropic estimates."""
        with torch.no_grad():
            eye_in = torch.eye(
                self.in_features,
                dtype=self.config.factor_dtype,
                device=self.A_ema.device,
            )
            eye_out = torch.eye(
                self.out_features,
                dtype=self.config.factor_dtype,
                device=self.B_ema.device,
            )
            self.A_ema.copy_(eye_in)
            self.B_ema.copy_(eye_out)
            self._refresh_whiteners(cache_bases=True)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, "
            f"ema_decay={self.config.ema_decay}, update_interval={self.config.update_interval}"
        )

    @staticmethod
    def _matrix_inv_sqrt(matrix: Tensor, min_eig: float) -> Tensor:
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        eigvals = torch.clamp(eigvals, min=min_eig)
        inv_sqrt = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T
        return inv_sqrt


def attach_fisher_lora(
    module: nn.Module,
    *,
    target_modules: Optional[Iterable[str]] = None,
    config: Optional[FisherLoRAConfig] = None,
) -> Dict[str, FisherLoRALinear]:
    """Replace selected ``nn.Linear`` submodules with Fisher-LoRA variants.

    Args:
        module: Root module to search.
        target_modules: Iterable of fully-qualified module names relative to
            ``module`` that should be replaced. If ``None`` all Linear
            submodules are adapted.
        config: Shared Fisher-LoRA configuration.

    Returns:
        Mapping from fully-qualified module name to the inserted
        :class:`FisherLoRALinear` instance.
    """

    if config is None:
        raise ValueError("config must be provided")

    targets = set(target_modules) if target_modules is not None else None
    replaced: Dict[str, FisherLoRALinear] = {}

    def _should_replace(full_name: str) -> bool:
        return targets is None or full_name in targets

    def _recurse(current: nn.Module, prefix: str) -> None:
        for name, child in list(current.named_children()):
            full_name = f"{prefix}{name}" if prefix else name
            if isinstance(child, FisherLoRALinear):
                replaced[full_name] = child
                continue
            if isinstance(child, nn.Linear) and _should_replace(full_name):
                fisher = FisherLoRALinear.from_linear(child, config=config)
                current.add_module(name, fisher)
                replaced[full_name] = fisher
            else:
                _recurse(child, f"{full_name}.")

    _recurse(module, "")

    if targets is not None:
        missing = sorted(name for name in targets if name not in replaced)
        if missing:
            raise ValueError(f"Could not find target Linear modules: {missing}")

    return replaced
