"""Fisher LoRA adapters with full K-FAC statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import logging

import torch
from torch import Tensor, nn


logger = logging.getLogger(__name__)


@dataclass
class FisherLoRAConfig:
    """Configuration options controlling Fisher-LoRA behavior."""

    rank: int
    ema_decay: float = 0.95
    update_interval: int = 32
    damping: float = 1.0e-5
    min_factor_eig: float = 1.0e-6
    freeze_base: bool = True
    train_U: bool = True
    train_V: bool = True

    use_S: bool = True                 # enable diagonal S
    train_S: bool = True               # whether S is trainable
    s_init_value: float = 0.0          # usually 0.0 for zero adapter at start
    init_scale: float = 1.0e-3
    factor_dtype: torch.dtype = torch.float32
    track_fisher: bool = True
    whiteness_log_interval: Optional[int] = None

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
        if not isinstance(self.rank, int):
            raise TypeError("rank must be an integer")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        if self.whiteness_log_interval is not None and self.whiteness_log_interval <= 0:
            raise ValueError("whiteness_log_interval must be positive when provided")


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
        self.rank = int(min(config.rank, in_features, out_features))
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
        # Defer whitening refresh until the next forward pass when set to True
        self.register_buffer(
            "refresh_pending",
            torch.zeros((), dtype=torch.bool, device=device),
        )

        if self.rank > 0:
            init_scale = self.config.init_scale
            #u = torch.randn(out_features, self.rank, device=device, dtype=dtype) * init_scale
            #v = torch.randn(in_features, self.rank, device=device, dtype=dtype) * init_scale
            if self.config.use_S:
                # Small-random U,V; S = 0 → ΔW starts at 0, grads flow to S immediately
                u = torch.randn(out_features, self.rank, device=device, dtype=dtype) * (1.0 / (out_features ** 0.5))
                v = torch.randn(in_features,  self.rank, device=device, dtype=dtype) * (1.0 / (in_features  ** 0.5))
                s = torch.full((self.rank,), fill_value=self.config.s_init_value, device=device, dtype=dtype)
            else:
                # UV-only: LoRA-style safe init (U random, V zero) to keep ΔW=0
                u = torch.randn(out_features, self.rank, device=device, dtype=dtype) * (1.0 / (out_features ** 0.5))
                v = torch.zeros(in_features,  self.rank, device=device, dtype=dtype)
                s = None
            self.U = nn.Parameter(u, requires_grad=self.config.train_U)
            self.V = nn.Parameter(v, requires_grad=self.config.train_V)

            if self.config.use_S:
                self.S = nn.Parameter(s, requires_grad=self.config.train_S)
            else:
                self.register_parameter("S", None)

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

        # Set by attach_fisher_lora to identify metrics in logs
        self._fisher_lora_name: Optional[str] = None
        self.just_reparam = False
        self._refresh_whiteners(cache_bases=True)
        self.just_reparam = False

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

        # If whitening refresh is pending, perform it now before using the adapters
        if self.refresh_pending.item():
            self._refresh_whiteners(cache_bases=True)
            self.refresh_pending.zero_()

        L, R = self._skinny_bases()
        input_2d = input.reshape(-1, self.in_features)
        proj = torch.matmul(input_2d, R)  # (N, r)

        # NEW: scale per mode with diagonal S
        if self.config.use_S:
            proj = proj * self.S.to(proj.dtype)  # broadcast over batch dim

        adapter_2d = torch.matmul(proj, L.T)     # (N, dout)
        adapter = adapter_2d.reshape(*output.shape)
        result = output + adapter


        if self.training and self.config.track_fisher:
            self._register_fisher_hooks(input, result)
        return result

    def _register_fisher_hooks(self, input: Tensor, output: Tensor) -> None:
        activations = input.detach()
        if not output.requires_grad:
            assert False, "output must require grad"

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

            if self.config.track_fisher:
                log_interval = self.config.whiteness_log_interval or self.config.update_interval
                step = int(self.step_count.item())
                if step == 1 or (log_interval and step % log_interval == 0):
                    eye_in = torch.eye(
                        self.in_features,
                        dtype=factor_dtype,
                        device=self.A_ema.device,
                    )
                    eye_out = torch.eye(
                        self.out_features,
                        dtype=factor_dtype,
                        device=self.B_ema.device,
                    )
                    a_damped = self.A_ema + self.config.damping * eye_in
                    b_damped = self.B_ema + self.config.damping * eye_out
                    e_a, e_b = self._compute_whiteness_errors(a_damped, b_damped, eye_in, eye_out)
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(
                            "Fisher-LoRA whiteness (pre-refresh, step=%d): eA=%.3e eB=%.3e",
                            step,
                            float(e_a),
                            float(e_b),
                        )
                    self._log_whiteness_metrics(float(e_a), float(e_b), phase="pre_refresh")
            if int(self.step_count.item()) % self.config.update_interval == 0:
                # Defer refresh to next forward pass
                self.refresh_pending.fill_(True)

    def _skinny_bases(self) -> tuple[Tensor, Tensor]:
        if self.rank == 0:
            raise RuntimeError("adapter rank is zero")
        factor_dtype = self.config.factor_dtype
        adapter_dtype = self.base.weight.dtype
        if self.config.train_U:
            L = (self.B_inv_sqrt @ self.U.to(factor_dtype)).to(adapter_dtype)
        else:
            if not self._cache_l_valid:
                with torch.no_grad():
                    tmp = self.B_inv_sqrt @ self.U.detach().to(factor_dtype)
                    self.L0_cached.copy_(tmp.to(adapter_dtype))
                    self._cache_l_valid = True
            L = self.L0_cached
        if self.config.train_V:
            R = (self.A_inv_sqrt @ self.V.to(factor_dtype)).to(adapter_dtype)
        else:
            if not self._cache_r_valid:
                with torch.no_grad():
                    #tmp = torch.matmul(self.A_inv_sqrt, self.V.detach().to(factor_dtype))
                    tmp = self.A_inv_sqrt @ self.V.detach().to(factor_dtype)
                    self.R0_cached.copy_(tmp.to(adapter_dtype))
                    self._cache_r_valid = True
            R = self.R0_cached
        return L, R

    def _refresh_whiteners(self, *, cache_bases: bool) -> None:
        with torch.no_grad():
            adapter_jump = None
            U_t: Optional[Tensor] = None
            V_t: Optional[Tensor] = None
            delta_old: Optional[Tensor] = None
            B_inv_old: Optional[Tensor] = None
            A_inv_old: Optional[Tensor] = None
            if self.rank > 0:
                factor_dtype = self.config.factor_dtype
                B_inv_old = self.B_inv_sqrt.clone()
                A_inv_old = self.A_inv_sqrt.clone()
                U_t = self.U.detach().to(factor_dtype)
                V_t = self.V.detach().to(factor_dtype)
                if self.config.track_fisher:
                    L_old = B_inv_old @ U_t
                    R_old = A_inv_old @ V_t
                    delta_old = L_old @ R_old.T

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

            if self.config.track_fisher:
                if self.rank > 0 and U_t is not None and V_t is not None and delta_old is not None:
                    L_new = self.B_inv_sqrt @ U_t
                    R_new = self.A_inv_sqrt @ V_t
                    delta_new = L_new @ R_new.T
                    diff = (delta_new - delta_old).norm()
                    denom = delta_old.norm() + 1.0e-12
                    adapter_jump = float((diff / denom).item())
                e_a, e_b = self._compute_whiteness_errors(a, b, eye_in, eye_out)
                if logger.isEnabledFor(logging.INFO):
                    if adapter_jump is not None:
                        logger.info(
                            "Fisher-LoRA whitener refreshed (step=%d): eA=%.3e eB=%.3e jump=%.3e",
                            int(self.step_count.item()),
                            float(e_a),
                            float(e_b),
                            adapter_jump,
                        )
                    else:
                        logger.info(
                            "Fisher-LoRA whitener refreshed (step=%d): eA=%.3e eB=%.3e",
                            int(self.step_count.item()),
                            float(e_a),
                            float(e_b),
                        )
                self._log_whiteness_metrics(float(e_a), float(e_b), phase="post_refresh")
                if adapter_jump is not None:
                    self._log_adapter_jump(adapter_jump)
            if self.rank > 0 and B_inv_old is not None and A_inv_old is not None and U_t is not None and V_t is not None:
                S_B = torch.linalg.solve(self.B_inv_sqrt, B_inv_old)
                S_A = torch.linalg.solve(self.A_inv_sqrt, A_inv_old)
                new_U = S_B @ U_t
                new_V = S_A @ V_t
                self.U.copy_(new_U.to(self.U.dtype))
                self.V.copy_(new_V.to(self.V.dtype))
                self.just_reparam = True
            if cache_bases and self.rank > 0:
                self._cache_l_valid = False
                self._cache_r_valid = False

    def refresh(self) -> None:
        """Force-refresh whitening factors from the current EMAs."""
        self._refresh_whiteners(cache_bases=True)
        # Clear any pending refresh since we just performed it
        self.refresh_pending.zero_()

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
            # Clear any pending refresh since we just performed it
            self.refresh_pending.zero_()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, "
            f"ema_decay={self.config.ema_decay}, update_interval={self.config.update_interval}"
        )
    
    @staticmethod
    def _matrix_inv_sqrt(matrix: Tensor, min_eig: float) -> Tensor:
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        eigvals = eigvals.clamp_min(min_eig).rsqrt()
        return (eigvecs * eigvals) @ eigvecs.T

    def _compute_whiteness_errors(
        self,
        a_matrix: Tensor,
        b_matrix: Tensor,
        eye_in: Tensor,
        eye_out: Tensor,
    ) -> tuple[Tensor, Tensor]:
        whitened_a = self.A_inv_sqrt @ a_matrix @ self.A_inv_sqrt
        whitened_b = self.B_inv_sqrt @ b_matrix @ self.B_inv_sqrt
        e_a = (whitened_a - eye_in).norm() / max(1, self.in_features)
        e_b = (whitened_b - eye_out).norm() / max(1, self.out_features)
        return e_a, e_b

    def _log_whiteness_metrics(self, e_a: float, e_b: float, *, phase: str) -> None:
        """Optionally log whiteness metrics to wandb."""

        try:
            import wandb  # type: ignore
        except ImportError:
            return

        run = getattr(wandb, "run", None)
        if run is None:
            return

        prefix = self._fisher_lora_name or "fisher_lora"
        metrics = {
            f"fisher/{prefix}/whiteness_{phase}_eA": e_a,
            f"fisher/{prefix}/whiteness_{phase}_eB": e_b,
        }
        step = int(self.step_count.item()) if hasattr(self, "step_count") else None
        wandb.log(metrics, step=step)

    def _log_adapter_jump(self, jump: float) -> None:
        """Optionally log adapter jump metrics to wandb."""

        try:
            import wandb  # type: ignore
        except ImportError:
            return

        run = getattr(wandb, "run", None)
        if run is None:
            return

        prefix = self._fisher_lora_name or "fisher_lora"
        metrics = {
            f"fisher/{prefix}/adapter_jump": jump,
        }
        step = int(self.step_count.item()) if hasattr(self, "step_count") else None
        wandb.log(metrics, step=step)


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
                if not getattr(child, "_fisher_lora_name", None):
                    child._fisher_lora_name = full_name
                replaced[full_name] = child
                continue
            if isinstance(child, nn.Linear) and _should_replace(full_name):
                fisher = FisherLoRALinear.from_linear(child, config=config)
                fisher._fisher_lora_name = full_name
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
