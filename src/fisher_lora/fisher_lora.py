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
    damping: float = 1.0e-5
    min_factor_eig: float = 1.0e-6
    freeze_base: bool = True
    train_U: bool = True
    train_V: bool = True

    use_S: bool = True                 
    train_S: bool = True               
    s_init_value: float = 0.0          #0.0 might cause grad flow issues
    init_scale: float = 1.0e-3
    factor_dtype: torch.dtype = torch.float32
    track_fisher: bool = True
    whiteness_log_interval: Optional[int] = None
    diagnostics_interval: int = 1

    def __post_init__(self) -> None:
        if not (0.0 <= self.rank):
            raise ValueError("rank must be non-negative")
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
        if self.diagnostics_interval <= 0:
            raise ValueError("diagnostics_interval must be positive")


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

        self._calibrating = False
        self._fisher_frozen = False

        self.register_buffer(
            "_A_sum",
            torch.zeros(
                in_features,
                in_features,
                dtype=self.config.factor_dtype,
                device=device,
            ),
        )
        self.register_buffer(
            "_B_sum",
            torch.zeros(
                out_features,
                out_features,
                dtype=self.config.factor_dtype,
                device=device,
            ),
        )
        self.register_buffer(
            "_n_calib",
            torch.zeros((), dtype=torch.long, device=device),
        )
        self.register_buffer(
            "_diag_step_mod",
            torch.zeros((), dtype=torch.long, device=device),
        )
        self._last_proj_rms_preS: Optional[float] = None
        self._last_proj_rms_postS: Optional[float] = None
        self._last_adapter_rms: Optional[float] = None
        self._last_energy_capture: Optional[float] = None

        if self.rank > 0:
            init_scale = self.config.init_scale
            #u = torch.randn(out_features, self.rank, device=device, dtype=dtype) * init_scale
            #v = torch.randn(in_features, self.rank, device=device, dtype=dtype) * init_scale
            if self.config.use_S:
                u = torch.randn(out_features, self.rank, device=device, dtype=dtype) * (1.0 / (out_features ** 0.5))
                v = torch.zeros(in_features,  self.rank, device=device, dtype=dtype)   # keep ΔW=0
                s0 = self.config.s_init_value if self.config.s_init_value != 0.0 else 1.0  # or alpha/r
                s = torch.full((self.rank,), fill_value=s0, device=device, dtype=dtype)
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
            self.register_buffer(
                "last_refresh_step",
                torch.zeros((), dtype=torch.long, device=device),
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
        self._last_adapter_jump = 0.0
        # Transient matrices for transporting optimizer moments after reparameterization
        self._T_B_invT: Optional[Tensor] = None
        self._T_A_invT: Optional[Tensor] = None
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
        
        with torch.amp.autocast(device_type="cuda", enabled=False):
            input_2d_f32 = input.reshape(-1, self.in_features).to(torch.float32)
            L32 = L.to(torch.float32)
            R32 = R.to(torch.float32)

            proj32 = input_2d_f32 @ R32              # (N, r)
            if self.config.use_S and self.S is not None:
                proj32 = proj32 * self.S.to(torch.float32)  # per-mode scale

            adapter_2d32 = proj32 @ L32.T            # (N, d_out)

        adapter = adapter_2d32.to(output.dtype).reshape_as(output)
        result = output + adapter
        '''
        input_2d = input.reshape(-1, self.in_features)
        proj = torch.matmul(input_2d, R)  # (N, r)

        # NEW: scale per mode with diagonal S
        if self.config.use_S:
            proj = proj * self.S.to(proj.dtype)  # broadcast over batch dim

        adapter_2d = torch.matmul(proj, L.T)     # (N, dout)
        adapter = adapter_2d.reshape(*output.shape)
        result = output + adapter
        '''
        # ---- numeric diagnostics (fp32 RMS) ----
        try:
            with torch.no_grad():
                R_32 = R.to(torch.float32)
                L_32 = L.to(torch.float32)
                proj_preS = input_2d_f32 @ R_32
                proj_postS = (
                    proj_preS * self.S.to(torch.float32)
                    if (self.config.use_S and self.S is not None)
                    else proj_preS
                )
                adapter_2d_32 = proj_postS @ L_32.T
                self._last_proj_rms_preS = float(proj_preS.pow(2).mean().sqrt().item())
                self._last_proj_rms_postS = float(proj_postS.pow(2).mean().sqrt().item())
                self._last_adapter_rms = float(adapter_2d_32.pow(2).mean().sqrt().item())
        except Exception:
            pass
       
        if self.training and self.config.track_fisher:
            self._register_fisher_hooks(input, result)
        return result

    def begin_calibration(self) -> None:
        """Enter calibration mode: accumulate unbiased Fisher factors."""
        self._calibrating = True
        self._fisher_frozen = False
        with torch.no_grad():
            self._A_sum.zero_()
            self._B_sum.zero_()
            self._n_calib.zero_()
            self.step_count.zero_()
            self._diag_step_mod.zero_()
            if hasattr(self, "refresh_pending"):
                self.refresh_pending.zero_()

    @torch.no_grad()
    def finalize_calibration(
        self,
        *,
        shrink: float = 0.10,
        damping: float = 1e-4,
        min_eig: float = 1e-4,
    ) -> None:
        """Finish calibration: compute whiteners once and freeze Fisher statistics."""
        n = int(self._n_calib.item())
        if n > 0:
            A = self._A_sum / float(n)
            B = self._B_sum / float(n)
            IA = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
            IB = torch.eye(B.shape[0], dtype=B.dtype, device=B.device)
            tauA = (A.trace() / A.shape[0]).clamp_min(1e-12)
            tauB = (B.trace() / B.shape[0]).clamp_min(1e-12)
            A = (1.0 - shrink) * A + (shrink * tauA) * IA + damping * IA
            B = (1.0 - shrink) * B + (shrink * tauB) * IB + damping * IB
            self.A_ema.copy_(A)
            self.B_ema.copy_(B)
            self.A_inv_sqrt.copy_(self._matrix_inv_sqrt(A, min_eig))
            self.B_inv_sqrt.copy_(self._matrix_inv_sqrt(B, min_eig))
            if self.rank > 0:
                self._cache_l_valid = False
                self._cache_r_valid = False
                self._T_B_invT = None
                self._T_A_invT = None
                self.just_reparam = False
        self._calibrating = False
        self._fisher_frozen = True
        if hasattr(self, "refresh_pending"):
            self.refresh_pending.zero_()

    def _register_fisher_hooks(self, input: Tensor, output: Tensor) -> None:
        activations = input.detach()
        if not output.requires_grad:
            assert False, "output must require grad"

        def _capture_grad(grad_output: Tensor) -> Tensor:
            self._update_fisher_stats(activations, grad_output.detach())
            return grad_output

        output.register_hook(_capture_grad)

    @torch.no_grad()
    def delta_w_fro_norm(self) -> float:
        if self.rank == 0:
            return 0.0
        L, R = self._skinny_bases()
        l2 = (L * L).sum(dim=0)
        r2 = (R * R).sum(dim=0)
        if self.config.use_S and (self.S is not None):
            s2 = self.S.to(L.dtype) ** 2
        else:
            s2 = torch.ones_like(l2)
        val2 = (s2 * l2 * r2).sum()
        return float(val2.sqrt().item())

    @torch.no_grad()
    def _energy_capture_from_batch(self, a: Tensor, g: Tensor) -> Optional[float]:
        if self.rank == 0:
            return None
        if a.numel() == 0 or g.numel() == 0:
            return None
        N = a.shape[0]
        if N == 0:
            return None
        G = (g.T @ a) / float(N)
        B_inv = self.B_inv_sqrt.to(torch.float32)
        A_inv = self.A_inv_sqrt.to(torch.float32)
        Gt = B_inv @ G @ A_inv
        L, R = self._skinny_bases()
        L_32 = L.to(torch.float32)
        R_32 = R.to(torch.float32)
        try:
            QL, _ = torch.linalg.qr(L_32, mode="reduced")
            QR, _ = torch.linalg.qr(R_32, mode="reduced")
        except RuntimeError:
            return None
        M = QL.T @ Gt @ QR
        num = (M * M).sum()
        den = (Gt * Gt).sum() + 1e-30
        return float((num / den).item())

    @torch.no_grad()
    def get_last_diagnostics(self) -> dict:
        return {
            "proj_rms_preS": self._last_proj_rms_preS,
            "proj_rms_postS": self._last_proj_rms_postS,
            "adapter_rms": self._last_adapter_rms,
            "energy_capture": self._last_energy_capture,
            "deltaW_fro": self.delta_w_fro_norm(),
        }

    def _update_fisher_stats(self, activations: Tensor, grad_output: Tensor) -> None:
        if self.rank == 0:
            return
        if getattr(self, "_fisher_frozen", False):
            return

        with torch.no_grad():
            a = activations.reshape(-1, self.in_features)
            g = grad_output.reshape(-1, self.out_features)
            if a.numel() == 0 or g.numel() == 0:
                return

            fdtype = self.config.factor_dtype
            a_t = a.to(fdtype)
            g_t = g.to(fdtype)
            Ab = (a_t.T @ a_t) / a.shape[0]
            Bb = (g_t.T @ g_t) / g.shape[0]

            if getattr(self, "_calibrating", False):
                self._A_sum.add_(Ab)
                self._B_sum.add_(Bb)
                self._n_calib.add_(1)
                self.step_count.add_(1)
                return

            # No online updates when not calibrating.
            return

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
            self.just_reparam = False
            self._T_B_invT = None
            self._T_A_invT = None
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
                    #delta_old = L_old @ R_old.T
                    delta_old = (L_old * self.S.view(1,-1)) @ R_old.T 

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
                    #delta_new = L_new @ R_new.T
                    delta_new = (L_new * self.S.view(1,-1)) @ R_new.T
                    diff = (delta_new - delta_old).norm()
                    denom = delta_old.norm() + 1.0e-12
                    adapter_jump = float((diff / denom).item())
                    self.just_reparam = adapter_jump is not None and adapter_jump > 1e-6
                    self._last_adapter_jump = adapter_jump or 0.0

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

                factor_dtype = self.config.factor_dtype
                I_out = torch.eye(
                    self.out_features,
                    dtype=factor_dtype,
                    device=self.B_ema.device,
                )
                I_in = torch.eye(
                    self.in_features,
                    dtype=factor_dtype,
                    device=self.A_ema.device,
                )
                T_B_invT = torch.linalg.solve(S_B.T, I_out)
                T_A_invT = torch.linalg.solve(S_A.T, I_in)
                self._T_B_invT = T_B_invT
                self._T_A_invT = T_A_invT
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

    @torch.no_grad()
    def balance_columns(self, eps: float = 1e-12) -> None:
        if self.rank == 0: return
        u = self.U.norm(dim=0).clamp_min(eps)
        v = self.V.norm(dim=0).clamp_min(eps)
        s = (u / v).sqrt()
        self.U.div_(s)     # U_i <- U_i / s_i
        self.V.mul_(s)     # V_i <- V_i * s_i


    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, "
            f"damping={self.config.damping}"
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

        diff_a = self.A_inv_sqrt @ a_matrix @ self.A_inv_sqrt - eye_in
        diff_b = self.B_inv_sqrt @ b_matrix @ self.B_inv_sqrt - eye_out

        e_a = torch.linalg.norm(diff_a, ord='fro') / (self.in_features ** 0.5)
        e_b = torch.linalg.norm(diff_b, ord='fro') / (self.out_features ** 0.5)

        return e_a, e_b

    def _log_whiteness_metrics(self, e_a: float, e_b: float, *, phase: str) -> None:
        pass
    def _log_adapter_jump(self, jump: float) -> None:
        pass



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
