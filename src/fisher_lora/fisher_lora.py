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
    ema_decay: float = 0.99
    update_interval: int = 32
    ema_decay_start: Optional[float] = 0.70
    ema_decay_anneal_steps: Optional[int] = 1000
    update_interval_start: Optional[int] = 4
    update_interval_anneal_steps: Optional[int] = 2500
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
    use_fisher_frame_optim: bool = True
    ff_lr: float = 1e-3
    ff_betas: tuple[float, float] = (0.9, 0.999)
    ff_eps: float = 1e-8
    ff_weight_decay: float = 0.0
    ff_amsgrad: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.rank):
            raise ValueError("rank must be non-negative")
        if not (0.0 < self.ema_decay < 1.0):
            raise ValueError("ema_decay must be in (0, 1)")
        if (self.ema_decay_start is None) != (self.ema_decay_anneal_steps is None):
            raise ValueError("ema_decay_start and ema_decay_anneal_steps must be provided together")
        if self.ema_decay_start is not None:
            if not (0.0 < self.ema_decay_start < 1.0):
                raise ValueError("ema_decay_start must be in (0, 1) when provided")
        if self.ema_decay_anneal_steps is not None and self.ema_decay_anneal_steps <= 0:
            raise ValueError("ema_decay_anneal_steps must be positive when provided")
        if self.ema_decay_start is not None and self.ema_decay_start > self.ema_decay:
            raise ValueError("ema_decay_start cannot exceed ema_decay")
        if self.update_interval <= 0:
            raise ValueError("update_interval must be positive")
        if (self.update_interval_start is None) != (self.update_interval_anneal_steps is None):
            raise ValueError(
                "update_interval_start and update_interval_anneal_steps must be provided together"
            )
        if self.update_interval_start is not None:
            if self.update_interval_start <= 0:
                raise ValueError("update_interval_start must be positive when provided")
        if self.update_interval_anneal_steps is not None and self.update_interval_anneal_steps <= 0:
            raise ValueError("update_interval_anneal_steps must be positive when provided")
        if (
            self.update_interval_start is not None
            and self.update_interval_start > self.update_interval
        ):
            raise ValueError("update_interval_start cannot exceed update_interval")
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
        self.register_buffer(
            "_diag_step_mod",
            torch.zeros((), dtype=torch.long, device=device),
        )
        self._last_proj_rms_preS: Optional[float] = None
        self._last_proj_rms_postS: Optional[float] = None
        self._last_adapter_rms: Optional[float] = None
        self._last_energy_capture: Optional[float] = None
        self.register_buffer(
            "_last_eA",
            torch.zeros((), dtype=factor_dtype, device=device),
        )
        self.register_buffer(
            "_last_eB",
            torch.zeros((), dtype=factor_dtype, device=device),
        )

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

            state_device = self.base.weight.device
            self.register_buffer(
                "ff_step",
                torch.zeros((), dtype=torch.long, device=state_device),
            )
            self.register_buffer(
                "mL",
                torch.zeros(out_features, self.rank, dtype=torch.float32, device=state_device),
            )
            self.register_buffer(
                "vL",
                torch.zeros(out_features, self.rank, dtype=torch.float32, device=state_device),
            )
            self.register_buffer(
                "mR",
                torch.zeros(in_features, self.rank, dtype=torch.float32, device=state_device),
            )
            self.register_buffer(
                "vR",
                torch.zeros(in_features, self.rank, dtype=torch.float32, device=state_device),
            )
            self.register_buffer(
                "mS",
                torch.zeros(self.rank, dtype=torch.float32, device=state_device),
            )
            self.register_buffer(
                "vS",
                torch.zeros(self.rank, dtype=torch.float32, device=state_device),
            )
            if self.config.ff_amsgrad:
                self.register_buffer("vhL_max", torch.zeros_like(self.vL))
                self.register_buffer("vhR_max", torch.zeros_like(self.vR))
                self.register_buffer("vhS_max", torch.zeros_like(self.vS))
            else:
                self.register_buffer("vhL_max", torch.tensor([], device=state_device))
                self.register_buffer("vhR_max", torch.tensor([], device=state_device))
                self.register_buffer("vhS_max", torch.tensor([], device=state_device))

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
        self._last_adapter_jump = 0.0
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

        # ---- D2 numeric range diagnostics (fp32 RMS, cheap) ----
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

    @torch.no_grad()
    def fisher_frame_adamw_step(self) -> Optional[float]:
        """Perform one AdamW update in the Fisher frame and map increments back."""
        if self.rank == 0 or not self.config.use_fisher_frame_optim:
            return None

        gU = None
        if getattr(self, "U", None) is not None and self.U.requires_grad:
            gU = getattr(self.U, "grad", None)
        gV = None
        if getattr(self, "V", None) is not None and self.V.requires_grad:
            gV = getattr(self.V, "grad", None)
        gS = None
        if getattr(self, "S", None) is not None and self.S.requires_grad:
            gS = getattr(self.S, "grad", None)

        if gU is None and gV is None and gS is None:
            return None

        B_inv = self.B_inv_sqrt.to(torch.float32)
        A_inv = self.A_inv_sqrt.to(torch.float32)

        gL = torch.zeros_like(self.mL)
        if gU is not None:
            gL = torch.linalg.solve(B_inv, gU.detach().to(torch.float32))

        gR = torch.zeros_like(self.mR)
        if gV is not None:
            gR = torch.linalg.solve(A_inv, gV.detach().to(torch.float32))

        gS32: Optional[Tensor] = None
        if gS is not None:
            gS32 = gS.detach().to(torch.float32)

        if self.config.ff_weight_decay > 0.0:
            decay = self.config.ff_weight_decay
            if getattr(self, "U", None) is not None:
                L32 = (self.B_inv_sqrt @ self.U.detach().to(self.B_inv_sqrt.dtype)).to(torch.float32)
                gL.add_(L32, alpha=decay)
            if getattr(self, "V", None) is not None:
                R32 = (self.A_inv_sqrt @ self.V.detach().to(self.A_inv_sqrt.dtype)).to(torch.float32)
                gR.add_(R32, alpha=decay)
            if gS32 is not None and getattr(self, "S", None) is not None:
                gS32.add_(self.S.detach().to(torch.float32), alpha=decay)

        beta1, beta2 = self.config.ff_betas
        eps = self.config.ff_eps
        lr = self.config.ff_lr

        self.mL.mul_(beta1).add_(gL, alpha=1 - beta1)
        self.vL.mul_(beta2).addcmul_(gL, gL, value=1 - beta2)
        self.mR.mul_(beta1).add_(gR, alpha=1 - beta1)
        self.vR.mul_(beta2).addcmul_(gR, gR, value=1 - beta2)

        if gS32 is not None:
            self.mS.mul_(beta1).add_(gS32, alpha=1 - beta1)
            self.vS.mul_(beta2).addcmul_(gS32, gS32, value=1 - beta2)

        if self.config.ff_amsgrad:
            torch.maximum(self.vhL_max, self.vL, out=self.vhL_max)
            torch.maximum(self.vhR_max, self.vR, out=self.vhR_max)
            if gS32 is not None:
                torch.maximum(self.vhS_max, self.vS, out=self.vhS_max)

        self.ff_step.add_(1)
        t = int(self.ff_step.item())
        bias1 = 1.0 - beta1 ** t
        bias2 = 1.0 - beta2 ** t

        vL_eff = self.vhL_max if self.config.ff_amsgrad else self.vL
        vR_eff = self.vhR_max if self.config.ff_amsgrad else self.vR

        mL_hat = self.mL / bias1
        vL_hat = vL_eff / bias2
        mR_hat = self.mR / bias1
        vR_hat = vR_eff / bias2

        eff_vals = []
        if mL_hat.numel():
            eff_vals.append((mL_hat.abs() / (vL_hat.sqrt() + eps)).mean())
        if mR_hat.numel():
            eff_vals.append((mR_hat.abs() / (vR_hat.sqrt() + eps)).mean())

        dL = -lr * mL_hat / (vL_hat.sqrt() + eps)
        dR = -lr * mR_hat / (vR_hat.sqrt() + eps)

        if getattr(self, "U", None) is not None and self.U.requires_grad:
            dU = torch.linalg.solve(B_inv, dL)
            self.U.data.add_(dU.to(self.U.dtype))
        if getattr(self, "V", None) is not None and self.V.requires_grad:
            dV = torch.linalg.solve(A_inv, dR)
            self.V.data.add_(dV.to(self.V.dtype))

        if gS32 is not None and getattr(self, "S", None) is not None and self.S.requires_grad:
            vS_eff = self.vhS_max if self.config.ff_amsgrad else self.vS
            mS_hat = self.mS / bias1
            vS_hat = vS_eff / bias2
            dS = -lr * mS_hat / (vS_hat.sqrt() + eps)
            self.S.data.add_(dS.to(self.S.dtype))
            eff_vals.append((mS_hat.abs() / (vS_hat.sqrt() + eps)).mean())

        if getattr(self, "U", None) is not None and self.U.grad is not None:
            self.U.grad = None
        if getattr(self, "V", None) is not None and self.V.grad is not None:
            self.V.grad = None
        if getattr(self, "S", None) is not None and self.S.grad is not None:
            self.S.grad = None

        finite_vals = [val for val in eff_vals if torch.isfinite(val)]
        if not finite_vals:
            return None
        eff = torch.stack(finite_vals).mean()
        return float(eff.item()) if torch.isfinite(eff) else None

    @torch.no_grad()
    def fisher_eff_scale(self) -> Optional[float]:
        if self.rank == 0 or not self.config.use_fisher_frame_optim:
            return None
        eps = self.config.ff_eps
        vals = []
        if self.vL.numel():
            vals.append((self.mL.abs() / (self.vL.sqrt() + eps)).mean())
        if self.vR.numel():
            vals.append((self.mR.abs() / (self.vR.sqrt() + eps)).mean())
        if self.vS.numel():
            vals.append((self.mS.abs() / (self.vS.sqrt() + eps)).mean())
        finite_vals = [val for val in vals if torch.isfinite(val)]
        if not finite_vals:
            return None
        eff = torch.stack(finite_vals).mean()
        return float(eff.item()) if torch.isfinite(eff) else None

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

    def _current_ema_decay(self, *, step: Optional[int] = None) -> float:
        cfg = self.config
        start = cfg.ema_decay_start
        steps = cfg.ema_decay_anneal_steps
        end = cfg.ema_decay
        if start is None or steps is None or steps <= 0:
            return float(end)
        if step is None:
            step = int(self.step_count.item())
        if step <= 0:
            return float(start)
        if step >= steps:
            return float(end)
        alpha = step / steps
        value = start + (end - start) * alpha
        return float(value)

    def _current_update_interval(self, *, step: Optional[int] = None) -> int:
        cfg = self.config
        start = cfg.update_interval_start
        steps = cfg.update_interval_anneal_steps
        end = cfg.update_interval
        if start is None or steps is None or steps <= 0:
            return int(end)
        if step is None:
            step = int(self.step_count.item())
        if step <= 0:
            return int(start)
        if step >= steps:
            return int(end)
        alpha = step / steps
        value = start + (end - start) * alpha
        value_int = int(round(value))
        value_int = max(1, value_int)
        return int(min(end, value_int))

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
            pre_step = int(self.step_count.item())
            decay = self._current_ema_decay(step=pre_step)
            self.A_ema.mul_(decay).add_(a_factor, alpha=1.0 - decay)
            self.B_ema.mul_(decay).add_(g_factor, alpha=1.0 - decay)
            self.step_count.add_(1)
            step = int(self.step_count.item())
            current_interval = self._current_update_interval(step=step)

            def _compute_and_cache_whiteness() -> tuple[Tensor, Tensor]:
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
                a_damped_local = self.A_ema + self.config.damping * eye_in
                b_damped_local = self.B_ema + self.config.damping * eye_out
                e_a_local, e_b_local = self._compute_whiteness_errors(
                    a_damped_local, b_damped_local, eye_in, eye_out
                )
                self._last_eA.copy_(e_a_local)
                self._last_eB.copy_(e_b_local)
                return e_a_local, e_b_local

            whiteness_recent = False
            e_a = self._last_eA
            e_b = self._last_eB

            # ---- D3: energy capture every diagnostics_interval steps ----
            self._diag_step_mod.add_(1)
            diag_interval = int(getattr(self.config, "diagnostics_interval", 256))
            if diag_interval > 0 and int(self._diag_step_mod.item()) % diag_interval == 0:
                try:
                    self._last_energy_capture = self._energy_capture_from_batch(
                        a.to(torch.float32),
                        g.to(torch.float32),
                    )
                except Exception:
                    pass
            # -------------------------------------------------------------

            if self.config.track_fisher:
                log_interval = self.config.whiteness_log_interval or current_interval
                if step == 1 or (log_interval and step % log_interval == 0):
                    e_a, e_b = _compute_and_cache_whiteness()
                    whiteness_recent = True
                    self._log_whiteness_metrics(float(e_a), float(e_b), phase="pre_refresh")
            #if step % current_interval == 0:
                # Defer refresh to next forward pass
                #should_refresh = (float(e_a) > 0.025) or (float(e_b) > 0.02)  # RMS thresholds
                #should_refresh = True
                #if should_refresh:
                 #   self.refresh_pending.fill_(True)
                #else:
                 #   print(f"step {step} eA={float(e_a)} eB={float(e_b)}")
                 #   print("Should refresh: False")
            # how many local steps since last refresh for THIS layer
            elapsed = int(self.step_count.item()) - int(self.last_refresh_step.item())
            interval = self._current_update_interval(step=step)

            # decide whether we WANT to refresh this layer now
            # (fix the threshold if you meant 0.25 rather than 0.025)
            if elapsed >= interval:
                if not whiteness_recent:
                    e_a, e_b = _compute_and_cache_whiteness()
                should_refresh = (float(e_a) > 0.05) or (float(e_b) > 0.005)
                if should_refresh:
                    self.refresh_pending.fill_(True)
                    self.last_refresh_step.copy_(self.step_count)  # update timestamp
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(
                            "Fisher-LoRA schedule: REFRESH layer=%s step=%d elapsed=%d interval=%d eA=%.3e eB=%.3e",
                            self._fisher_lora_name or "<?>",
                            step, elapsed, interval, float(e_a), float(e_b)
                        )
                else:
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(
                            "Fisher-LoRA schedule: SKIP layer=%s step=%d elapsed=%d interval=%d eA=%.3e eB=%.3e",
                            self._fisher_lora_name or "<?>",
                            step, elapsed, interval, float(e_a), float(e_b)
                        )

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

        diff_a = self.A_inv_sqrt @ a_matrix @ self.A_inv_sqrt - eye_in
        diff_b = self.B_inv_sqrt @ b_matrix @ self.B_inv_sqrt - eye_out

        e_a = torch.linalg.norm(diff_a, ord='fro') / (self.in_features ** 0.5)
        e_b = torch.linalg.norm(diff_b, ord='fro') / (self.out_features ** 0.5)

        return e_a, e_b

    def _log_whiteness_metrics(self, e_a: float, e_b: float, *, phase: str) -> None:
        try:
            ea_tensor = torch.as_tensor(
                e_a,
                dtype=self._last_eA.dtype,
                device=self._last_eA.device,
            )
            eb_tensor = torch.as_tensor(
                e_b,
                dtype=self._last_eB.dtype,
                device=self._last_eB.device,
            )
            self._last_eA.copy_(ea_tensor)
            self._last_eB.copy_(eb_tensor)
        except Exception:
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
