"""DinoV2 model factory that integrates Fisher-LoRA adapters."""

from __future__ import annotations

import fnmatch
import logging
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from fisher_lora import FisherLoRAConfig, attach_fisher_lora

from .dinov2_lora import DinoV2WithClassificationHead


logger = logging.getLogger(__name__)


_DTYPE_ALIASES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def _to_dtype(value: object) -> torch.dtype:
    """Convert configuration values into a torch dtype."""

    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        key = value.lower()
        if key not in _DTYPE_ALIASES:
            raise ValueError(f"Unsupported factor dtype string: {value}")
        return _DTYPE_ALIASES[key]
    raise TypeError(f"factor_dtype must be a torch.dtype or str, got {type(value)}")


def _normalize_targets(targets: Optional[Iterable[str]]) -> Optional[List[str]]:
    if targets is None:
        return None
    if hasattr(targets, "to_container"):
        targets = targets.to_container()
    if isinstance(targets, str):
        raise TypeError("target_modules must be an iterable of strings, not a single string")
    return list(targets)


def _expand_target_patterns(root: nn.Module, patterns: Optional[Sequence[str]]) -> Optional[List[str]]:
    if patterns is None:
        return None

    linear_names = [name for name, module in root.named_modules() if isinstance(module, nn.Linear)]
    resolved: List[str] = []
    for pattern in patterns:
        matches = [name for name in linear_names if fnmatch.fnmatch(name, pattern)]
        if not matches:
            raise ValueError(f"Target pattern '{pattern}' did not match any Linear submodules")
        resolved.extend(matches)
    # Remove duplicates while preserving order
    seen = set()
    unique_resolved: List[str] = []
    for name in resolved:
        if name not in seen:
            unique_resolved.append(name)
            seen.add(name)
    return unique_resolved


def _build_fisher_config(rank: int, fisher_cfg: dict) -> FisherLoRAConfig:
    cfg = dict(fisher_cfg)
    cfg.setdefault("track_fisher", True)
    cfg.setdefault("freeze_base", True)
    cfg["rank"] = rank
    cfg["factor_dtype"] = _to_dtype(cfg.get("factor_dtype", torch.float32))
    return FisherLoRAConfig(**cfg)


def create_dinov2_fisher_lora_model(
    *,
    model_name: str,
    num_classes: int,
    dropout: float,
    fisher_rank: int,
    fisher_config: dict,
    target_patterns: Optional[Iterable[str]] = None,
) -> nn.Module:
    """Create a DinoV2 classification model instrumented with Fisher-LoRA.

    Args:
        model_name: HuggingFace model identifier.
        num_classes: Number of output classes.
        dropout: Dropout probability for the classifier head.
        fisher_rank: Rank of the Fisher-LoRA adapters.
        fisher_config: Additional keyword arguments for :class:`FisherLoRAConfig`.
        target_patterns: Optional patterns describing which Linear layers to adapt.

    Returns:
        ``nn.Module`` ready for fine-tuning with Fisher-LoRA adapters.
    """

    logger.info("Creating DinoV2 Fisher-LoRA model with backbone %s", model_name)

    backbone = DinoV2WithClassificationHead(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
        freeze_backbone=True,
    )

    normalized_patterns = _normalize_targets(target_patterns)
    resolved_targets = _expand_target_patterns(backbone, normalized_patterns)
    fisher_cfg = _build_fisher_config(fisher_rank, fisher_config)

    replaced = attach_fisher_lora(backbone, target_modules=resolved_targets, config=fisher_cfg)
    backbone.fisher_lora_modules = replaced

    logger.info("Attached Fisher-LoRA modules to %d Linear layers", len(replaced))
    logger.debug("Fisher-LoRA layers: %s", sorted(replaced.keys()))

    return backbone

