"""Optional NineToothed kernels for the MiniMax-H3 audio VAE."""

from __future__ import annotations

import types

import torch
from torch.nn.utils import remove_weight_norm

from infinicore.nn.functional import snake_beta_ninetoothed


def materialize_audio_weight_norm(audio_vae: torch.nn.Module) -> int:
    """Freeze legacy weight-norm hooks into ordinary inference weights."""
    materialized = 0
    for module in audio_vae.modules():
        if not (hasattr(module, "weight_g") and hasattr(module, "weight_v")):
            continue
        remove_weight_norm(module)
        materialized += 1
    return materialized


def _snake_beta_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    if self.training or torch.is_grad_enabled():
        return self._infinilm_snake_beta_original(hidden_states)

    alpha = self.alpha
    beta = self.beta
    key = (
        alpha.data_ptr(),
        alpha._version,
        beta.data_ptr(),
        beta._version,
    )
    if getattr(self, "_infinilm_snake_beta_key", None) != key:
        object.__setattr__(self, "_infinilm_snake_beta_alpha_exp", alpha.detach().exp())
        object.__setattr__(self, "_infinilm_snake_beta_beta_exp", beta.detach().exp())
        object.__setattr__(self, "_infinilm_snake_beta_key", key)

    return snake_beta_ninetoothed(
        hidden_states,
        self._infinilm_snake_beta_alpha_exp,
        self._infinilm_snake_beta_beta_exp,
    )


def install_ninetoothed_audio_kernels(audio_vae: torch.nn.Module) -> int:
    """Replace BigVGAN SnakeBeta forwards while preserving checkpoint state."""
    installed = 0
    for module in audio_vae.modules():
        if module.__class__.__name__ != "MiniMaxH3AudioSnakeBeta":
            continue
        if getattr(module, "_infinilm_snake_beta_nt", False):
            continue
        if module.training:
            raise RuntimeError("Install NineToothed audio kernels after calling eval()")
        object.__setattr__(
            module,
            "_infinilm_snake_beta_original",
            module.forward,
        )
        object.__setattr__(
            module,
            "forward",
            types.MethodType(_snake_beta_forward, module),
        )
        object.__setattr__(module, "_infinilm_snake_beta_nt", True)
        installed += 1
    return installed
