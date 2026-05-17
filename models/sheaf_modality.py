"""
models/sheaf_modality.py — Main Sheaf Modality Model
=====================================================
Integrates all prior components into the full forward pass:

  L0: z-score normalisation (applied before DataLoader; model receives
      pre-normalised tensors — no normalisation layer needed here)
  L1: Stalk assembly — slice [B, 107] into 7 per-modality tensors
  L2: Augmented sheaf diffusion — 2 rounds over the modality graph
  L3: Consistency energy from INITIAL (pre-diffusion) stalks
  L4: Classification head — post-diffusion stalks → 49-class softmax

Architecture decisions (see §5 of methodology document)
---------------------------------------------------------
1.  Restriction maps are GLOBAL parameters (not feature-conditioned).
    All 2,450 nuclei share the same fixed modality graph.

    Two backends are supported via the optional restriction_maps arg:
      - QR parametric (default): maps are nn.Parameters trained by SGD.
      - CCA analytical:          maps are frozen register_buffers
                                  derived from training cross-covariance.

2.  E_i is computed from PRE-DIFFUSION stalks x_v^(0), not x_v^(2).
    This measures intrinsic inter-modality coherence of the raw damage
    pattern, not the smoothed post-diffusion representation.

3.  W1^(t) is a k × k matrix acting in the AGREEMENT SPACE.
    Each directed edge e contributes a k_min-dimensional discrepancy;
    W1 is applied after zero-padding to the full k dimension (or by
    truncating k to k_min, whichever is smaller per edge).

4.  W1 structure options:
      'identity'  — no learnable mixing (ablation baseline)
      'diagonal'  — learn per-dimension scaling in agreement space
      'full'      — learn arbitrary k × k linear transformation (default)

5.  The epsilon residual connection from Bodnar et al. Eq. (55) is an
    optional ablation.  It is OFF by default.

Forward pass shapes (k=24, B=batch_size)
-----------------------------------------
  Input X:              [B, 107]       pre-normalised feature matrix
  Stalk x_v^(0):        [B, d_v]       initial stalk per modality
  Discrepancy δ_e^(t):  [B, k_min(e)]  per-edge, per-round
  W1^(t) δ:             [B, k]         after padding/truncating to k
  Update Δ_v^(t):       [B, d_v]       aggregated at head node
  x_v^(2):              [B, d_v]       post-diffusion stalk
  x_cat^(2):            [B, 107]       concatenated post-diffusion
  E_i:                  [B]            per-nucleus consistency energy
  e_ei:                 [42, B]        per-edge energy decomposition
  logits:               [B, 49]        condition class logits

Revision note (v1.1 — 24 March 2026):
  Added optional `restriction_maps` constructor argument to support
  CCA-derived analytical maps (restriction_maps_cca.py) alongside the
  original QR parametric maps (restriction_maps_qr.py).  All downstream
  code is unchanged — the public API is identical for both backends.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    MODALITY_ORDER, D_V, STALK_SLICES, DIRECTED_EDGES,
    TOTAL_FEATURES, get_k_eff, N_CONDITIONS,
)
# Default backend: QR parametric maps (used when restriction_maps=None)
from models.restriction_maps_qr import ModalityRestrictionMaps as _DefaultMaps
from models.restriction_maps_qr import MapDict
from models.laplacian_hetero import (
    compute_consistency_energy, edge_k_min,
)


# ─────────────────────────────────────────────────────────────
# W1 weight matrix builder — fully dynamic, no hardcoded dims
# ─────────────────────────────────────────────────────────────

def _build_w1(k: int, structure: str) -> nn.Module:
    """
    Build the per-round agreement-space weight matrix W1^(t).

    All dims derived from k — no hardcoded values.

    Args:
        k:         Global agreement space dimension.
        structure: 'identity' | 'diagonal' | 'full'

    Returns:
        nn.Module whose forward(x) applies the weight transform
        to a [..., k] tensor.
    """
    if structure == 'identity':
        return _IdentityW1()
    elif structure == 'diagonal':
        return _DiagonalW1(k=k)
    elif structure == 'full':
        return _FullW1(k=k)
    else:
        raise ValueError(
            f"w1_structure must be 'identity', 'diagonal', or 'full', "
            f"got '{structure}'"
        )


class _IdentityW1(nn.Module):
    """Identity: W1 x = x — no parameters, no transform."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _DiagonalW1(nn.Module):
    """Diagonal W1: element-wise scaling in agreement space."""
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.scale = nn.Parameter(torch.ones(k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class _FullW1(nn.Module):
    """Full k × k linear transformation in agreement space."""
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.linear = nn.Linear(k, k, bias=False)
        nn.init.eye_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ─────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────

class ModalitySheafModel(nn.Module):
    """
    Sheaf neural network over the fixed 7-node modality graph.

    Each forward pass processes a batch of nuclei (B independent
    section evaluations of the fixed modality graph).

    Args:
        k:                Global agreement space dimension (pre k_eff capping).
        n_classes:        Number of condition classes. Default: 49.
        n_diffusion_rounds: Number of sheaf diffusion rounds. Default: 2.
        w1_structure:     Weight matrix structure: 'identity'|'diagonal'|'full'.
        mlp_hidden:       Hidden dimension of classification head. Default: 64.
        alpha:            Diffusion step size. Default: 1.0.
        use_epsilon_residual: Enable Bodnar Eq.(55) epsilon residual. Default: False.
        restriction_maps: Optional pre-constructed restriction map module.
                          If None (default), creates QR parametric maps.
                          Pass a ModalityRestrictionMaps instance from
                          restriction_maps_cca.py for the analytical CCA model.
    """

    def __init__(
        self,
        k: int = 24,
        n_classes: int = N_CONDITIONS,
        n_diffusion_rounds: int = 2,
        w1_structure: str = 'full',
        mlp_hidden: int = 64,
        alpha: float = 1.0,
        use_epsilon_residual: bool = False,
        restriction_maps: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.k                    = k
        self.n_classes            = n_classes
        self.n_rounds             = n_diffusion_rounds
        self.w1_structure         = w1_structure
        self.alpha                = alpha
        self.use_epsilon_residual = use_epsilon_residual

        self.k_eff: Dict[str, int] = {
            mod: get_k_eff(k, mod) for mod in MODALITY_ORDER
        }

        # ── Restriction maps ────────────────────────────────────
        # Accept an externally-constructed module (CCA analytical)
        # or build the default QR parametric version.
        if restriction_maps is not None:
            self.restriction_maps = restriction_maps
        else:
            self.restriction_maps = _DefaultMaps(k=k)

        # ── Per-round W1 matrices ────────────────────────────────
        self.W1 = nn.ModuleList([
            _build_w1(k=k, structure=w1_structure)
            for _ in range(n_diffusion_rounds)
        ])

        # ── Optional epsilon residual (Bodnar Eq. 55) ───────────
        if use_epsilon_residual:
            self.epsilons = nn.ParameterList([
                nn.ParameterDict({
                    mod: nn.Parameter(torch.zeros(1))
                    for mod in MODALITY_ORDER
                })
                for _ in range(n_diffusion_rounds)
            ])

        # ── Classification head ──────────────────────────────────
        total_d = TOTAL_FEATURES
        self.classifier = nn.Sequential(
            nn.Linear(total_d, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, n_classes),
        )

    # ── Forward pass ──────────────────────────────────────────

    def forward(
        self,
        X: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Args:
            X: [B, 107] pre-normalised feature batch.

        Returns:
            logits: [B, n_classes]  condition class logits
            E_i:    [B]             per-nucleus consistency energy (from x^(0))
            e_ei:   [42, B]         per-edge energy decomposition (from x^(0))
        """
        B = X.shape[0]

        # ── L1: Stalk assembly ────────────────────────────────
        x0: Dict[str, torch.Tensor] = {}
        for mod in MODALITY_ORDER:
            sv, ev = STALK_SLICES[mod]
            x0[mod] = X[:, sv:ev]

        # ── Get all 84 restriction maps ────────────────────────
        maps: MapDict = self.restriction_maps.get_all_maps()

        # ── L2: Augmented sheaf diffusion ─────────────────────
        x_t: Dict[str, torch.Tensor] = {
            mod: x0[mod].clone() for mod in MODALITY_ORDER
        }

        for t in range(self.n_rounds):
            W1_t = self.W1[t]
            updates: Dict[str, torch.Tensor] = {
                mod: torch.zeros_like(x_t[mod]) for mod in MODALITY_ORDER
            }

            for (u, v) in DIRECTED_EDGES:
                F_v = maps[(v, u, v)]
                F_u = maps[(u, u, v)]
                k_min = edge_k_min(u, v, self.k)

                proj_v = x_t[v] @ F_v[:k_min].T
                proj_u = x_t[u] @ F_u[:k_min].T
                delta_e = proj_v - proj_u

                if k_min < self.k:
                    pad_size = self.k - k_min
                    delta_padded = F.pad(delta_e, (0, pad_size))
                    weighted_full = W1_t(delta_padded)
                    weighted = weighted_full[:, :k_min]
                else:
                    weighted = W1_t(delta_e)

                updates[v] = updates[v] + weighted @ F_v[:k_min]

            for mod in MODALITY_ORDER:
                new_x = x_t[mod] - self.alpha * updates[mod]

                if self.use_epsilon_residual:
                    eps = self.epsilons[t][mod]
                    coeff = 1.0 + torch.tanh(eps)
                    new_x = coeff * x0[mod] - new_x

                x_t[mod] = F.elu(new_x)

        # ── L3: Consistency energy from INITIAL stalks ────────
        X_initial = torch.cat(
            [x0[mod] for mod in MODALITY_ORDER], dim=1
        )

        E_i, e_ei = compute_consistency_energy(
            X_initial, maps, self.k, return_per_edge=True
        )

        # ── L4: Classification head ───────────────────────────
        x_post = torch.cat(
            [x_t[mod] for mod in MODALITY_ORDER], dim=1
        )
        logits = self.classifier(x_post)

        return logits, E_i, e_ei

    # ── Parameter grouping ────────────────────────────────────

    def grouped_parameters(
        self,
    ) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """
        Separate restriction map parameters from all other parameters.

        For QR parametric maps: map_params = Householder weight matrices.
        For CCA analytical maps: map_params = [] (buffers, not parameters).

        Returns:
            (map_params, other_params)
        """
        map_param_ids = {id(p) for p in self.restriction_maps.parameters()}

        map_params   = list(self.restriction_maps.parameters())
        other_params = [
            p for p in self.parameters()
            if id(p) not in map_param_ids
        ]
        return map_params, other_params

    # ── Diagnostics ───────────────────────────────────────────

    def parameter_summary(self) -> str:
        """Human-readable breakdown of parameter counts."""
        map_params, other_params = self.grouped_parameters()

        n_map   = sum(p.numel() for p in map_params)
        n_w1    = sum(p.numel() for p in self.W1.parameters())
        n_clf   = sum(p.numel() for p in self.classifier.parameters())
        n_eps   = (
            sum(p.numel() for p in self.epsilons.parameters())
            if self.use_epsilon_residual else 0
        )
        n_total = sum(p.numel() for p in self.parameters())

        lines = [
            "─" * 52,
            f"ModalitySheafModel  k={self.k}  rounds={self.n_rounds}",
            f"  w1_structure={self.w1_structure}  "
            f"epsilon_residual={self.use_epsilon_residual}",
            "─" * 52,
            f"  {'restriction_maps':<24} {n_map:>8,}",
            f"  {'W1 matrices':<24} {n_w1:>8,}",
            f"  {'classifier (MLP)':<24} {n_clf:>8,}",
        ]
        if self.use_epsilon_residual:
            lines.append(f"  {'epsilon_residuals':<24} {n_eps:>8,}")
        lines += [
            "─" * 52,
            f"  {'TOTAL':<24} {n_total:>8,}",
            "─" * 52,
        ]
        return "\n".join(lines)

    def extra_repr(self) -> str:
        backend = type(self.restriction_maps).__module__.split('.')[-1]
        return (
            f"k={self.k}, n_classes={self.n_classes}, "
            f"n_rounds={self.n_rounds}, "
            f"w1={self.w1_structure}, "
            f"alpha={self.alpha}, "
            f"maps_backend={backend}"
        )
