"""
models/restriction_maps_qr.py — QR-Reparametrised Restriction Maps
====================================================================
Replaces the Householder parametrization in restriction_maps.py with
a direct unconstrained weight matrix + QR projection.

Why the change was necessary
-----------------------------
The Householder parametrization initialised with nn.init.orthogonal_
places every map at F ≈ [I_k | 0] — the identity frame.  Near this
point, many Householder parameter directions produce zero net change
in F (a degenerate symmetry), so gradients for those directions
are numerically zero.  After 65 epochs, all 84 rotation angles were
< 0.001 radians — the maps had not moved.

The QR reparametrisation stores a raw unconstrained weight matrix
W ∈ R^{d_v × k_eff} and computes F = QR(W)^T at each forward pass.
Gradients flow freely through torch.linalg.qr (supported since
PyTorch 1.9), no symmetry issue exists, and the maps can learn
arbitrary rotations.

Initialisation: W ~ N(0, 1/sqrt(d_v))  — non-degenerate start,
maps begin away from identity, non-zero rotation angles from epoch 0.

Document version: v1.0  (23 March 2026)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
from torch import nn

from config import (
    MODALITY_ORDER, D_V, DIRECTED_EDGES,
    get_k_eff, get_all_k_eff,
)

MapDict = Dict[Tuple[str, str, str], torch.Tensor]


class QRRestrictionMap(nn.Module):
    """
    Single restriction map F_{v←e} ∈ R^{k_eff × d_v} via QR projection.

    Stores unconstrained weight W ∈ R^{d_v × k_eff}.
    Forward produces F = QR(W)^T with F F^T = I_{k_eff}.

    Args:
        d_v:   Stalk dimension of the modality node.
        k_eff: Effective agreement space dimension (must satisfy k_eff < d_v).
    """

    def __init__(self, d_v: int, k_eff: int):
        super().__init__()
        if k_eff >= d_v:
            raise ValueError(
                f"QRRestrictionMap requires k_eff < d_v. "
                f"Got k_eff={k_eff}, d_v={d_v}."
            )
        self.d_v   = d_v
        self.k_eff = k_eff

        # Unconstrained weight matrix [d_v, k_eff]
        # Init: N(0, 1/sqrt(d_v)) — non-degenerate, maps away from identity
        self.weight = nn.Parameter(
            torch.randn(d_v, k_eff) / math.sqrt(d_v)
        )

    def forward(self) -> torch.Tensor:
        """
        Compute restriction map via QR decomposition.

        Returns:
            F: [k_eff, d_v] with F F^T = I_{k_eff}.
        """
        # QR: weight [d_v, k_eff] → Q [d_v, k_eff], Q^T Q = I_{k_eff}
        Q, _ = torch.linalg.qr(self.weight, mode='reduced')
        return Q.T   # [k_eff, d_v]

    def rotation_angle(self) -> float:
        """
        Effective rotation angle from the canonical projection P₀ = [I_{k_eff} | 0].

            θ = arccos( trace(F · P₀ᵀ) / k_eff )
              = arccos( trace(F[:, :k_eff]) / k_eff )   ∈ [0, π/2]

        Correction note (v1.1):
          The original formula arccos(trace(F F^T)/k_eff) is always 0 because
          F F^T = I_{k_eff} by orthonormality — it measures whether F is
          orthonormal (trivially true), not how much it has rotated.
          The correct formula compares F against the canonical projection P₀,
          so that θ = 0 when F = P₀ and θ > 0 for any non-trivial rotation.
        """
        with torch.no_grad():
            F       = self.forward()                          # [k_eff, d_v]
            # trace(F · P₀ᵀ) = sum of diagonal of F[:, :k_eff]
            overlap = F[:, :self.k_eff].diagonal().sum()     # scalar
            cos_θ   = (overlap / self.k_eff).clamp(-1.0, 1.0)
            return math.acos(cos_θ.item())

    def orthogonality_error(self) -> float:
        """|| F F^T - I_{k_eff} ||_F — should be < 1e-5."""
        with torch.no_grad():
            F   = self.forward()
            I_k = torch.eye(self.k_eff, device=F.device, dtype=F.dtype)
            return (F @ F.T - I_k).norm(p='fro').item()

    def extra_repr(self) -> str:
        return f"d_v={self.d_v}, k_eff={self.k_eff}"


class ModalityRestrictionMaps(nn.Module):
    """
    Global shared restriction map parameters for Project [3].

    Contains 84 = 42 × 2 QRRestrictionMap modules as nn.ModuleDict.
    Each module stores one unconstrained weight matrix and produces
    its restriction map on demand via QR decomposition.

    Replaces the Householder-based ModalityRestrictionMaps in
    restriction_maps.py.  The public API is identical so that
    downstream code (laplacian_hetero.py, sheaf_modality.py,
    outputs/writer.py) requires no changes.

    Args:
        k (int): Global agreement space dimension.
    """

    def __init__(self, k: int):
        super().__init__()

        self.k = k
        self.k_eff_per_mod: Dict[str, int] = get_all_k_eff(k)

        # Build one QRRestrictionMap per (edge, role) pair: 84 total
        self.maps = nn.ModuleDict()

        for (u, v) in DIRECTED_EDGES:
            k_eff_v = self.k_eff_per_mod[v]
            k_eff_u = self.k_eff_per_mod[u]
            d_v     = D_V[v]
            d_u     = D_V[u]

            # Head map: F_{v←e}, projects stalk of v into agreement space
            head_key = f"F_head_{v}__edge_{u}_{v}"
            self.maps[head_key] = QRRestrictionMap(d_v=d_v, k_eff=k_eff_v)

            # Tail map: F_{u←e}, projects stalk of u into agreement space
            tail_key = f"F_tail_{u}__edge_{u}_{v}"
            self.maps[tail_key] = QRRestrictionMap(d_v=d_u, k_eff=k_eff_u)

    # ── Map access ────────────────────────────────────────────

    def get_map(
        self,
        node: str,
        edge: Tuple[str, str],
        role: str,
    ) -> torch.Tensor:
        """
        Produce restriction map F_{node←e} for directed edge e = (u → v).

        Args:
            node:  The modality node.
            edge:  (source, target) directed edge tuple.
            role:  'head' if node is the target, 'tail' if source.

        Returns:
            F: [k_eff(node), d_v(node)] with F F^T = I_k.
        """
        u, v = edge
        if role == 'head':
            key = f"F_head_{v}__edge_{u}_{v}"
        elif role == 'tail':
            key = f"F_tail_{u}__edge_{u}_{v}"
        else:
            raise ValueError(f"role must be 'head' or 'tail', got '{role}'")
        return self.maps[key]()

    def get_all_maps(self) -> MapDict:
        """
        Produce all 84 restriction maps.

        Returns:
            Dict keyed by (node, source, target) → [k_eff(node), d_v(node)].
            Compatible with the API expected by laplacian_hetero.py and
            sheaf_modality.py.
        """
        result: MapDict = {}
        for (u, v) in DIRECTED_EDGES:
            result[(v, u, v)] = self.get_map(v, (u, v), role='head')
            result[(u, u, v)] = self.get_map(u, (u, v), role='tail')
        return result

    # ── Diagnostics ───────────────────────────────────────────

    def get_rotation_angles(self) -> Dict[Tuple[str, str, str], float]:
        """Rotation angles for all 84 maps."""
        angles = {}
        for (u, v) in DIRECTED_EDGES:
            angles[(v, u, v)] = self.maps[f"F_head_{v}__edge_{u}_{v}"].rotation_angle()
            angles[(u, u, v)] = self.maps[f"F_tail_{u}__edge_{u}_{v}"].rotation_angle()
        return angles

    def get_orthogonality_errors(self) -> Dict[Tuple[str, str, str], float]:
        """Orthogonality errors ||F F^T - I||_F for all 84 maps."""
        errors = {}
        for (u, v) in DIRECTED_EDGES:
            errors[(v, u, v)] = self.maps[f"F_head_{v}__edge_{u}_{v}"].orthogonality_error()
            errors[(u, u, v)] = self.maps[f"F_tail_{u}__edge_{u}_{v}"].orthogonality_error()
        return errors

    def parameter_summary(self) -> str:
        """Human-readable breakdown of parameter counts."""
        lines = [
            "─" * 64,
            f"ModalityRestrictionMaps (QR)  k={self.k}",
            "─" * 64,
            f"  {'Modality':<6}  {'d_v':>4}  {'k_eff':>5}  "
            f"{'params/map':>10}  {'n_maps':>6}  {'total':>8}",
            f"  {'─'*6}  {'─'*4}  {'─'*5}  {'─'*10}  {'─'*6}  {'─'*8}",
        ]
        grand_total = 0
        for mod in MODALITY_ORDER:
            d_v    = D_V[mod]
            k_eff  = self.k_eff_per_mod[mod]
            n_maps = sum(1 for (u, v) in DIRECTED_EDGES if v == mod) + \
                     sum(1 for (u, v) in DIRECTED_EDGES if u == mod)
            params_each = d_v * k_eff
            total_mod   = n_maps * params_each
            grand_total += total_mod
            lines.append(
                f"  {mod:<6}  {d_v:>4}  {k_eff:>5}  "
                f"{params_each:>10}  {n_maps:>6}  {total_mod:>8}"
            )
        lines += [
            f"  {'─'*6}  {'─'*4}  {'─'*5}  {'─'*10}  {'─'*6}  {'─'*8}",
            f"  {'TOTAL':<6}  {'':>4}  {'':>5}  {'':>10}  {'84':>6}  "
            f"{grand_total:>8}",
            "─" * 64,
        ]
        return "\n".join(lines)

    def grouped_parameters(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """All parameters are map parameters — no 'other' group."""
        return list(self.parameters()), []

    def extra_repr(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        return f"k={self.k}, n_maps=84, total_params={total}, backend=QR"
