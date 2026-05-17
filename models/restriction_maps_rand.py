"""
models/restriction_maps_rand.py — Haar-Random Fixed Restriction Maps
=====================================================================
Generates restriction maps by drawing from the Haar measure on the Stiefel
manifold V_{k_edge}(R^{d_v}) — the uniform distribution over all matrices
with k_edge orthonormal rows in R^{d_v}.

Scientific role
---------------
This is the Phase 2 null baseline for Project [3]. It tests whether the
structure observed in the CCA E_i landscape (ρ(|r|, E_i) = −0.679, p=0.008)
is a genuine geometric property of CCA maps or merely a consequence of using
*any* set of frozen orthonormal maps.

Null hypothesis: frozen Haar-random maps will produce:
  (a) ρ(|r|, E_i) ≈ 0 — no correlation with [2]'s covariance structure
  (b) A structureless E_i landscape — no LET or O₂ gradient
  (c) Lower classification accuracy than CCA (less discriminative geometry)

If all three hold, CCA geometry is validated as non-trivial.

Mathematical construction
-------------------------
For each directed edge (u → v):
  k_edge = min(k_eff(u), k_eff(v))     [same k_min logic as all other maps]

  For the head map F_{v←e} ∈ R^{k_edge × d_v}:
    G ~ N(0, I)^{d_v × k_edge}          [Gaussian draw]
    Q, R = QR(G)                         [thin QR, Q ∈ R^{d_v × k_edge}]
    Q ← Q · diag(sign(diag(R)))          [sign correction → Haar measure]
    F_{v←e} = Q^T                        [k_edge orthonormal rows in R^{d_v}]

  F_{u←e} constructed identically with d_u.

The sign correction is essential for the Haar measure (Stewart 1980). Without
it, QR(G) would produce a non-uniform distribution over the Stiefel manifold
because the R diagonal signs are not consistently positive.

Reproducibility: all maps are deterministically generated from map_seed.
map_seed = training_seed + 1000 by convention, so map randomness is
decoupled from training randomness (data split, weight initialisation).

Data independence: unlike CCA maps, RandEdge maps require NO training data.
The constructor signature takes only (k, map_seed) — X_train is not accepted.
This is the defining property of the null baseline.

Public API is identical to restriction_maps_cca.ModalityRestrictionMaps and
restriction_maps_qr.ModalityRestrictionMaps, so all downstream code (laplacian_
hetero.py, sheaf_modality.py, outputs/writer.py) requires no changes.

Document version: v1.0  (25 March 2026)
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

# Type alias matching restriction_maps_qr.py
MapDict = Dict[Tuple[str, str, str], torch.Tensor]


# ─────────────────────────────────────────────────────────────────────────────
# Haar-random Stiefel draw
# ─────────────────────────────────────────────────────────────────────────────

def _haar_stiefel(k: int, d: int, rng: torch.Generator) -> torch.Tensor:
    """
    Draw a Haar-uniform random matrix F ∈ R^{k × d} with orthonormal rows.
    (i.e. a uniform sample from the Stiefel manifold V_k(R^d))

    Args:
        k:   Number of orthonormal rows (agreement space dimension).
        d:   Ambient dimension (stalk dimension d_v).
        rng: Seeded torch.Generator for reproducibility.

    Returns:
        F: [k, d] with F @ F.T = I_k (orthonormal rows).

    Raises:
        ValueError: if k > d (cannot have more orthonormal rows than columns).
    """
    if k > d:
        raise ValueError(
            f"Cannot draw {k} orthonormal rows in R^{d}: k must be ≤ d."
        )

    # Step 1: Draw a d × k Gaussian matrix
    G = torch.randn(d, k, generator=rng)   # [d, k]

    # Step 2: Thin QR decomposition  →  Q ∈ R^{d × k}, R ∈ R^{k × k}
    Q, R = torch.linalg.qr(G, mode='reduced')

    # Step 3: Diagonal sign correction for exact Haar measure (Stewart 1980)
    #   Without this, P(Q) ∝ Haar but not exactly Haar because QR is only
    #   unique up to sign flips of each column of Q.
    signs = torch.sign(torch.diagonal(R))   # [k]
    signs[signs == 0] = 1.0                 # treat exact-zero as positive
    Q = Q * signs.unsqueeze(0)              # [d, k] — rescaled columns

    # Step 4: Transpose → [k, d] with orthonormal rows
    F = Q.T.contiguous()
    return F


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class ModalityRestrictionMaps(nn.Module):
    """
    Haar-random fixed restriction maps for Project [3] Phase 2 null baseline.

    Maps are drawn once from the Stiefel manifold at construction time and
    stored as register_buffer tensors — device-tracked, not gradient-tracked,
    and frozen throughout training.

    Unlike CCA maps, no training data is required. The constructor takes only
    (k, map_seed). Downstream code that calls get_all_maps() or get_map() is
    fully compatible with this class without modification.

    Args:
        k:        Global agreement space dimension.
        map_seed: Integer seed for Haar-random map generation. Convention:
                  map_seed = training_seed + 1000, decoupling map randomness
                  from training randomness (data split, weight init).
    """

    def __init__(self, k: int, map_seed: int = 1000):
        super().__init__()

        self.k        = k
        self.map_seed = map_seed
        self.k_eff_per_mod: Dict[str, int] = get_all_k_eff(k)

        # One shared generator — seeded once, then advanced sequentially.
        # This ensures all 84 maps are jointly reproducible from map_seed.
        rng = torch.Generator()
        rng.manual_seed(map_seed)

        for (u, v) in DIRECTED_EDGES:
            k_eff_u = self.k_eff_per_mod[u]
            k_eff_v = self.k_eff_per_mod[v]
            k_edge  = min(k_eff_u, k_eff_v)

            # Head map: F_{v←e} ∈ R^{k_edge × d_v}
            F_head = _haar_stiefel(k_edge, D_V[v], rng)
            # Tail map: F_{u←e} ∈ R^{k_edge × d_u}
            F_tail = _haar_stiefel(k_edge, D_V[u], rng)

            # Defensive shape assertions
            assert F_head.shape == (k_edge, D_V[v]), (
                f"F_head shape error for edge ({u}→{v}): "
                f"{F_head.shape} vs ({k_edge}, {D_V[v]})"
            )
            assert F_tail.shape == (k_edge, D_V[u]), (
                f"F_tail shape error for edge ({u}→{v}): "
                f"{F_tail.shape} vs ({k_edge}, {D_V[u]})"
            )

            self.register_buffer(f"F_head_{v}__edge_{u}_{v}", F_head)
            self.register_buffer(f"F_tail_{u}__edge_{u}_{v}", F_tail)

        # Cache k_edge per directed edge for diagnostics
        self._k_edge_cache: Dict[Tuple[str, str], int] = {
            (u, v): min(self.k_eff_per_mod[u], self.k_eff_per_mod[v])
            for (u, v) in DIRECTED_EDGES
        }

    # ── Map access — identical API to restriction_maps_cca ────────────────────

    def get_map(
        self,
        node: str,
        edge: Tuple[str, str],
        role: str,
    ) -> torch.Tensor:
        """
        Return restriction map F_{node←e} for directed edge e = (u → v).

        Args:
            node: The modality node.
            edge: (source, target) directed edge tuple.
            role: 'head' if node is the target, 'tail' if source.

        Returns:
            F: [k_edge, d_v(node)] with F @ F.T = I_{k_edge}.
        """
        u, v = edge
        if role == 'head':
            return getattr(self, f"F_head_{v}__edge_{u}_{v}")
        elif role == 'tail':
            return getattr(self, f"F_tail_{u}__edge_{u}_{v}")
        else:
            raise ValueError(f"role must be 'head' or 'tail', got '{role}'")

    def get_all_maps(self) -> MapDict:
        """
        Return all 84 restriction maps as a dict keyed by (node, source, target).
        Identical API to restriction_maps_cca.get_all_maps().
        """
        result: MapDict = {}
        for (u, v) in DIRECTED_EDGES:
            result[(v, u, v)] = self.get_map(v, (u, v), role='head')
            result[(u, u, v)] = self.get_map(u, (u, v), role='tail')
        return result

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_rotation_angles(self) -> Dict[Tuple[str, str, str], float]:
        """
        Rotation angles θ = arccos(trace(F[:k,:][:,:k].diagonal()) / k_edge)
        for all 84 maps. θ = 0° ↔ F equals the canonical projection P₀.
        For Haar-random maps, angles are broadly distributed near 90°.
        """
        angles = {}
        maps = self.get_all_maps()
        for key, F in maps.items():
            k_edge = F.shape[0]
            with torch.no_grad():
                overlap   = F[:, :k_edge].diagonal().sum()
                cos_theta = (overlap / k_edge).clamp(-1.0, 1.0)
                angles[key] = math.acos(cos_theta.item())
        return angles

    def get_orthogonality_errors(self) -> Dict[Tuple[str, str, str], float]:
        """||F F^T - I_{k_edge}||_F for all 84 maps. Should be < 1e-5."""
        errors = {}
        maps = self.get_all_maps()
        for key, F in maps.items():
            k_edge = F.shape[0]
            with torch.no_grad():
                I_k = torch.eye(k_edge, device=F.device, dtype=F.dtype)
                errors[key] = (F @ F.T - I_k).norm(p='fro').item()
        return errors

    def grouped_parameters(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """
        Maps have no parameters — they are fixed random buffers.
        Returns empty map_params so training optimises only W1 and classifier.
        """
        return [], []

    def parameter_summary(self) -> str:
        n_buffers = sum(1 for _ in self.buffers())
        angles    = list(self.get_rotation_angles().values())
        ang_deg   = [math.degrees(a) for a in angles]
        lines = [
            "─" * 60,
            f"ModalityRestrictionMaps (RandEdge)  k={self.k}",
            f"  map_seed = {self.map_seed}  (= training_seed + 1000)",
            "─" * 60,
            f"  Buffers: {n_buffers} (84 maps, no singular-value diagnostics)",
            f"  Rotation angle — mean: {sum(ang_deg)/len(ang_deg):.2f}°"
            f"   std: {(sum((a-sum(ang_deg)/len(ang_deg))**2 for a in ang_deg)/len(ang_deg))**0.5:.2f}°"
            f"   range: [{min(ang_deg):.2f}°, {max(ang_deg):.2f}°]",
            "─" * 60,
        ]
        return "\n".join(lines)

    def extra_repr(self) -> str:
        n_buf = sum(1 for _ in self.buffers())
        return (
            f"k={self.k}, map_seed={self.map_seed}, "
            f"n_map_buffers={n_buf}, backend=RandEdge (Haar-random, frozen)"
        )
