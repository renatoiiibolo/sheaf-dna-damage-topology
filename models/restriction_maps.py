"""
models/restriction_maps.py — Global Restriction Map Parameters
==============================================================
Stores and parametrises all 84 restriction maps as direct
nn.Parameter tensors — one per directed edge per endpoint.

Architecture decision
---------------------
In Bodnar et al. (2022), restriction maps are outputs of a sheaf
learner Φ(x_v, x_u) — an MLP conditioned on current node features.
This is appropriate when each graph instance has different nodes.

In Project [3], all 2,450 nuclei share the SAME 7 modality nodes.
Feature-conditioned maps would produce per-nucleus maps, destroying
the "universal inter-modality geometry" interpretation and making
E_i incoherent as a measurement of fixed structure.

Therefore: maps are direct nn.Parameter tensors, trained by gradient
descent simultaneously across all 2,450 nuclei.  This is consistent
with Hansen & Gebhart (2020)'s original SNN formulation.

Parameter naming convention
---------------------------
For directed edge e = (u → v):
  - F_{v←e}  :  restriction map at HEAD node v
                projects x_v (stalk of v) into agreement space
                param name: 'F_head_{v}__edge_{u}_{v}'

  - F_{u←e}  :  restriction map at TAIL node u
                projects x_u (stalk of u) into agreement space
                param name: 'F_tail_{u}__edge_{u}_{v}'

Each parameter has shape [d_v, k_eff(v)] — the Householder
reflector parameters (NOT the actual map; the map is obtained
by calling ThinOrthogonal.forward on the parameter).

Map shapes after ThinOrthogonal.forward:
  F_{v←e}: [k_eff(v), d_v]   with F F^T = I_{k_eff(v)}
  F_{u←e}: [k_eff(u), d_u]   with F F^T = I_{k_eff(u)}

k_eff constraint
-----------------
k_eff(v) = min(k, d_v).  For m4 (d_v=10) and m7 (d_v=10),
k_eff = 10 for all k ≥ 10.  The Householder parametrization
requires k_eff < d_v (strict), so k_eff is strictly less than d_v.

Document version: v1.0  (23 March 2026)
"""

import math
from typing import Dict, List, Tuple

import torch
from torch import nn

from config import (
    MODALITY_ORDER, D_V, DIRECTED_EDGES, UNDIRECTED_PAIRS,
    get_k_eff, get_all_k_eff,
)
from models.orthogonal_thin import ThinOrthogonal


# Type alias: (node, source, target) → actual restriction map tensor
MapDict = Dict[Tuple[str, str, str], torch.Tensor]


class ModalityRestrictionMaps(nn.Module):
    """
    Global shared restriction map parameters for Project [3].

    Contains 84 = 42 × 2 Householder parameter matrices as nn.Parameters.
    Actual restriction map tensors (orthonormal-row matrices) are produced
    on demand by calling get_all_maps() or get_map().

    The nn.Parameters are the *Householder reflector parameters*, not the
    maps themselves.  Gradient descent updates these parameters; the maps
    are recomputed each forward pass via ThinOrthogonal.

    Args:
        k (int): Global agreement space dimension (before k_eff capping).
                 Should be one of {10, 16, 24, 32} per the sweep grid.
    """

    def __init__(self, k: int):
        super().__init__()

        self.k = k
        self.k_eff_per_mod: Dict[str, int] = get_all_k_eff(k)

        # Build one ThinOrthogonal module per modality (k_eff may differ)
        # These are NOT nn.Modules with parameters — they are stateless
        # transform functions.  We store them in a plain dict so they
        # are not counted in model.parameters().
        self._orth: Dict[str, ThinOrthogonal] = {
            mod: ThinOrthogonal(d_v=D_V[mod], k=self.k_eff_per_mod[mod])
            for mod in MODALITY_ORDER
        }

        # Build Householder parameter tensors for all 84 maps.
        # Use nn.ParameterDict (keys must be strings, no slashes/dots).
        self.params = nn.ParameterDict()

        for (u, v) in DIRECTED_EDGES:
            k_eff_v = self.k_eff_per_mod[v]
            k_eff_u = self.k_eff_per_mod[u]
            d_v     = D_V[v]
            d_u     = D_V[u]

            # F_{v←e}: projects stalk of HEAD node v into agreement space
            head_key = f"F_head_{v}__edge_{u}_{v}"
            self.params[head_key] = nn.Parameter(
                torch.empty(d_v, k_eff_v)
            )

            # F_{u←e}: projects stalk of TAIL node u into agreement space
            tail_key = f"F_tail_{u}__edge_{u}_{v}"
            self.params[tail_key] = nn.Parameter(
                torch.empty(d_u, k_eff_u)
            )

        # Initialise all parameters with Haar-random orthogonal matrices
        self._init_parameters()

    # ── Initialisation ────────────────────────────────────────

    def _init_parameters(self) -> None:
        """
        Haar-random initialisation on the Stiefel manifold.

        nn.init.orthogonal_ produces an orthonormal matrix from a
        random Gaussian matrix via QR decomposition — this is the
        canonical Haar measure on O(d) restricted to the Stiefel
        manifold.

        Applied to the Householder parameter matrices (not the maps
        themselves), this gives a well-conditioned starting point.
        """
        for param in self.params.values():
            nn.init.orthogonal_(param)

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
            node:  The modality node (head or tail of edge).
            edge:  (source, target) tuple for the directed edge.
            role:  'head' if node is v (the target of u→v),
                   'tail' if node is u (the source of u→v).

        Returns:
            F: [k_eff(node), d_v(node)] restriction map with F F^T = I_k.
        """
        u, v = edge
        if role == 'head':
            key = f"F_head_{v}__edge_{u}_{v}"
        elif role == 'tail':
            key = f"F_tail_{u}__edge_{u}_{v}"
        else:
            raise ValueError(f"role must be 'head' or 'tail', got '{role}'")

        param = self.params[key]           # [d_v, k_eff]
        return self._orth[node](param)     # [k_eff, d_v]

    def get_all_maps(self) -> MapDict:
        """
        Produce all 84 restriction maps.

        Returns a dict keyed by (node, source, target):
          - (v, u, v): F_{v←e} for edge e = (u → v)  [k_eff(v), d_v]
          - (u, u, v): F_{u←e} for edge e = (u → v)  [k_eff(u), d_u]

        This is the primary interface used by laplacian_hetero.py
        and sheaf_modality.py.
        """
        maps: MapDict = {}
        for (u, v) in DIRECTED_EDGES:
            maps[(v, u, v)] = self.get_map(v, (u, v), role='head')
            maps[(u, u, v)] = self.get_map(u, (u, v), role='tail')
        return maps

    # ── Diagnostics ──────────────────────────────────────────

    def get_rotation_angles(self) -> Dict[Tuple[str, str, str], float]:
        """
        Compute rotation angles θ_{v←e} for all 84 maps.

            θ = arccos( trace(F · P₀ᵀ) / k_eff ) = arccos( trace(F[:, :k_eff]) / k_eff )

        where P₀ = [I_{k_eff} | 0] is the canonical (no-rotation) projection.
        θ = 0 when F equals the canonical frame; θ increases with rotation.

        Correction note (v1.1):
          The original formula arccos(trace(F F^T)/k_eff) always returned 0
          because F F^T = I_{k_eff} by construction. Fixed to compare against P₀.

        Returns:
            Dict keyed by (node, source, target) → angle in radians.
        """
        angles = {}
        maps = self.get_all_maps()
        for key, F in maps.items():
            node  = key[0]
            k_eff = self.k_eff_per_mod[node]
            with torch.no_grad():
                # trace(F · P₀ᵀ) = diagonal sum of first k_eff columns of F
                overlap   = F[:, :k_eff].diagonal().sum()
                cos_theta = (overlap / k_eff).clamp(-1.0, 1.0)
                angles[key] = math.acos(cos_theta.item())
        return angles

    def get_orthogonality_errors(self) -> Dict[Tuple[str, str, str], float]:
        """
        Compute || F F^T - I_k ||_F for all 84 maps.

        All values should be < 1e-5.  Large values indicate numerical
        instability in the Householder transform.

        Returns:
            Dict keyed by (node, source, target) → Frobenius error.
        """
        errors = {}
        maps = self.get_all_maps()
        for key, F in maps.items():
            node = key[0]
            k_eff = self.k_eff_per_mod[node]
            with torch.no_grad():
                FF_T = F @ F.T
                I_k  = torch.eye(k_eff, device=F.device, dtype=F.dtype)
                errors[key] = (FF_T - I_k).norm(p='fro').item()
        return errors

    def parameter_summary(self) -> str:
        """
        Return a human-readable summary of all parameter shapes and counts.
        """
        lines = [
            "─" * 64,
            f"ModalityRestrictionMaps  k={self.k}",
            "─" * 64,
            f"  {'Modality':<6}  {'d_v':>4}  {'k_eff':>5}  "
            f"{'n_params_per_map':>16}  {'n_maps':>6}  {'total':>8}",
            f"  {'─'*6}  {'─'*4}  {'─'*5}  {'─'*16}  {'─'*6}  {'─'*8}",
        ]
        grand_total = 0
        for mod in MODALITY_ORDER:
            d_v    = D_V[mod]
            k_eff  = self.k_eff_per_mod[mod]
            # Number of edges incident to this modality as head or tail
            n_head = sum(1 for (u, v) in DIRECTED_EDGES if v == mod)
            n_tail = sum(1 for (u, v) in DIRECTED_EDGES if u == mod)
            n_maps = n_head + n_tail     # = 12 for each modality in K_7
            n_params_each = d_v * k_eff  # params per map (Householder matrix)
            total_mod     = n_maps * n_params_each
            grand_total  += total_mod
            lines.append(
                f"  {mod:<6}  {d_v:>4}  {k_eff:>5}  "
                f"{n_params_each:>16}  {n_maps:>6}  {total_mod:>8}"
            )
        lines += [
            f"  {'─'*6}  {'─'*4}  {'─'*5}  {'─'*16}  {'─'*6}  {'─'*8}",
            f"  {'TOTAL':<6}  {'':>4}  {'':>5}  {'':>16}  {'84':>6}  "
            f"{grand_total:>8}",
            "─" * 64,
            f"  nn.Parameter count (Householder): {grand_total}",
            f"  Actual restriction maps produced:  84",
            "─" * 64,
        ]
        return "\n".join(lines)

    def grouped_parameters(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """
        Separate restriction map parameters from other model parameters.

        Used by the training loop to allow different learning rates or
        weight decay for map parameters vs. W1 / classifier parameters.

        Returns:
            (map_params, []):  All parameters are map parameters here.
        """
        return list(self.params.values()), []

    def extra_repr(self) -> str:
        total_params = sum(p.numel() for p in self.params.values())
        return (
            f"k={self.k}, "
            f"n_edges=42, "
            f"n_maps=84, "
            f"total_param_elements={total_params}"
        )
