"""
models/restriction_maps_cca.py — CCA-Derived Analytical Restriction Maps
=========================================================================
Computes restriction maps analytically from the training-set cross-covariance
structure via Canonical Correlation Analysis (CCA) through SVD.

Scientific motivation
---------------------
The biological question is: do the 7 measurement modalities give geometrically
coherent accounts of the same radiation event within a single nucleus?

The restriction maps should encode the population-level geometric relationship
between modality pairs — derived from the data itself, free of classification
pressure. For each modality pair (u, v), the canonical correlation directions
are the subspaces in which u and v are maximally correlated across the training
population. These become the fixed restriction maps.

The per-nucleus consistency energy E_i then measures how much nucleus i's
feature configuration deviates from the population-level inter-modality
covariance structure — a direct biophysical measurement, not a classification
artefact.

Mathematical construction
-------------------------
For directed edge e = (u → v), with canonical pair ordering c0 ≤ c1
(lexicographic), and C_{c0,c1} = X_{c0}^T X_{c1} / n_train:

  SVD:  C_{c0,c1} = U Σ Vhᵀ
    U   ∈ R^{d_{c0} × r}  (left  singular vectors: canonical dirs in c0's space)
    Vhᵀ ∈ R^{r × d_{c1}}  (right singular vectors: canonical dirs in c1's space)
    r   = min(d_{c0}, d_{c1})

  k_edge = min(k_eff(u), k_eff(v))   [same k_min logic as laplacian_hetero.py]

  If u = c0 (canonical-first ordering):
    F_{u←e} = U[:, :k_edge]ᵀ  ∈ R^{k_edge × d_u}   (orthonormal rows)
    F_{v←e} = Vh[:k_edge, :]  ∈ R^{k_edge × d_v}   (orthonormal rows)

  If u = c1 (reverse ordering, u > v lexicographically):
    F_{u←e} = Vh[:k_edge, :]  ∈ R^{k_edge × d_u}
    F_{v←e} = U[:, :k_edge]ᵀ ∈ R^{k_edge × d_v}

Note on symmetry: for a given unordered pair {u, v}, both directed edges
(u→v) and (v→u) use the same projection matrices for each modality. The
sheaf asymmetry arises from the coboundary sign convention, not from distinct
maps per direction. The sum over 42 directed edges gives exactly 2× the sum
over 21 undirected pairs — a constant factor absorbed into E_i magnitude.

Storage: maps are register_buffer tensors — device-tracked, not autograd-tracked.
Maps are frozen throughout training. Only W1 and the classifier train.

Public API is identical to restriction_maps_qr.ModalityRestrictionMaps so all
downstream code (laplacian_hetero.py, sheaf_modality.py, outputs/writer.py)
requires no changes.

Document version: v1.0  (24 March 2026)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
from torch import nn

from config import (
    MODALITY_ORDER, D_V, DIRECTED_EDGES,
    STALK_SLICES, get_k_eff, get_all_k_eff,
)

# Type alias: (node, source, target) → restriction map tensor
MapDict = Dict[Tuple[str, str, str], torch.Tensor]


class ModalityRestrictionMaps(nn.Module):
    """
    CCA-derived analytical restriction maps for Project [3].

    Computes and stores 84 restriction maps from the training-set
    cross-covariance structure. Maps are frozen register_buffer
    tensors — not nn.Parameters, gradients do not flow through them.

    All downstream code that calls get_all_maps() or get_map() is
    fully compatible with this class without modification.

    Args:
        k:       Global agreement space dimension.
        X_train: [n_train, 107] z-score normalised training features.
                 Must be computed from the training split only.
    """

    def __init__(self, k: int, X_train: torch.Tensor):
        super().__init__()

        self.k = k
        self.k_eff_per_mod: Dict[str, int] = get_all_k_eff(k)
        n_train = X_train.shape[0]

        # ── Step 1: Compute SVD once per unordered modality pair ──────────
        # Canonical ordering: lexicographic on modality name strings.
        # This ensures C_{c0,c1} is always computed in the same direction
        # regardless of which directed edge triggers the computation.

        pair_svd: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        for (u, v) in DIRECTED_EDGES:
            # Establish canonical pair order
            c0, c1 = (u, v) if u <= v else (v, u)

            if (c0, c1) not in pair_svd:
                s0, e0 = STALK_SLICES[c0]
                s1, e1 = STALK_SLICES[c1]
                X_c0 = X_train[:, s0:e0]   # [n_train, d_c0]
                X_c1 = X_train[:, s1:e1]   # [n_train, d_c1]

                # Cross-covariance: C_{c0,c1} ∈ R^{d_c0 × d_c1}
                C = (X_c0.T @ X_c1) / float(n_train)

                # Full thin SVD (no gradient needed here)
                with torch.no_grad():
                    U, S, Vh = torch.linalg.svd(C, full_matrices=False)
                # U:  [d_c0, r]   left  singular vectors
                # S:  [r]         singular values
                # Vh: [r, d_c1]   right singular vectors (already transposed)
                # r = min(d_c0, d_c1)

                pair_svd[(c0, c1)] = (U, S, Vh)

        # ── Step 2: Assign maps to all 84 directed edges ──────────────────
        # For each directed edge (u→v), determine which modality is c0 and
        # which is c1, then extract the appropriate singular vectors.

        for (u, v) in DIRECTED_EDGES:
            c0, c1 = (u, v) if u <= v else (v, u)
            U, S, Vh = pair_svd[(c0, c1)]

            # Agreement space dimension for this edge
            k_eff_u = self.k_eff_per_mod[u]
            k_eff_v = self.k_eff_per_mod[v]
            k_edge  = min(k_eff_u, k_eff_v)

            # r = number of available singular vectors
            r = S.shape[0]
            if k_edge > r:
                # Should not happen given k_eff ≤ d_v - 1 and r = min(d_c0, d_c1),
                # but guard defensively
                k_edge = r

            if u == c0:
                # u is canonical-first: F_{u←e} from left SVs, F_{v←e} from right SVs
                F_tail = U[:, :k_edge].T.contiguous()   # [k_edge, d_u]
                F_head = Vh[:k_edge, :].contiguous()    # [k_edge, d_v]
            else:
                # u is canonical-second: roles are swapped
                # C_{c0,c1} = C_{v,u}: left SVs belong to v, right SVs to u
                F_tail = Vh[:k_edge, :].contiguous()    # [k_edge, d_u]
                F_head = U[:, :k_edge].T.contiguous()   # [k_edge, d_v]

            # Verify shapes before registering
            assert F_tail.shape == (k_edge, D_V[u]), (
                f"F_tail shape mismatch for edge ({u}→{v}): "
                f"got {F_tail.shape}, expected ({k_edge}, {D_V[u]})"
            )
            assert F_head.shape == (k_edge, D_V[v]), (
                f"F_head shape mismatch for edge ({u}→{v}): "
                f"got {F_head.shape}, expected ({k_edge}, {D_V[v]})"
            )

            # Register as buffers (device-tracked, not gradient-tracked)
            self.register_buffer(f"F_head_{v}__edge_{u}_{v}", F_head)
            self.register_buffer(f"F_tail_{u}__edge_{u}_{v}", F_tail)

        # ── Step 3: Store top singular values for diagnostic access ───────
        # Singular values of C_{c0,c1}/n_train are the cross-covariance
        # singular values. These are NOT canonical correlations (which require
        # whitening by marginal standard deviations). For z-scored features
        # with correlated dimensions within each modality, singular values
        # can exceed 1.0. They are stored purely for interpretability —
        # the maps themselves are mathematically valid regardless of magnitude.

        for (c0, c1), (U, S, Vh) in pair_svd.items():
            # Retrieve k_edge for this canonical pair from either direction
            k_eff_c0 = self.k_eff_per_mod[c0]
            k_eff_c1 = self.k_eff_per_mod[c1]
            k_edge = min(k_eff_c0, k_eff_c1, S.shape[0])
            self.register_buffer(
                f"S__pair_{c0}_{c1}",
                S[:k_edge].contiguous()
            )

        # Cache k_edge per directed edge for shape-aware diagnostics
        self._k_edge_cache: Dict[Tuple[str, str], int] = {
            (u, v): min(self.k_eff_per_mod[u], self.k_eff_per_mod[v])
            for (u, v) in DIRECTED_EDGES
        }

    # ── Map access — identical API to restriction_maps_qr ─────────────────

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
            F: [k_edge, d_v(node)] with F F^T = I_{k_edge}.
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
        Return all 84 restriction maps.

        Returns a dict keyed by (node, source, target):
          - (v, u, v): F_{v←e} for edge e = (u → v)  [k_edge, d_v]
          - (u, u, v): F_{u←e} for edge e = (u → v)  [k_edge, d_u]
        """
        result: MapDict = {}
        for (u, v) in DIRECTED_EDGES:
            result[(v, u, v)] = self.get_map(v, (u, v), role='head')
            result[(u, u, v)] = self.get_map(u, (u, v), role='tail')
        return result

    # ── Diagnostics ───────────────────────────────────────────────────────

    def get_rotation_angles(self) -> Dict[Tuple[str, str, str], float]:
        """
        Rotation angles θ = arccos(trace(F · P₀ᵀ) / k_edge) for all 84 maps,
        where P₀ = [I_{k_edge} | 0] is the canonical projection.
        θ = 0 ↔ F equals the first-k_edge rows of I_{d_v}.
        """
        angles = {}
        maps = self.get_all_maps()
        for key, F in maps.items():
            k_edge = F.shape[0]
            with torch.no_grad():
                # diagonal of F[:, :k_edge] gives trace(F · P₀ᵀ)
                overlap   = F[:, :k_edge].diagonal().sum()
                cos_theta = (overlap / k_edge).clamp(-1.0, 1.0)
                angles[key] = math.acos(cos_theta.item())
        return angles

    def get_orthogonality_errors(self) -> Dict[Tuple[str, str, str], float]:
        """|| F F^T - I_{k_edge} ||_F for all 84 maps. Should be < 1e-5."""
        errors = {}
        maps = self.get_all_maps()
        for key, F in maps.items():
            k_edge = F.shape[0]
            with torch.no_grad():
                I_k = torch.eye(k_edge, device=F.device, dtype=F.dtype)
                errors[key] = (F @ F.T - I_k).norm(p='fro').item()
        return errors

    def get_canonical_correlations(self) -> Dict[Tuple[str, str], torch.Tensor]:
        """
        Return top-k singular values of C_{c0,c1}/n_train for each
        canonical pair. These are cross-covariance singular values,
        NOT canonical correlations proper (which would require whitening
        by within-modality covariance). Values may exceed 1.0 for
        z-scored (unit-variance) but internally correlated modalities.
        Used for interpretability only — the maps are valid regardless.

        Returns:
            Dict keyed by (c0, c1) with c0 ≤ c1 lexicographically.
        """
        corrs = {}
        for name, buf in self.named_buffers():
            if name.startswith("S__pair_"):
                # Extract pair names from buffer name: "S__pair_{c0}_{c1}"
                parts = name[len("S__pair_"):].split("_")
                # Careful: modality names are m1..m7 so split on "_" gives
                # exactly two parts: ["m1", "m2"] etc.
                # But "carbon_dsobp" etc. don't appear in modality names, fine.
                c0, c1 = parts[0], parts[1]
                corrs[(c0, c1)] = buf
        return corrs

    def grouped_parameters(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """
        Maps have no parameters — they are fixed analytical buffers.
        Returns empty map_params so the training loop optimises only
        W1 and the classifier.
        """
        return [], []

    def parameter_summary(self) -> str:
        """Human-readable summary of map shapes and buffer count."""
        n_buffers = sum(1 for _ in self.buffers())
        lines = [
            "─" * 64,
            f"ModalityRestrictionMaps (CCA)  k={self.k}",
            "─" * 64,
            f"  {'Pair':<12}  {'d_c0':>5}  {'d_c1':>5}  {'k_edge':>6}  "
            f"{'top_sv':>9}",
            f"  {'─'*12}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*9}",
        ]
        corrs = self.get_canonical_correlations()
        for (c0, c1), S in sorted(corrs.items()):
            d_c0 = D_V[c0]
            d_c1 = D_V[c1]
            k_edge = S.shape[0]
            top = S[0].item() if len(S) > 0 else float('nan')
            lines.append(
                f"  {c0+'–'+c1:<12}  {d_c0:>5}  {d_c1:>5}  {k_edge:>6}  {top:>9.4f}"
            )
        lines += [
            "─" * 64,
            f"  Buffers: {n_buffers} total  "
            f"(84 maps + {n_buffers - 84} singular-value diagnostics)",
            "─" * 64,
        ]
        return "\n".join(lines)

    def extra_repr(self) -> str:
        n_buf = sum(1 for _ in self.buffers())
        return f"k={self.k}, n_map_buffers={n_buf}, backend=CCA (analytical, frozen)"
