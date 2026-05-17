"""
models/laplacian_hetero.py — Heterogeneous Sheaf Laplacian
===========================================================
Builds the 107 × 107 sheaf Laplacian L_F = δ^T δ from 84
restriction maps and computes the per-nucleus consistency energy
E_i = x_i^T L_F x_i for batches of nuclei.

Mathematical background
-----------------------
For a directed edge e = (u → v) with restriction maps:
  F_{v←e} ∈ R^{k_eff(v) × d_v}   (head node v)
  F_{u←e} ∈ R^{k_eff(u) × d_u}   (tail node u)

the coboundary discrepancy at edge e for nucleus i is:

  (δx)_{e,i} = F_{v←e}[:k_min] x_{v,i} − F_{u←e}[:k_min] x_{u,i}
                                                        ∈ R^{k_min}

where k_min(e) = min(k_eff(v), k_eff(u)) is the effective agreement
space dimension for this edge — computed dynamically, never hardcoded.

The per-edge Laplacian contribution is (δ_e)^T (δ_e):

  Diagonal    L_{vv} += (F_v[:k_min])^T (F_v[:k_min])   [d_v × d_v]
              L_{uu} += (F_u[:k_min])^T (F_u[:k_min])   [d_u × d_u]
  Off-diag    L_{vu} -= (F_v[:k_min])^T (F_u[:k_min])   [d_v × d_u]
              L_{uv} -= (F_u[:k_min])^T (F_v[:k_min])   [d_u × d_v]

Symmetry: L_{uv} = L_{vu}^T by construction.
PSD: L = δ^T δ is PSD by construction.

When k_eff(v) = k_eff(u) (all same-d_v modality pairs), k_min = k_eff
and the formulas reduce to the standard case.  When k_eff differs
(e.g., edges between m1 d_v=33 and m4 d_v=10 at k=24), k_min is the
smaller value and both maps are truncated to their first k_min rows.

Per-nucleus consistency energy
------------------------------
  E_i = x_i^T L_F x_i = Σ_{e} ||(δx)_{e,i}||^2
                       = Σ_{e} ||F_v[:k_min] x_{v,i}
                                − F_u[:k_min] x_{u,i}||^2

Computed directly from the per-edge discrepancies without explicitly
building L_F — more efficient and numerically equivalent.

Per-edge energy decomposition
------------------------------
  e_{e,i} = ||F_v[:k_min] x_{v,i} − F_u[:k_min] x_{u,i}||^2

The [42, B] matrix of per-edge energies is the primary source for
all interpretability outputs (§9 of the methodology document).

Design choices
--------------
1. L_F is assembled as a dense 107 × 107 matrix (small enough).
   Sparsity overhead is not justified at this graph scale.

2. k_min per edge is computed dynamically from k_eff values — never
   hardcoded.  All block sizes derive from config.py.

3. Normalised Laplacian Δ_F = D^{-1/2} L_F D^{-1/2} uses per-block
   eigendecomposition since the diagonal blocks are projection matrices
   (not scalar multiples of I), requiring the full batched_sym_matrix_pow
   path (as in Bodnar's GeneralLaplacianBuilder).

4. Augmented normalisation D* = D + I is used as default to prevent
   singularity when a modality stalk contributes near-zero diagonal.

Document version: v1.0  (23 March 2026)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from config import (
    MODALITY_ORDER, D_V, STALK_SLICES, DIRECTED_EDGES,
    EDGE_TO_IDX, TOTAL_FEATURES, get_k_eff,
)

# Type alias used throughout this module
MapDict = Dict[Tuple[str, str, str], torch.Tensor]


# ─────────────────────────────────────────────────────────────
# k_min helper — fully dynamic, no hardcoded values
# ─────────────────────────────────────────────────────────────

def edge_k_min(u: str, v: str, k: int) -> int:
    """
    Effective agreement space dimension for directed edge (u → v).

    When k_eff(v) == k_eff(u) this equals k_eff.
    When they differ, the smaller dimension is used so that both
    restriction maps can be truncated to the same row count before
    computing coboundary discrepancies and Laplacian blocks.

    This is computed dynamically from config.get_k_eff — no hardcoding.

    Args:
        u: Source modality name.
        v: Target modality name.
        k: Global agreement space dimension.

    Returns:
        k_min(e) = min(k_eff(v), k_eff(u))
    """
    return min(get_k_eff(k, v), get_k_eff(k, u))


def all_edge_k_mins(k: int) -> Dict[Tuple[str, str], int]:
    """
    Pre-compute k_min for all 42 directed edges given global k.

    Returns:
        Dict mapping (u, v) → k_min(e) for all directed edges.
    """
    return {(u, v): edge_k_min(u, v, k) for (u, v) in DIRECTED_EDGES}


# ─────────────────────────────────────────────────────────────
# Sheaf Laplacian assembly
# ─────────────────────────────────────────────────────────────

def build_sheaf_laplacian(
    maps: MapDict,
    k: int,
) -> torch.Tensor:
    """
    Assemble the 107 × 107 un-normalised sheaf Laplacian L_F = δ^T δ.

    Each directed edge e = (u → v) contributes four blocks to L_F
    using only the first k_min rows of each restriction map.

    Args:
        maps: Dict from ModalityRestrictionMaps.get_all_maps().
              Keys: (node, src, tgt) → [k_eff(node), d_v(node)] tensor.
        k:    Global agreement space dimension (used to compute k_min).

    Returns:
        L_F: [TOTAL_FEATURES, TOTAL_FEATURES] symmetric PSD tensor.
             dtype and device inferred from maps.
    """
    # Infer device and dtype from maps
    sample = next(iter(maps.values()))
    device, dtype = sample.device, sample.dtype

    L = torch.zeros(TOTAL_FEATURES, TOTAL_FEATURES, device=device, dtype=dtype)

    for (u, v) in DIRECTED_EDGES:
        F_v = maps[(v, u, v)]   # [k_eff(v), d_v]
        F_u = maps[(u, u, v)]   # [k_eff(u), d_u]

        k_min = edge_k_min(u, v, k)  # dynamic, not hardcoded

        # Truncate to k_min rows (no-op when k_eff == k_min)
        Fv = F_v[:k_min]   # [k_min, d_v]
        Fu = F_u[:k_min]   # [k_min, d_u]

        sv, ev = STALK_SLICES[v]
        su, eu = STALK_SLICES[u]

        # Diagonal blocks: F^T F  (d_v × d_v, d_u × d_u)
        L[sv:ev, sv:ev] += Fv.T @ Fv
        L[su:eu, su:eu] += Fu.T @ Fu

        # Off-diagonal blocks: -F_v^T F_u  (d_v × d_u)
        off = Fv.T @ Fu          # [d_v, d_u]
        L[sv:ev, su:eu] -= off
        L[su:eu, sv:ev] -= off.T

    return L


def build_normalised_laplacian(
    L: torch.Tensor,
    augmented: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the normalised sheaf Laplacian Δ_F = D^{-1/2} L_F D^{-1/2}.

    D is the block-diagonal of L, with blocks D_v = L[sv:ev, sv:ev]
    of size d_v × d_v.  D^{-1/2} is computed block-wise via
    eigendecomposition (not scalar — required because D_v is a sum
    of projection matrices, not a scalar multiple of I_{d_v}).

    Args:
        L:         [TOTAL_FEATURES, TOTAL_FEATURES] un-normalised Laplacian.
        augmented: If True, use D* = D + I to prevent singularity.
                   Default: True.

    Returns:
        delta_F:     [TOTAL_FEATURES, TOTAL_FEATURES] normalised Laplacian.
        D_sqrt_inv:  [TOTAL_FEATURES, TOTAL_FEATURES] block-diagonal D^{-1/2}.
    """
    device, dtype = L.device, L.dtype
    D_sqrt_inv = torch.zeros_like(L)

    for mod in MODALITY_ORDER:
        sv, ev = STALK_SLICES[mod]
        d_v = D_V[mod]  # derived from config, not hardcoded

        D_v = L[sv:ev, sv:ev].clone()  # [d_v, d_v]

        if augmented:
            D_v_aug = D_v + torch.eye(d_v, device=device, dtype=dtype)
        else:
            D_v_aug = D_v

        # Eigendecomposition for symmetric PSD matrix
        try:
            eigvals, eigvecs = torch.linalg.eigh(D_v_aug)
        except RuntimeError as e:
            raise RuntimeError(
                f"Eigendecomposition failed for modality {mod} "
                f"(block [{sv}:{ev}]): {e}"
            )

        # Compute D_v^{-1/2} — clip negative eigenvalues (numerical noise)
        eigvals_clipped = eigvals.clamp(min=0.0)
        # Avoid division by zero for near-zero eigenvalues
        safe_inv_sqrt = torch.where(
            eigvals_clipped > 1e-10,
            eigvals_clipped.pow(-0.5),
            torch.zeros_like(eigvals_clipped),
        )

        D_v_inv_sqrt = eigvecs @ torch.diag(safe_inv_sqrt) @ eigvecs.T
        D_sqrt_inv[sv:ev, sv:ev] = D_v_inv_sqrt

    delta_F = D_sqrt_inv @ L @ D_sqrt_inv
    return delta_F, D_sqrt_inv


# ─────────────────────────────────────────────────────────────
# Consistency energy — primary measurement output
# ─────────────────────────────────────────────────────────────

def compute_consistency_energy(
    X: torch.Tensor,
    maps: MapDict,
    k: int,
    return_per_edge: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute per-nucleus consistency energy E_i = x_i^T L_F x_i.

    Computed efficiently from per-edge discrepancies — equivalent to
    the quadratic form but avoids explicitly building L_F:

        e_{e,i} = ||F_v[:k_min] x_{v,i} − F_u[:k_min] x_{u,i}||^2
        E_i     = Σ_e e_{e,i}

    Args:
        X:              [B, TOTAL_FEATURES] batch of normalised feature vectors.
                        These should be the INITIAL (pre-diffusion) stalks.
        maps:           Dict from ModalityRestrictionMaps.get_all_maps().
        k:              Global agreement space dimension.
        return_per_edge: If True, also return the [42, B] per-edge matrix.
                         The per-edge decomposition is the source for all
                         interpretability analyses (§9 of methodology).

    Returns:
        E_i:   [B] per-nucleus total consistency energy.
        e_ei:  [42, B] per-edge energies if return_per_edge=True, else None.
    """
    B = X.shape[0]
    edge_energies: List[torch.Tensor] = []

    for idx, (u, v) in enumerate(DIRECTED_EDGES):
        F_v = maps[(v, u, v)]   # [k_eff(v), d_v]
        F_u = maps[(u, u, v)]   # [k_eff(u), d_u]

        k_min = edge_k_min(u, v, k)  # dynamic

        sv, ev = STALK_SLICES[v]
        su, eu = STALK_SLICES[u]

        x_v = X[:, sv:ev]   # [B, d_v]
        x_u = X[:, su:eu]   # [B, d_u]

        # Project both stalks into the k_min-dimensional agreement space
        # Using matrix multiplication: (B, d_v) @ (d_v, k_min) → (B, k_min)
        proj_v = x_v @ F_v[:k_min].T   # [B, k_min]
        proj_u = x_u @ F_u[:k_min].T   # [B, k_min]

        delta  = proj_v - proj_u        # [B, k_min]
        e_e    = (delta ** 2).sum(dim=1)  # [B]
        edge_energies.append(e_e)

    e_ei = torch.stack(edge_energies, dim=0)   # [42, B]
    E_i  = e_ei.sum(dim=0)                     # [B]

    return E_i, (e_ei if return_per_edge else None)


def compute_hodge_decomposition(
    X: torch.Tensor,
    L: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute the Hodge decomposition of each nucleus's feature vector.

    For the sheaf Laplacian L_F, the Hodge decomposition gives:
        x_i = h_i + c_i
    where:
        h_i ∈ ker(L_F)    — harmonic component (global section projection)
        c_i ∈ im(δ^T)    — coboundary component (source of energy E_i)

    E_i = ||c_i||^2 and the incoherence fraction = ||c_i||^2 / ||x_i||^2.

    Implemented via eigendecomposition of L_F (107 × 107 — trivially fast).
    The harmonic space ker(L_F) is spanned by eigenvectors with eigenvalue
    below a numerical threshold.

    Args:
        X: [B, 107] normalised feature batch (pre-diffusion stalks).
        L: [107, 107] un-normalised sheaf Laplacian (from build_sheaf_laplacian).

    Returns:
        Dict with keys:
            'h_norm_sq':        [B] ||h_i||^2 — harmonic component energy
            'c_norm_sq':        [B] ||c_i||^2 — coboundary component energy (= E_i)
            'x_norm_sq':        [B] ||x_i||^2 — total stalk energy
            'incoherence_frac': [B] ||c_i||^2 / ||x_i||^2
            'harmonic_dim':     int — dim(ker(L_F))
    """
    HARMONIC_THRESHOLD = 1e-5  # eigenvalue threshold for harmonic space

    with torch.no_grad():
        eigvals, eigvecs = torch.linalg.eigh(L)   # [107], [107, 107]

        # Identify harmonic eigenvectors (ker(L_F))
        harmonic_mask = eigvals < HARMONIC_THRESHOLD
        harmonic_dim  = harmonic_mask.sum().item()

        if harmonic_dim > 0:
            # V_h: [107, harmonic_dim] — harmonic eigenvector basis
            V_h = eigvecs[:, harmonic_mask]

            # Project x_i onto harmonic space
            # h_i = V_h V_h^T x_i
            # X: [B, 107], V_h: [107, harmonic_dim]
            proj  = X @ V_h          # [B, harmonic_dim]
            H     = proj @ V_h.T     # [B, 107] — harmonic components
            C     = X - H            # [B, 107] — coboundary components
        else:
            H = torch.zeros_like(X)
            C = X.clone()

        x_norm_sq   = (X ** 2).sum(dim=1)          # [B]
        h_norm_sq   = (H ** 2).sum(dim=1)          # [B]
        c_norm_sq   = (C ** 2).sum(dim=1)          # [B]

        # Guard against zero-norm stalks
        incoherence_frac = torch.where(
            x_norm_sq > 1e-12,
            c_norm_sq / x_norm_sq,
            torch.zeros_like(x_norm_sq),
        )

    return {
        'h_norm_sq':        h_norm_sq,
        'c_norm_sq':        c_norm_sq,
        'x_norm_sq':        x_norm_sq,
        'incoherence_frac': incoherence_frac,
        'harmonic_dim':     int(harmonic_dim),
    }


def compute_spectral_properties(
    L: torch.Tensor,
) -> Dict[str, object]:
    """
    Compute spectral properties of the sheaf Laplacian.

    Since L_F is 107 × 107, full eigendecomposition is trivially fast
    (< 1 ms on CPU, < 0.1 ms on GPU).

    Args:
        L: [107, 107] un-normalised sheaf Laplacian.

    Returns:
        Dict with keys:
            'eigenvalues':      [107] tensor, sorted ascending
            'eigenvectors':     [107, 107] tensor
            'spectral_gap':     float — λ_1 - λ_0 (first nonzero minus zero)
            'harmonic_dim':     int — #{λ_i < 1e-5}
            'lambda_max':       float — largest eigenvalue
            'lambda_min_nz':    float — smallest nonzero eigenvalue
    """
    HARMONIC_THRESHOLD = 1e-5

    with torch.no_grad():
        eigvals, eigvecs = torch.linalg.eigh(L)   # ascending order

        harmonic_mask = eigvals < HARMONIC_THRESHOLD
        harmonic_dim  = int(harmonic_mask.sum().item())

        nz_eigvals = eigvals[~harmonic_mask]
        lambda_min_nz = float(nz_eigvals.min().item()) if len(nz_eigvals) > 0 else float('nan')
        lambda_max    = float(eigvals.max().item())

        # Spectral gap: smallest nonzero eigenvalue
        spectral_gap = lambda_min_nz if harmonic_dim < len(eigvals) else 0.0

    return {
        'eigenvalues':   eigvals,
        'eigenvectors':  eigvecs,
        'spectral_gap':  spectral_gap,
        'harmonic_dim':  harmonic_dim,
        'lambda_max':    lambda_max,
        'lambda_min_nz': lambda_min_nz,
    }


# ─────────────────────────────────────────────────────────────
# PSD verification utility
# ─────────────────────────────────────────────────────────────

def verify_psd(L: torch.Tensor, tol: float = 1e-5) -> Dict[str, object]:
    """
    Verify that L is symmetric and positive semi-definite.

    Used in tests and diagnostics.  Not called during training.

    Args:
        L:   [N, N] matrix to test.
        tol: Tolerance for negative eigenvalues and asymmetry.

    Returns:
        Dict with keys:
            'is_symmetric':   bool
            'symmetry_error': float — ||L - L^T||_F
            'is_psd':         bool — all eigenvalues >= -tol
            'min_eigenvalue': float
            'max_eigenvalue': float
            'n_negative':     int — number of eigenvalues < -tol
    """
    with torch.no_grad():
        sym_err   = (L - L.T).norm(p='fro').item()
        is_sym    = sym_err < tol

        eigvals, _ = torch.linalg.eigh(L)
        min_ev     = eigvals.min().item()
        max_ev     = eigvals.max().item()
        n_neg      = int((eigvals < -tol).sum().item())
        is_psd     = n_neg == 0

    return {
        'is_symmetric':   is_sym,
        'symmetry_error': sym_err,
        'is_psd':         is_psd,
        'min_eigenvalue': min_ev,
        'max_eigenvalue': max_ev,
        'n_negative':     n_neg,
    }
