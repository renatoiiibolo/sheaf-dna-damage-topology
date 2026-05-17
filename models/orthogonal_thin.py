"""
models/orthogonal_thin.py — Thin Orthogonal Map Parametrization
================================================================
Implements parametrization of elements of the Stiefel manifold

    V_k(R^{d_v}) = { F ∈ R^{k × d_v} : F F^T = I_k }

i.e., k × d_v matrices with orthonormal rows (orthonormal k-frames
in R^{d_v}).  Requires k < d_v (strict thin constraint).

Mathematical background
-----------------------
The Stiefel manifold V_k(R^{d_v}) has dimension

    dim = d_v * k - k*(k-1)/2

Every element can be written as Q^T where Q is a d_v × k matrix
with orthonormal COLUMNS produced by k Householder reflections
acting on a d_v × k lower-triangular parameter matrix P:

    hh = P.tril(diagonal=-1) + eye(d_v, k)
    Q  = householder_orgqr(hh)   # [d_v, k],  Q^T Q = I_k
    F  = Q^T                     # [k, d_v],  F F^T = I_k

This is the parametrization used by Obukhov (2021) torch-householder,
which supports thin (d_v > k) matrices natively and is both faster
and more memory-efficient than torch.matrix_exp or PyTorch's built-in
householder_product for this use case.

The k_eff constraint
--------------------
For [3]'s heterogeneous stalks, k_eff(v) = min(k, d_v).
Modalities m4 (d_v=10) and m7 (d_v=10) have k_eff=10 for all k ≥ 10.
This class is always instantiated with the *already-capped* k_eff, so
the caller is responsible for applying config.get_k_eff() before
constructing ThinOrthogonal.

Usage
-----
    from models.orthogonal_thin import ThinOrthogonal

    # For modality m1 (d_v=33) with k=24
    layer = ThinOrthogonal(d_v=33, k=24)
    params = torch.randn(33, 24)         # parameter matrix
    F = layer(params)                    # [24, 33], F F^T = I_24

    # Batched (one param matrix per edge)
    params_batch = torch.randn(42, 33, 24)
    F_batch = layer(params_batch)        # [42, 24, 33]

Document version: v1.0  (23 March 2026)
"""

import math
import torch
from torch import nn

# Try importing torch_householder; fall back to torch.linalg.householder_product
# with a warning if the package is not installed.
try:
    from torch_householder import torch_householder_orgqr
    _HAVE_TORCH_HOUSEHOLDER = True
except ImportError:
    _HAVE_TORCH_HOUSEHOLDER = False


class ThinOrthogonal(nn.Module):
    """
    Differentiable parametrization of V_k(R^{d_v}) via Householder reflectors.

    Produces a k × d_v matrix F with F F^T = I_k from a d_v × k
    lower-triangular Householder parameter matrix.

    Supports:
      - Single map:  params [d_v, k]       → F [k, d_v]
      - Batched:     params [B, d_v, k]    → F [B, k, d_v]

    Args:
        d_v (int): Stalk dimension. Must satisfy d_v > k.
        k   (int): Agreement space dimension (k_eff, already capped).
    """

    def __init__(self, d_v: int, k: int):
        super().__init__()

        if k >= d_v:
            raise ValueError(
                f"ThinOrthogonal requires strict k < d_v. "
                f"Got k={k}, d_v={d_v}. "
                f"Apply config.get_k_eff(k, modality) before constructing."
            )
        if k < 1:
            raise ValueError(f"k must be ≥ 1, got {k}.")
        if d_v < 1:
            raise ValueError(f"d_v must be ≥ 1, got {d_v}.")

        self.d_v = d_v
        self.k   = k

        # Number of free parameters = dim(V_k(R^{d_v}))
        # = d_v*k - k*(k-1)/2  (lower-triangular entries below diagonal)
        self.n_params = d_v * k - k * (k - 1) // 2

        if not _HAVE_TORCH_HOUSEHOLDER:
            import warnings
            warnings.warn(
                "torch_householder not found. Falling back to "
                "torch.linalg.householder_product, which is slower and "
                "uses more memory. Install with: pip install torch-householder",
                stacklevel=2,
            )

    # ── Forward ──────────────────────────────────────────────

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Convert Householder parameter matrix to orthonormal frame.

        Args:
            params: Householder reflector parameters.
                    Shape [d_v, k]      — single map.
                    Shape [B, d_v, k]   — batched maps.

        Returns:
            F: Restriction map(s) with F F^T = I_k.
               Shape [k, d_v]      — single map.
               Shape [B, k, d_v]   — batched maps.
        """
        single = (params.dim() == 2)
        if single:
            params = params.unsqueeze(0)   # [1, d_v, k]

        B = params.shape[0]
        assert params.shape[1] == self.d_v, (
            f"params.shape[1]={params.shape[1]} != d_v={self.d_v}"
        )
        assert params.shape[2] == self.k, (
            f"params.shape[2]={params.shape[2]} != k={self.k}"
        )

        # Build Householder matrix: lower-triangular + unit diagonal
        # Convention from Obukhov (2021) and torch-householder docs
        eye = torch.eye(
            self.d_v, self.k,
            device=params.device, dtype=params.dtype
        ).unsqueeze(0).expand(B, -1, -1)  # [B, d_v, k]

        hh = params.tril(diagonal=-1) + eye   # [B, d_v, k]

        # Produce orthonormal columns Q: [B, d_v, k], Q^T Q = I_k
        if _HAVE_TORCH_HOUSEHOLDER:
            Q = torch_householder_orgqr(hh)   # [B, d_v, k]
        else:
            # Fallback: torch.linalg.householder_product expects (A, tau)
            # For compatibility, use QR decomposition on the parameter matrix
            Q, _ = torch.linalg.qr(hh, mode='reduced')  # [B, d_v, k]

        # Transpose to get F = Q^T: [B, k, d_v], F F^T = I_k
        F = Q.transpose(-2, -1)   # [B, k, d_v]

        if single:
            F = F.squeeze(0)   # [k, d_v]

        return F

    # ── Diagnostics ──────────────────────────────────────────

    def rotation_angle(self, F: torch.Tensor) -> float:
        """
        Compute effective rotation angle from canonical projection P₀ = [I_k | 0].

            θ = arccos( trace(F · P₀ᵀ) / k )
              = arccos( trace(F[:, :k]) / k )    ∈ [0, π/2]

        θ = 0 when F equals the canonical first-k-rows frame.
        θ increases as F rotates away from that frame.

        Correction note (v1.1):
          Original formula arccos(trace(F F^T)/k) is always 0 because
          F F^T = I_k for any valid orthonormal frame. The correct formula
          compares F against the canonical projection P₀.

        Args:
            F: [k, d_v] restriction map tensor.

        Returns:
            Rotation angle in radians (float).
        """
        with torch.no_grad():
            # trace(F · P₀ᵀ) = sum of diagonal of F[:, :self.k]
            overlap   = F[:, :self.k].diagonal().sum()
            cos_theta = (overlap / self.k).clamp(-1.0, 1.0)
            return math.acos(cos_theta.item())

    def orthogonality_error(self, F: torch.Tensor) -> float:
        """
        Measure deviation from perfect orthonormality.

            err = || F F^T - I_k ||_F

        Should be < 1e-5 for a well-conditioned map.

        Args:
            F: [k, d_v] restriction map tensor.

        Returns:
            Frobenius norm of (F F^T - I_k) as float.
        """
        with torch.no_grad():
            FF_T = F @ F.T
            I_k  = torch.eye(self.k, device=F.device, dtype=F.dtype)
            return (FF_T - I_k).norm(p='fro').item()

    def extra_repr(self) -> str:
        return (
            f"d_v={self.d_v}, k={self.k}, "
            f"n_params={self.n_params}, "
            f"backend={'torch_householder' if _HAVE_TORCH_HOUSEHOLDER else 'torch.linalg.qr'}"
        )
