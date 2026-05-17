"""
config.py — Project-wide constants for sheaf-dna-damage-topology
=================================================================
Defines modality structure, stalk dimensions, graph topology,
and utility functions shared across models/, training/, and analysis/.

All downstream code imports from here; do not hardcode these values
in individual scripts.

Project: [3] — Sheaf-learned restriction maps for radiation-induced
               DNA damage topology
Arc:     Project [3] of the Bolo & Bagunu computational radiation
         biophysics arc (2024–2027)
Dataset: 2,450 nuclei × 49 conditions (7 particles × 7 O₂ levels × 50 nuclei)
         107-dimensional feature matrix across 7 modalities (m1–m7)
"""

from __future__ import annotations
from itertools import combinations, permutations
from typing import Dict, List, Tuple


# ── Modality order and stalk dimensions ───────────────────────────────────────

MODALITY_ORDER: List[str] = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7']

D_V: Dict[str, int] = {
    'm1': 33,   # Spatial distribution
    'm2': 14,   # Radial track structure
    'm3': 13,   # Local energy heterogeneity
    'm4': 10,   # Dose distribution
    'm5': 16,   # Genomic location
    'm6': 11,   # Damage complexity
    'm7': 10,   # Topological summaries
}

TOTAL_FEATURES: int = sum(D_V.values())   # = 107

# Human-readable modality descriptions (for plots and reports)
MODALITY_DESCRIPTIONS: Dict[str, str] = {
    'm1': 'Spatial distribution',
    'm2': 'Radial track structure',
    'm3': 'Local energy heterogeneity',
    'm4': 'Dose distribution',
    'm5': 'Genomic location',
    'm6': 'Damage complexity',
    'm7': 'Topological summaries',
}


# ── Stalk slices into the 107-dimensional feature vector ──────────────────────

def _build_stalk_slices() -> Dict[str, Tuple[int, int]]:
    slices: Dict[str, Tuple[int, int]] = {}
    idx = 0
    for mod in MODALITY_ORDER:
        slices[mod] = (idx, idx + D_V[mod])
        idx += D_V[mod]
    assert idx == TOTAL_FEATURES, \
        f"Slice sum {idx} != TOTAL_FEATURES {TOTAL_FEATURES}"
    return slices


STALK_SLICES: Dict[str, Tuple[int, int]] = _build_stalk_slices()
# STALK_SLICES['m1'] = (0, 33)
# STALK_SLICES['m2'] = (33, 47)
# STALK_SLICES['m3'] = (47, 60)
# STALK_SLICES['m4'] = (60, 70)
# STALK_SLICES['m5'] = (70, 86)
# STALK_SLICES['m6'] = (86, 97)
# STALK_SLICES['m7'] = (97, 107)


# ── Modality graph topology ───────────────────────────────────────────────────

# Fully connected directed graph: 7 nodes × 6 outgoing edges = 42 directed edges
DIRECTED_EDGES: List[Tuple[str, str]] = [
    (u, v)
    for u in MODALITY_ORDER
    for v in MODALITY_ORDER
    if u != v
]
assert len(DIRECTED_EDGES) == 42

# 21 canonical undirected pairs (for CCSV deduplication)
UNDIRECTED_PAIRS: List[Tuple[str, str]] = [
    (u, v)
    for i, u in enumerate(MODALITY_ORDER)
    for v in MODALITY_ORDER[i + 1:]
]
assert len(UNDIRECTED_PAIRS) == 21


# ── Simulation conditions ─────────────────────────────────────────────────────

N_CONDITIONS: int = 49   # 7 particles × 7 O₂ levels

# Particle configurations: (label, LET in keV/µm, SOBP position)
PARTICLE_CONFIGS = [
    ('e-',  0.2,  'secondary_spectrum'),
    ('p+p', 4.6,  'pSOBP'),
    ('p+d', 8.1,  'dSOBP'),
    ('Hep', 10.0, 'pSOBP'),
    ('Hed', 30.0, 'dSOBP'),
    ('Cp',  40.9, 'pSOBP'),
    ('Cd',  70.7, 'dSOBP'),
]

# O₂ levels (%, v/v), ordered anoxia → normoxia
O2_LEVELS = [0.005, 0.021, 0.1, 0.5, 2.1, 5.0, 21.0]

# VOxA kinetic threshold (K_fix + K_repair)
VOXYA_O2_THRESHOLD_PCT  = 0.371    # % O₂ (v/v)
VOXYA_O2_THRESHOLD_MMHG = 2.82    # mmHg

# OER_max (bootstrap median and 95% CI from VOxA)
OER_MAX_MEDIAN   = 3.38
OER_MAX_CI_LOWER = 3.19
OER_MAX_CI_UPPER = 4.20


# ── Agreement space utilities ─────────────────────────────────────────────────

def get_k_eff(k: int, modality: str) -> int:
    """
    Effective agreement space dimension for a single modality.

    The QR and Householder parametrizations require k < d_v (strict),
    so k_eff is capped at d_v - 1.

    Args:
        k:        Global agreement space dimension.
        modality: One of MODALITY_ORDER.

    Returns:
        k_eff(v) = min(k, d_v - 1).
    """
    return min(k, D_V[modality] - 1)


def get_all_k_eff(k: int) -> Dict[str, int]:
    """
    Effective agreement space dimension for all 7 modalities.

    Args:
        k: Global agreement space dimension.

    Returns:
        Dict mapping modality label to k_eff.
    """
    return {mod: get_k_eff(k, mod) for mod in MODALITY_ORDER}


def edge_k_min(u: str, v: str, k: int) -> int:
    """
    Edge stalk dimension k_e = min(k_eff(u), k_eff(v)).

    Args:
        u, v: Source and target modality labels.
        k:    Global agreement space dimension.

    Returns:
        k_e = min(k_eff(u), k_eff(v)).
    """
    return min(get_k_eff(k, u), get_k_eff(k, v))


# ── Amalfi Coast colour palette (used in all figures) ────────────────────────
# Source: Bolo & Bagunu arc-wide visualisation standard (Project [1] onwards)

PALETTE = {
    'CCSV':     '#1D4E63',   # marine deep
    'QR':       '#CD5F00',   # chili cliff
    'RandEdge': '#D4845A',   # melon flesh
}

# Per-O₂-level colours (anoxia → normoxia)
O2_PALETTE = [
    '#1D4E63',   # 0.005 %  — dark marine
    '#2A6F8A',   # 0.021 %
    '#3D8FA8',   # 0.1 %
    '#5AACCB',   # 0.5 %  — VOxA anchor
    '#88C9E0',   # 2.1 %
    '#B5DEF0',   # 5.0 %
    '#E0F4FA',   # 21.0 % — normoxia, pale aqua
]

# Per-particle colours
PARTICLE_PALETTE = {
    'e-':  '#F4A261',
    'p+p': '#E76F51',
    'p+d': '#E9C46A',
    'Hep': '#2A9D8F',
    'Hed': '#264653',
    'Cp':  '#6D6875',
    'Cd':  '#B5838D',
}


# ── Reproducibility seeds used in the paper ───────────────────────────────────
# Production results reported in the manuscript use seeds 0, 1, 2.
# RandEdge map seeds are training_seed + 1000 (decouples map and training
# randomness).

TRAINING_SEEDS   = [0, 1, 2]
MAP_SEED_OFFSET  = 1000   # RandEdge map_seed = training_seed + MAP_SEED_OFFSET
