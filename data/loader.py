"""
data/loader.py — Project [3] Data Loader
=========================================
Loads feature_matrix.csv from Project [2], applies z-score normalisation
computed from the training set only, slices into modality stalks, and
performs a stratified 70/15/15 train/val/test split by condition label.

Key design constraints:
  - No per-batch normalisation anywhere in the pipeline.
  - Normalisation statistics computed from TRAINING rows only.
  - Stratification ensures every condition (49) is represented in all splits.
  - Stalk slicing is determined by MODALITY_ORDER in config.py (canonical).

Outputs:
  - DataSplit dataclass with tensors, metadata DataFrames, and artefacts.
  - Saves feature_means.npy and feature_stds.npy for inference use.

Document version: v1.0  (23 March 2026)
"""

import os
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import StratifiedShuffleSplit

from config import (
    MODALITY_ORDER, D_V, ALL_FEATURES, STALK_SLICES,
    TOTAL_FEATURES, META_COLUMNS, N_NUCLEI, N_CONDITIONS,
    N_NUCLEI_PER_CONDITION,
)


# ─────────────────────────────────────────────────────────────
# DataSplit container
# ─────────────────────────────────────────────────────────────

@dataclass
class DataSplit:
    """
    Holds the complete train/val/test split for Project [3].

    All feature tensors are z-score normalised using training-set statistics.
    Metadata DataFrames carry the original Project [2] metadata columns.
    """
    # Feature tensors [N_split, 107] — float32, normalised
    X_train: torch.Tensor
    X_val:   torch.Tensor
    X_test:  torch.Tensor

    # Condition label tensors [N_split] — int64, range 0..48
    y_train: torch.Tensor
    y_val:   torch.Tensor
    y_test:  torch.Tensor

    # Original metadata rows (particle_key, o2, let, etc.)
    meta_train: pd.DataFrame
    meta_val:   pd.DataFrame
    meta_test:  pd.DataFrame

    # Normalisation artefacts (computed from training set)
    feature_means: np.ndarray    # [107]  float64
    feature_stds:  np.ndarray    # [107]  float64

    # Condition label map: (particle_key, o2_str) → label int
    label_map: Dict[Tuple[str, str], int] = field(default_factory=dict)

    # Reverse map: label int → (particle_key, o2_str)
    label_to_condition: Dict[int, Tuple[str, str]] = field(default_factory=dict)

    def describe(self) -> str:
        """Return a human-readable summary of the split."""
        lines = [
            "─" * 56,
            "DataSplit Summary",
            "─" * 56,
            f"  Train nuclei  : {len(self.X_train):>6}",
            f"  Val   nuclei  : {len(self.X_val):>6}",
            f"  Test  nuclei  : {len(self.X_test):>6}",
            f"  Total         : {len(self.X_train)+len(self.X_val)+len(self.X_test):>6}",
            f"  Feature dim   : {self.X_train.shape[1]:>6}",
            f"  Classes       : {len(self.label_map):>6}",
            "",
            "  Normalisation (train set):",
            f"    feature_means  min={self.feature_means.min():.4f}  "
            f"max={self.feature_means.max():.4f}",
            f"    feature_stds   min={self.feature_stds.min():.4f}  "
            f"max={self.feature_stds.max():.4f}",
            "",
            "  Stalk dimensions:",
        ]
        for mod in MODALITY_ORDER:
            sv, ev = STALK_SLICES[mod]
            lines.append(
                f"    {mod}  cols [{sv:>3}:{ev:>3}]  d_v={D_V[mod]}"
            )
        lines.append("─" * 56)
        return "\n".join(lines)

    def stalk(self, X: torch.Tensor, modality: str) -> torch.Tensor:
        """
        Extract stalk tensor for one modality from a feature matrix.

        Args:
            X:        [B, 107] feature batch (must be normalised).
            modality: One of MODALITY_ORDER.

        Returns:
            [B, d_v] stalk tensor.
        """
        sv, ev = STALK_SLICES[modality]
        return X[:, sv:ev]


# ─────────────────────────────────────────────────────────────
# Label construction
# ─────────────────────────────────────────────────────────────

def build_condition_labels(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[Tuple[str, str], int], Dict[int, Tuple[str, str]]]:
    """
    Encode (particle_key, o2) pairs as integer condition labels 0..48.

    Labels are assigned in lexicographic order of (particle_key, o2_str)
    for reproducibility. The ordering is deterministic regardless of the
    row order in the DataFrame.

    Args:
        df: Full feature_matrix DataFrame with 'particle_key' and 'o2' columns.

    Returns:
        labels:             [N_nuclei] int array, values 0..48.
        label_map:          {(particle_key, o2_str): label_int}
        label_to_condition: {label_int: (particle_key, o2_str)}
    """
    # Get sorted unique (particle_key, o2) pairs
    pairs = sorted(
        df[["particle_key", "o2"]].drop_duplicates()
        .itertuples(index=False, name=None)
    )
    assert len(pairs) == N_CONDITIONS, (
        f"Expected {N_CONDITIONS} conditions, found {len(pairs)}. "
        f"Check that feature_matrix.csv is from Project [2] v2.7."
    )

    label_map = {pair: i for i, pair in enumerate(pairs)}
    label_to_condition = {i: pair for pair, i in label_map.items()}

    labels = np.array(
        [label_map[(row.particle_key, str(row.o2))]
         for row in df[["particle_key", "o2"]].itertuples(index=False)],
        dtype=np.int64,
    )
    return labels, label_map, label_to_condition


# ─────────────────────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────────────────────

def load_dataset(
    csv_path: str,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
    save_artefacts_dir: Optional[str] = None,
    verbose: bool = True,
) -> DataSplit:
    """
    Load and prepare the Project [3] dataset.

    Args:
        csv_path:            Path to feature_matrix.csv from Project [2] v2.7.
        val_frac:            Fraction of data for validation (stratified).
        test_frac:           Fraction of data for test (stratified).
        seed:                Random seed for reproducibility.
        save_artefacts_dir:  If given, saves feature_means.npy / feature_stds.npy
                             and a split_summary.json to this directory.
        verbose:             Print progress messages.

    Returns:
        DataSplit: fully populated split container.

    Raises:
        FileNotFoundError:  If csv_path does not exist.
        AssertionError:     If the dataset fails structural integrity checks.
        ValueError:         If required feature columns are missing.
    """
    # ── 1. Load CSV ──────────────────────────────────────────
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"feature_matrix.csv not found at: {csv_path}")

    if verbose:
        print(f"[loader] Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    if verbose:
        print(f"[loader] Loaded {len(df)} rows × {len(df.columns)} columns")

    # ── 2. Structural integrity checks ───────────────────────
    _check_dataframe(df, verbose)

    # ── 3. Build ordered feature matrix [N, 107] ─────────────
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} feature columns in CSV.\n"
            f"First 5 missing: {missing[:5]}"
        )

    X_raw = df[ALL_FEATURES].values.astype(np.float64)  # [2450, 107]
    assert X_raw.shape == (N_NUCLEI, TOTAL_FEATURES), (
        f"Expected shape ({N_NUCLEI}, {TOTAL_FEATURES}), got {X_raw.shape}"
    )

    # ── 4. Build condition labels ─────────────────────────────
    # Ensure o2 column is string for consistent keying
    df["o2"] = df["o2"].astype(str)
    labels, label_map, label_to_condition = build_condition_labels(df)

    # Verify 50 nuclei per condition
    unique, counts = np.unique(labels, return_counts=True)
    assert len(unique) == N_CONDITIONS, (
        f"Expected {N_CONDITIONS} unique conditions, got {len(unique)}"
    )
    if not np.all(counts == N_NUCLEI_PER_CONDITION):
        bad = {label_to_condition[u]: int(c)
               for u, c in zip(unique, counts)
               if c != N_NUCLEI_PER_CONDITION}
        print(f"[loader] WARNING: Unequal nuclei per condition: {bad}")

    # ── 5. Stratified split ───────────────────────────────────
    train_idx, val_idx, test_idx = _stratified_split(
        labels, val_frac, test_frac, seed, verbose
    )

    # ── 6. Compute normalisation from training set ────────────
    X_train_raw = X_raw[train_idx]
    feature_means = X_train_raw.mean(axis=0)           # [107]
    feature_stds  = X_train_raw.std(axis=0, ddof=0)   # [107]

    # Guard against zero-variance features (e.g., constant columns)
    n_zero_var = (feature_stds < 1e-10).sum()
    if n_zero_var > 0:
        zero_feats = [ALL_FEATURES[i]
                      for i in np.where(feature_stds < 1e-10)[0]]
        print(f"[loader] WARNING: {n_zero_var} zero-variance features "
              f"(std set to 1.0): {zero_feats}")
        feature_stds[feature_stds < 1e-10] = 1.0

    def _normalise(arr: np.ndarray) -> np.ndarray:
        return (arr - feature_means) / feature_stds

    X_train_norm = _normalise(X_raw[train_idx])
    X_val_norm   = _normalise(X_raw[val_idx])
    X_test_norm  = _normalise(X_raw[test_idx])

    # ── 7. Sanity-check normalised training features ──────────
    _check_normalisation(X_train_norm, feature_means, feature_stds, verbose)

    # ── 8. Convert to torch tensors ──────────────────────────
    X_train = torch.tensor(X_train_norm, dtype=torch.float32)
    X_val   = torch.tensor(X_val_norm,   dtype=torch.float32)
    X_test  = torch.tensor(X_test_norm,  dtype=torch.float32)

    y_train = torch.tensor(labels[train_idx], dtype=torch.long)
    y_val   = torch.tensor(labels[val_idx],   dtype=torch.long)
    y_test  = torch.tensor(labels[test_idx],  dtype=torch.long)

    # ── 9. Metadata DataFrames ────────────────────────────────
    meta_cols_present = [c for c in META_COLUMNS if c in df.columns]
    meta_train = df.iloc[train_idx][meta_cols_present].reset_index(drop=True)
    meta_val   = df.iloc[val_idx][meta_cols_present].reset_index(drop=True)
    meta_test  = df.iloc[test_idx][meta_cols_present].reset_index(drop=True)

    # ── 10. Build DataSplit ───────────────────────────────────
    split = DataSplit(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        meta_train=meta_train, meta_val=meta_val, meta_test=meta_test,
        feature_means=feature_means,
        feature_stds=feature_stds,
        label_map=label_map,
        label_to_condition=label_to_condition,
    )

    # ── 11. Save artefacts ────────────────────────────────────
    if save_artefacts_dir is not None:
        _save_artefacts(split, save_artefacts_dir, train_idx, val_idx,
                        test_idx, verbose)

    if verbose:
        print(split.describe())

    return split


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _check_dataframe(df: pd.DataFrame, verbose: bool) -> None:
    """Run structural integrity checks on the raw DataFrame."""
    # Expected number of rows
    if len(df) != N_NUCLEI:
        print(
            f"[loader] WARNING: Expected {N_NUCLEI} rows, got {len(df)}. "
            "Proceeding with available data."
        )

    # Required metadata columns
    required_meta = ["particle_key", "o2"]
    missing_meta = [c for c in required_meta if c not in df.columns]
    if missing_meta:
        raise ValueError(
            f"Required metadata columns missing: {missing_meta}"
        )

    # No all-NaN columns in feature space
    nan_cols = [
        f for f in ALL_FEATURES
        if f in df.columns and df[f].isna().all()
    ]
    if nan_cols:
        raise ValueError(
            f"All-NaN feature columns found: {nan_cols[:5]}... "
            f"({len(nan_cols)} total)"
        )

    # NaN fraction per feature
    nan_fracs = {
        f: df[f].isna().mean()
        for f in ALL_FEATURES
        if f in df.columns and df[f].isna().any()
    }
    if nan_fracs:
        worst = max(nan_fracs, key=nan_fracs.get)
        print(
            f"[loader] WARNING: {len(nan_fracs)} features contain NaN. "
            f"Worst: {worst} ({nan_fracs[worst]:.1%}). "
            "Filling with column mean."
        )
        for f in nan_fracs:
            df[f].fillna(df[f].mean(), inplace=True)

    if verbose:
        print(f"[loader] DataFrame integrity checks passed.")


def _stratified_split(
    labels: np.ndarray,
    val_frac: float,
    test_frac: float,
    seed: int,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce stratified train / val / test index arrays.

    Strategy:
      1. Split off (val_frac + test_frac) as valtest, rest is train.
      2. Split valtest into val and test using proportional test_frac.
    """
    n = len(labels)
    valtest_frac = val_frac + test_frac

    splitter1 = StratifiedShuffleSplit(
        n_splits=1, test_size=valtest_frac, random_state=seed
    )
    train_idx, valtest_idx = next(splitter1.split(np.zeros(n), labels))

    # Second split on valtest
    relative_test_frac = test_frac / valtest_frac
    splitter2 = StratifiedShuffleSplit(
        n_splits=1, test_size=relative_test_frac, random_state=seed
    )
    valtest_labels = labels[valtest_idx]
    val_sub, test_sub = next(
        splitter2.split(np.zeros(len(valtest_idx)), valtest_labels)
    )
    val_idx  = valtest_idx[val_sub]
    test_idx = valtest_idx[test_sub]

    if verbose:
        print(
            f"[loader] Split sizes — "
            f"train: {len(train_idx)} ({len(train_idx)/n:.1%})  "
            f"val: {len(val_idx)} ({len(val_idx)/n:.1%})  "
            f"test: {len(test_idx)} ({len(test_idx)/n:.1%})"
        )

    # Verify stratification: all conditions present in each split
    for split_name, idx in [("train", train_idx),
                             ("val", val_idx),
                             ("test", test_idx)]:
        unique_in_split = np.unique(labels[idx])
        if len(unique_in_split) != N_CONDITIONS:
            print(
                f"[loader] WARNING: {split_name} split missing "
                f"{N_CONDITIONS - len(unique_in_split)} conditions."
            )
        else:
            if verbose:
                print(
                    f"[loader] {split_name}: all {N_CONDITIONS} conditions "
                    f"represented."
                )

    return train_idx, val_idx, test_idx


def _check_normalisation(
    X_train_norm: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    verbose: bool,
) -> None:
    """
    Verify that normalised training features have mean ≈ 0, std ≈ 1.
    Issues a warning if any feature deviates substantially.
    """
    actual_means = X_train_norm.mean(axis=0)
    actual_stds  = X_train_norm.std(axis=0, ddof=0)

    mean_max_abs = np.abs(actual_means).max()
    std_max_dev  = np.abs(actual_stds - 1.0).max()

    if mean_max_abs > 1e-6:
        worst_idx = np.abs(actual_means).argmax()
        print(
            f"[loader] WARNING: Normalised training mean not ~0 for "
            f"{ALL_FEATURES[worst_idx]} (mean={actual_means[worst_idx]:.6f})"
        )
    if std_max_dev > 1e-6:
        worst_idx = np.abs(actual_stds - 1.0).argmax()
        print(
            f"[loader] WARNING: Normalised training std not ~1 for "
            f"{ALL_FEATURES[worst_idx]} (std={actual_stds[worst_idx]:.6f})"
        )

    if verbose and mean_max_abs <= 1e-6 and std_max_dev <= 1e-6:
        print(
            f"[loader] Normalisation check passed: "
            f"max|mean|={mean_max_abs:.2e}, max|std-1|={std_max_dev:.2e}"
        )


def _save_artefacts(
    split: DataSplit,
    save_dir: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    verbose: bool,
) -> None:
    """Save normalisation statistics and split indices to disk."""
    import json

    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "feature_means.npy"), split.feature_means)
    np.save(os.path.join(save_dir, "feature_stds.npy"),  split.feature_stds)
    np.save(os.path.join(save_dir, "train_idx.npy"), train_idx)
    np.save(os.path.join(save_dir, "val_idx.npy"),   val_idx)
    np.save(os.path.join(save_dir, "test_idx.npy"),  test_idx)

    # Save label map as JSON-serialisable dict
    label_map_serialisable = {
        f"{k[0]}__{k[1]}": v
        for k, v in split.label_map.items()
    }
    with open(os.path.join(save_dir, "label_map.json"), "w") as f:
        json.dump(label_map_serialisable, f, indent=2)

    summary = {
        "n_train": len(train_idx),
        "n_val":   len(val_idx),
        "n_test":  len(test_idx),
        "n_conditions": len(split.label_map),
        "feature_dim": int(split.X_train.shape[1]),
        "feature_means_range": [
            float(split.feature_means.min()),
            float(split.feature_means.max()),
        ],
        "feature_stds_range": [
            float(split.feature_stds.min()),
            float(split.feature_stds.max()),
        ],
    }
    with open(os.path.join(save_dir, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"[loader] Artefacts saved to: {save_dir}")
