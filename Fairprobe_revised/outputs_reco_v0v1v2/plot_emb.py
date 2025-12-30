# plot_emb.py
# Plot PCA-2D scatter for reco_emb_v0/v1/v2.csv (numeric dim_0..dim_* OR embedding column)
# Also prints duplicate/unique counts and (optional) jitter to reveal overlapping points.

import os
import ast
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _find_embedding_column(df: pd.DataFrame):
    candidates = ["embedding", "emb", "vector", "reco_emb", "reco_embedding"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _extract_embeddings(df: pd.DataFrame) -> np.ndarray:
    emb_col = _find_embedding_column(df)

    # Case 1: list-like column
    if emb_col is not None:
        vecs = []
        for x in df[emb_col].tolist():
            if isinstance(x, (list, tuple, np.ndarray)):
                v = np.asarray(x, dtype=float)
            else:
                v = np.asarray(ast.literal_eval(str(x)), dtype=float)
            vecs.append(v)
        return np.vstack(vecs)

    # Case 2: numeric columns (your reco_emb_v*.csv is like dim_0..dim_49)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] == 0:
        raise ValueError(
            "No embedding column found and no numeric columns found. "
            "Please check your CSV format."
        )
    return num_df.to_numpy(dtype=float)


def pca_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    W = Vt[:2].T  # [D,2]
    return X @ W  # [N,2]


def report_dups(name: str, X: np.ndarray) -> None:
    dfX = pd.DataFrame(X)
    n = dfX.shape[0]
    n_unique = dfX.drop_duplicates().shape[0]
    n_dup = n - n_unique
    print(f"[{name}] total={n}, unique={n_unique}, duplicated={n_dup}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="Folder containing reco_emb_v0.csv / reco_emb_v1.csv / reco_emb_v2.csv",
    )
    ap.add_argument(
        "--out_png",
        type=str,
        default=None,
        help="Output PNG path. Default: <in_dir>/reco_emb_pca_v0v1v2.png",
    )
    ap.add_argument(
        "--max_points",
        type=int,
        default=0,
        help="Downsample PER FILE if >0. Set 0 to keep ALL points.",
    )
    ap.add_argument(
        "--jitter",
        type=float,
        default=0.02,
        help="Gaussian jitter std added after PCA to reveal overlapping points. Set 0 to disable.",
    )
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    in_dir = args.in_dir
    v0_path = os.path.join(in_dir, "reco_emb_v0.csv")
    v1_path = os.path.join(in_dir, "reco_emb_v1.csv")
    v2_path = os.path.join(in_dir, "reco_emb_v2.csv")

    for p in [v0_path, v1_path, v2_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    df0 = pd.read_csv(v0_path)
    df1 = pd.read_csv(v1_path)
    df2 = pd.read_csv(v2_path)

    X0 = _extract_embeddings(df0)
    X1 = _extract_embeddings(df1)
    X2 = _extract_embeddings(df2)

    # Print duplicate diagnostics (this explains "sparse-looking" plots)
    report_dups("v0", X0)
    report_dups("v1", X1)
    report_dups("v2", X2)

    rng = np.random.default_rng(42)

    def _downsample(X, k: int):
        if k is None or k <= 0 or X.shape[0] <= k:
            return X
        idx = rng.choice(X.shape[0], size=k, replace=False)
        return X[idx]

    X0 = _downsample(X0, args.max_points)
    X1 = _downsample(X1, args.max_points)
    X2 = _downsample(X2, args.max_points)

    X_all = np.vstack([X0, X1, X2])
    Z_all = pca_2d(X_all)

    n0 = X0.shape[0]
    n1 = X1.shape[0]
    Z0 = Z_all[:n0]
    Z1 = Z_all[n0 : n0 + n1]
    Z2 = Z_all[n0 + n1 :]

    # Optional jitter to make overlapping points visible
    if args.jitter and args.jitter > 0:
        Z0 = Z0 + rng.normal(scale=args.jitter, size=Z0.shape)
        Z1 = Z1 + rng.normal(scale=args.jitter, size=Z1.shape)
        Z2 = Z2 + rng.normal(scale=args.jitter, size=Z2.shape)

    plt.figure(figsize=(8, 6))
    plt.scatter(Z0[:, 0], Z0[:, 1], s=25, alpha=0.35, label=f"v0 (n={n0})")
    plt.scatter(Z1[:, 0], Z1[:, 1], s=25, alpha=0.35, label=f"v1 (n={n1})")
    plt.scatter(Z2[:, 0], Z2[:, 1], s=25, alpha=0.35, label=f"v2 (n={Z2.shape[0]})")

    plt.title("Recommendation Embeddings (PCA 2D): v0 vs v1 vs v2")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()

    out_png = args.out_png or os.path.join(in_dir, "reco_emb_pca_v0v1v2.png")
    plt.savefig(out_png, dpi=args.dpi)
    plt.close()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

"""
[v0] total=200, unique=18, duplicated=182
[v1] total=200, unique=21, duplicated=179
[v2] total=200, unique=20, duplicated=180
"""
