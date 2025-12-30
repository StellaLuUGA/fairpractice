# metrics.py
from __future__ import annotations
import numpy as np


# -----------------------------------------
# LAFT: Vector Distance Ratio (primary)
# -----------------------------------------
class LAFTMetrics:
    """
    Final metrics for *our* LAFT study (no FACTER baselines).

    Inputs:
      vector_rows: list[dict] from vector_extractor.py, each dict contains:
        - sample_id
        - movie_title
        - genre
        - v0, v1, v2  (numpy arrays)

    Outputs:
      1) Per-sample leakage:
         - VDR = ||v0 - v1||_2 / (||v0 - v2||_2 + eps)
         - logVDR = log(VDR)

      2) Proxy-specific leakage:
         - group by proxy = (movie_title, genre)
         - summarize mean/median logVDR

      3) Weighted aggregate leakage:
         - one scalar across proxies (weighted by n, sqrt(n), or uniform)
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def vdr(self, v0, v1, v2) -> float:
        print(v0.shape, v1.shape, v2.shape)
        """VDR(x) = ||v0-v1||_2 / (||v0-v2||_2 + eps)"""
        v0 = np.asarray(v0)
        v1 = np.asarray(v1)
        v2 = np.asarray(v2)
        d01 = np.linalg.norm(v0 - v1, ord=2)
        d02 = np.linalg.norm(v0 - v2, ord=2)
        return float(d01 / (d02 + self.eps))

    @staticmethod
    def log_vdr(vdr: float) -> float:
        """
        logVDR = log(VDR)
          = 0  -> equal distances
          < 0  -> v0 closer to v1 than v2 (stronger leakage signal)
          > 0  -> v0 closer to v2 than v1
        """
        return float(np.log(vdr))

    def per_sample_scores(self, vector_rows: list[dict]) -> list[dict]:
        """Compute VDR + logVDR for each sample."""
        out = []
        for s in vector_rows:
            score = self.vdr(s["v0"], s["v1"], s["v2"])
            out.append(
                {
                    "sample_id": s.get("sample_id"),
                    "movie_title": s.get("movie_title"),
                    "genre": s.get("genre"),
                    "vdr": score,
                    "log_vdr": self.log_vdr(score),
                }
            )
        return out

    # -----------------------------------------
    # Proxy-specific leakage (proxy = pair)
    # -----------------------------------------
    @staticmethod
    def proxy_specific_leakage(sample_scores: list[dict], min_samples: int = 5) -> list[dict]:
        """
        Group by proxy = (movie_title, genre).
        Summarize leakage using logVDR (recommended for aggregation).
        """
        groups: dict[tuple[str, str], list[float]] = {}
        for r in sample_scores:
            key = (r["movie_title"], r["genre"])
            groups.setdefault(key, []).append(float(r["log_vdr"]))

        proxies = []
        for (movie, genre), vals in groups.items():
            if len(vals) < min_samples:
                continue
            vals = np.asarray(vals, dtype=float)
            proxies.append(
                {
                    "movie_title": movie,
                    "genre": genre,
                    "n": int(len(vals)),
                    "mean_log_vdr": float(np.mean(vals)),
                    "median_log_vdr": float(np.median(vals)),
                }
            )

        # Sort: larger mean_log_vdr first (you can flip if you prefer "more negative = more leakage")
        proxies.sort(key=lambda x: x["mean_log_vdr"], reverse=True)
        return proxies

    # -----------------------------------------
    # Weighted aggregate leakage (global)
    # -----------------------------------------
    @staticmethod
    def weighted_aggregate_leakage(proxy_rows: list[dict], weight_mode: str = "n") -> float:
        """
        Produce one global scalar by aggregating proxy-level mean_log_vdr.

        weight_mode:
          - "n"       : w(p) = n_p (support)
          - "sqrt_n"  : w(p) = sqrt(n_p)
          - "uniform" : w(p) = 1
        """
        if not proxy_rows:
            return 0.0

        weights = []
        values = []
        for p in proxy_rows:
            n = float(p["n"])
            if weight_mode == "n":
                w = n
            elif weight_mode == "sqrt_n":
                w = np.sqrt(n)
            elif weight_mode == "uniform":
                w = 1.0
            else:
                raise ValueError("weight_mode must be one of: 'n', 'sqrt_n', 'uniform'")

            weights.append(w)
            values.append(float(p["mean_log_vdr"]))

        weights = np.asarray(weights, dtype=float)
        values = np.asarray(values, dtype=float)
        return float((weights * values).sum() / (weights.sum() + 1e-12))


# ---------------------------
# Convenience: LAFT-only evaluator
# ---------------------------
class Evaluator:
    """
    Minimal orchestrator for *our* LAFT-only study.

    You provide:
      - vector_rows: list of dicts containing v0/v1/v2 and movie_title/genre

    Returns:
      - sample_scores (per-sample VDR/logVDR)
      - proxy_rows (proxy-specific leakage by (movie_title, genre))
      - weighted_leakage (one scalar)
    """

    def __init__(self, min_proxy_samples: int = 5):
        self.min_proxy_samples = min_proxy_samples
        self.laft = LAFTMetrics()

    def evaluate(self, vector_rows, weight_mode: str = "n") -> dict:
        sample_scores = self.laft.per_sample_scores(vector_rows)
        proxy_rows = self.laft.proxy_specific_leakage(sample_scores, min_samples=self.min_proxy_samples)
        weighted_leakage = self.laft.weighted_aggregate_leakage(proxy_rows, weight_mode=weight_mode)

        return {
            "SampleScores": sample_scores,              # list[dict]
            "ProxySpecificLeakage": proxy_rows,         # list[dict], ranked
            "WeightedAggregateLeakage": weighted_leakage,  # float
        }
