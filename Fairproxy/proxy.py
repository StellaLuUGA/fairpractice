# proxy.py
"""
Proxy identification for study (Proxy = (movie_title, genre) pair)

pipeline:
1) llm_predictor.py produces, for each sample (x = movie_title, genre):
   - predicted_gender (â), confidence, male_prob/female_prob
   - ground_truth (a), correct, sample_id, movie_title, genre

2) vector_extractor.py produces, for the same sample_id:
   - v0 = p'(y | x)
   - v1 = p'(y | x, a = â)
   - v2 = p'(y | x, a != â)

Proxy strength per sample:
  ratio(x) = ||v0 - v1||_2 / (||v0 - v2||_2 + eps)

Proxy P samples ( definition):
  high-confidence AND high-accuracy outcomes
  -> (confidence >= conf_threshold) AND (correct == True)

This file:
- merges predictions + vectors by sample_id
- computes ratio per sample
- filters Proxy P
- aggregates by proxy pair (movie_title, genre)
"""

import numpy as np
import pandas as pd


class ProxyAnalyzer:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    # --------------------------
    # Core math
    # --------------------------
    @staticmethod
    def _to_np(v):
        """Ensure vectors are numpy arrays."""
        if isinstance(v, np.ndarray):
            return v
        # vector_extractor returns numpy arrays, but be safe
        return np.array(v)

    def _l2(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b, ord=2))

    # --------------------------
    # Merge predictor + extractor outputs
    # --------------------------
    def build_frame(self, predictions: list, vector_rows: list) -> pd.DataFrame:
        """
        predictions: output of LLaMAGenderPredictor.predict_batch(...)
        vector_rows: output of VectorExtractor.extract_batch(predictions)

        Must contain:
          predictions: sample_id, movie_title, genre, predicted_gender, confidence, ground_truth, correct
          vector_rows: sample_id, v0, v1, v2
        """
        pred_df = pd.DataFrame(predictions)
        vec_df = pd.DataFrame(vector_rows)

        required_pred = {"sample_id", "movie_title", "genre", "predicted_gender", "confidence", "ground_truth"}
        missing_pred = required_pred - set(pred_df.columns)
        if missing_pred:
            raise ValueError(f"predictions missing columns: {missing_pred}")

        required_vec = {"sample_id", "v0", "v1", "v2"}
        missing_vec = required_vec - set(vec_df.columns)
        if missing_vec:
            raise ValueError(f"vector_rows missing columns: {missing_vec}")

        # Ensure correct exists; if not, compute it
        if "correct" not in pred_df.columns:
            pred_df["correct"] = pred_df["predicted_gender"] == pred_df["ground_truth"]

        # Merge on sample_id
        df = pred_df.merge(
            vec_df[["sample_id", "v0", "v1", "v2"]],
            on="sample_id",
            how="inner",
        )

        if df.empty:
            raise ValueError("After merging predictions and vector_rows on sample_id, dataframe is empty.")

        # Compute distances + ratio per sample
        dist01, dist02, ratio = [], [], []
        for _, r in df.iterrows():
            v0 = self._to_np(r["v0"])
            v1 = self._to_np(r["v1"])
            v2 = self._to_np(r["v2"])

            d01 = self._l2(v0, v1)
            d02 = self._l2(v0, v2)

            dist01.append(d01)
            dist02.append(d02)
            ratio.append(d01 / (d02 + self.eps))

        df["dist_v0_v1"] = dist01
        df["dist_v0_v2"] = dist02
        df["proxy_ratio"] = ratio  # your score

        return df

    # --------------------------
    # Proxy P filtering
    # --------------------------
    @staticmethod
    def mark_proxy_p(df: pd.DataFrame, conf_threshold: float = 0.8) -> pd.DataFrame:
        """
        Proxy P = high-confidence AND high-accuracy outcomes (your definition).
        Here, high-accuracy is per-sample correctness (correct == True).
        """
        out = df.copy()
        out["is_proxy_p"] = (out["confidence"] >= conf_threshold) & (out["correct"] == True)
        return out

    # --------------------------
    # Aggregate by proxy pair (movie_title, genre)
    # --------------------------
    def summarize_by_pair(
        self,
        df: pd.DataFrame,
        use_proxy_p_only: bool = True,
        min_samples: int = 5,
    ) -> pd.DataFrame:
        """
        Group by proxy = (movie_title, genre) and compute:
          - n samples
          - accuracy (mean correct)
          - mean confidence
          - mean/median proxy_ratio
          - mean distances

        If use_proxy_p_only=True, only uses rows where is_proxy_p == True.
        """
        work = df.copy()
        if use_proxy_p_only:
            if "is_proxy_p" not in work.columns:
                raise ValueError("Dataframe missing is_proxy_p. Call mark_proxy_p(df) first.")
            work = work[work["is_proxy_p"] == True]

        if work.empty:
            # Return an empty summary with expected columns
            return pd.DataFrame(
                columns=[
                    "movie_title", "genre", "n",
                    "accuracy", "mean_confidence",
                    "mean_proxy_ratio", "median_proxy_ratio",
                    "mean_dist_v0_v1", "mean_dist_v0_v2",
                ]
            )

        agg = (
            work.groupby(["movie_title", "genre"])
            .agg(
                n=("sample_id", "count"),
                accuracy=("correct", "mean"),
                mean_confidence=("confidence", "mean"),
                mean_proxy_ratio=("proxy_ratio", "mean"),
                median_proxy_ratio=("proxy_ratio", "median"),
                mean_dist_v0_v1=("dist_v0_v1", "mean"),
                mean_dist_v0_v2=("dist_v0_v2", "mean"),
            )
            .reset_index()
        )

        # Filter out tiny groups (unstable estimates)
        agg = agg[agg["n"] >= min_samples].reset_index(drop=True)

        # Rank by your proxy strength
        agg = agg.sort_values("mean_proxy_ratio", ascending=False).reset_index(drop=True)
        return agg

    # --------------------------
    # One-call convenience
    # --------------------------
    def run(
        self,
        predictions: list,
        vector_rows: list,
        conf_threshold: float = 0.8,
        min_samples: int = 5,
    ):
        """
        Returns:
          - sample_level_df: merged per-sample table with proxy_ratio
          - proxy_p_summary: ranked proxy pairs using Proxy P samples only
          - all_summary: ranked proxy pairs using all merged samples (optional baseline)
        """
        sample_df = self.build_frame(predictions, vector_rows)
        sample_df = self.mark_proxy_p(sample_df, conf_threshold=conf_threshold)

        proxy_p_summary = self.summarize_by_pair(
            sample_df, use_proxy_p_only=True, min_samples=min_samples
        )
        all_summary = self.summarize_by_pair(
            sample_df, use_proxy_p_only=False, min_samples=min_samples
        )

        return sample_df, proxy_p_summary, all_summary
