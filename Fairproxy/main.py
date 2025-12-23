# main.py
"""Main experiment runner (LAFT-only, final)"""

import json
from pathlib import Path

import pandas as pd

from config import Config
from data import load_data
from llm_predictor import LLaMAGenderPredictor
from vector_extractor import VectorExtractor
from proxy import ProxyAnalyzer
from metrics import LAFTMetrics


def _limit_list(lst, n):
    """Return lst[:n] if n is a positive int; otherwise return lst unchanged."""
    if n is None:
        return lst
    try:
        n = int(n)
    except Exception:
        return lst
    return lst[:n] if n > 0 else lst


def main():
    print("=" * 60)
    print("LAFT: Leakage-Aware Fairness Testing (LAFT-only)")
    print("=" * 60)

    config = Config()

    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ========== STEP 1: Load Data ==========
    print("\n[1/4] Loading MovieLens-1M dataset...")
    data, item_db = load_data(config)
    print(f"Loaded {len(data)} rows")

    # ========== STEP 2: Gender Prediction ==========
    print("\n[2/4] Predicting gender from (movie_title, genre)...")
    predictor = LLaMAGenderPredictor(model_name=config.MODEL_NAME, device=config.DEVICE)

    # predict_batch filters by confidence threshold internally
    predictions = predictor.predict_batch(data, confidence_threshold=config.CONFIDENCE_THRESHOLD)

    # Optional cap
    predictions = _limit_list(predictions, getattr(config, "N_SAMPLES", None))

    print(f"Predictions kept: {len(predictions)}")
    if len(predictions) == 0:
        print("No predictions passed the confidence threshold. Exiting.")
        return

    # ========== STEP 3: Vector Extraction ==========
    print("\n[3/4] Extracting internal vectors (v0, v1, v2)...")
    extractor = VectorExtractor(
        model=predictor.model,
        tokenizer=predictor.tokenizer,
        layer_idx=config.LAYER_IDX,
        device=config.DEVICE,
    )
    vector_rows = extractor.extract_batch(predictions)
    print(f"Extracted vectors for {len(vector_rows)} samples")

    # ========== STEP 4: LAFT Metrics + Proxy Summaries ==========
    print("\n[4/4] Computing LAFT metrics + proxy leakage (proxy = movie_title + genre)...")

    laft = LAFTMetrics()

    # Per-sample LAFT
    sample_scores = laft.per_sample_scores(vector_rows)
    df_samples = pd.DataFrame(sample_scores)

    # Proxy-specific leakage (proxy = (movie_title, genre))
    min_proxy_samples = getattr(config, "MIN_PROXY_SAMPLES", 5)
    proxy_rows = laft.proxy_specific_leakage(sample_scores, min_samples=min_proxy_samples)
    df_proxy = pd.DataFrame(proxy_rows)

    # Weighted aggregate leakage (one scalar)
    weight_mode = getattr(config, "PROXY_WEIGHT_MODE", "n")
    weighted_leakage = laft.weighted_aggregate_leakage(proxy_rows, weight_mode=weight_mode)

    # Optional: extra proxy.py diagnostics (Proxy-P list, etc.)
    # This is NOT required for LAFT metrics, but can help you debug/inspect proxies.
    proxyP_num_pairs = None
    try:
        proxy_analyzer = ProxyAnalyzer()
        proxy_sample_df, proxyP_pairs_df, all_pairs_df = proxy_analyzer.run(
            predictions=predictions,
            vector_rows=vector_rows,
            conf_threshold=config.CONFIDENCE_THRESHOLD,
            min_samples=min_proxy_samples,
        )
        proxy_sample_df.to_csv(results_dir / "proxy_sample_level.csv", index=False)
        proxyP_pairs_df.to_csv(results_dir / "proxy_pairs_proxyP_only.csv", index=False)
        all_pairs_df.to_csv(results_dir / "proxy_pairs_all_samples.csv", index=False)
        proxyP_num_pairs = int(len(proxyP_pairs_df))
    except Exception as e:
        print(f"(Optional) proxy.py analysis skipped due to error: {e}")

    # Summary
    summary = {
        "model_name": str(config.MODEL_NAME),
        "layer_idx": int(config.LAYER_IDX),
        "device": str(config.DEVICE),
        "confidence_threshold": float(config.CONFIDENCE_THRESHOLD),
        "n_samples": int(len(df_samples)),
        "mean_logVDR": float(df_samples["log_vdr"].mean()) if not df_samples.empty else 0.0,
        "median_logVDR": float(df_samples["log_vdr"].median()) if not df_samples.empty else 0.0,
        "WeightedAggregateLeakage": float(weighted_leakage),
        "num_proxies_reported": int(len(df_proxy)),
        "min_proxy_samples": int(min_proxy_samples),
        "proxy_weight_mode": str(weight_mode),
    }
    if proxyP_num_pairs is not None:
        summary["proxyP_num_pairs"] = proxyP_num_pairs

    # Save
    df_samples.to_csv(results_dir / "laft_sample_scores.csv", index=False)
    df_proxy.to_csv(results_dir / "proxy_leakage_by_pair.csv", index=False)
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (LAFT-only)")
    print("=" * 60)
    print(f"Samples evaluated: {summary['n_samples']}")
    print(f"Mean logVDR:       {summary['mean_logVDR']:.4f}")
    print(f"Median logVDR:     {summary['median_logVDR']:.4f}")
    print(f"Weighted leakage:  {summary['WeightedAggregateLeakage']:.4f}")
    print(f"Proxy pairs saved: {summary['num_proxies_reported']}")
    if proxyP_num_pairs is not None:
        print(f"Proxy-P pairs:     {summary['proxyP_num_pairs']}")
    print(f"\nSaved to: {results_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
