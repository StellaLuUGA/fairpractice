# main.py
"""Main experiment runner (LAFT-only + Output Sensitivity Test + 2 histogram PNGs)"""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from config import Config
from data import load_data
from llm_predictor import LLaMAGenderPredictor
from vector_extractor import VectorExtractor
from proxy import ProxyAnalyzer
from metrics import LAFTMetrics

# NEW: output sensitivity
from output_sensitivity import OutputSensitivityTester, OSTConfig


def _limit_list(lst, n):
    """Return lst[:n] if n is a positive int; otherwise return lst unchanged."""
    if n is None:
        return lst
    try:
        n = int(n)
    except Exception:
        return lst
    return lst[:n] if n > 0 else lst


def _save_histogram_single_line_png(
    df_samples: pd.DataFrame,
    out_png: Path,
    line_value: float,
    line_label: str,
    title: str,
    bins: int = 50,
    linestyle: str = "--",
    linewidth: int = 3,
    dpi: int = 200,
):
    """Save ONE histogram PNG with ONE vertical line (either mean or median)."""
    if df_samples.empty or "log_vdr" not in df_samples.columns:
        print("(Plot) Skipped histogram: df_samples is empty or missing 'log_vdr'.")
        return

    x = df_samples["log_vdr"].astype(float).values

    plt.figure()
    plt.hist(x, bins=bins)
    plt.axvline(line_value, linestyle=linestyle, linewidth=linewidth, label=line_label)

    plt.xlabel("logVDR per sample")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_png, dpi=dpi)
    plt.close()
    print(f"(Plot) Saved: {out_png}")


def main():
    print("=" * 60)
    print("LAFT: Leakage-Aware Fairness Testing (LAFT-only + OST)")
    print("=" * 60)

    config = Config()

    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ========== STEP 1: Load Data ==========
    print("\n[1/5] Loading MovieLens-1M dataset...")
    data, item_db = load_data(config)
    print(f"Loaded {len(data)} rows")

    # ========== STEP 2: Gender Prediction ==========
    print("\n[2/5] Predicting gender from (movie_title, genre)...")
    predictor = LLaMAGenderPredictor(model_name=config.MODEL_NAME, device=config.DEVICE)

    predictions = predictor.predict_batch(data, confidence_threshold=config.CONFIDENCE_THRESHOLD)

    # Optional cap
    predictions = _limit_list(predictions, getattr(config, "N_SAMPLES", None))

    print(f"Predictions kept: {len(predictions)}")
    if len(predictions) == 0:
        print("No predictions passed the confidence threshold. Exiting.")
        return

    # ========== STEP 3: Vector Extraction ==========
    print("\n[3/5] Extracting internal vectors (v0, v1, v2)...")
    extractor = VectorExtractor(
        model=predictor.model,
        tokenizer=predictor.tokenizer,
        layer_idx=config.LAYER_IDX,
        device=config.DEVICE,
    )
    vector_rows = extractor.extract_batch(predictions)
    print(f"Extracted vectors for {len(vector_rows)} samples")

    # ========== STEP 4: LAFT Metrics + Proxy Summaries ==========
    print("\n[4/5] Computing LAFT metrics + proxy leakage (proxy = movie_title + genre)...")

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

    # Optional: extra proxy.py diagnostics
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

    # ---- histograms for logVDR (2 PNGs) ----
    mean_logvdr = float(df_samples["log_vdr"].mean()) if not df_samples.empty else 0.0
    median_logvdr = float(df_samples["log_vdr"].median()) if not df_samples.empty else 0.0

    _save_histogram_single_line_png(
        df_samples=df_samples,
        out_png=results_dir / "logvdr_hist_mean.png",
        line_value=mean_logvdr,
        line_label=f"Mean = {mean_logvdr:.4f}",
        title=f"logVDR Histogram (Mean)  n={len(df_samples)}",
        linestyle="--",
        linewidth=3,
    )

    _save_histogram_single_line_png(
        df_samples=df_samples,
        out_png=results_dir / "logvdr_hist_median.png",
        line_value=median_logvdr,
        line_label=f"Median = {median_logvdr:.4f}",
        title=f"logVDR Histogram (Median)  n={len(df_samples)}",
        linestyle="-",
        linewidth=3,
    )

    # ========== STEP 5: Output Sensitivity Test (OST) ==========
    print("\n[5/5] Running Output Sensitivity Test (y0 vs yM vs yF) per proxy...")

    if df_proxy.empty:
        print("No proxy pairs available from LAFT (df_proxy empty). Skipping OST.")
        df_ost_proxy = pd.DataFrame()
        df_proxy_laft_ost = pd.DataFrame()
        ost_summary = {"ran": False, "reason": "df_proxy empty"}
    else:
        # Run OST only ONCE per proxy (movie_title, genre) to save compute.
        # Optionally cap the number of proxies evaluated.
        ost_topk = getattr(config, "OST_TOPK_PROXIES", None)  # None => run all proxies
        df_proxy_for_ost = df_proxy.copy()
        if ost_topk is not None:
            try:
                ost_topk = int(ost_topk)
                if ost_topk > 0:
                    df_proxy_for_ost = df_proxy_for_ost.head(ost_topk)
            except Exception:
                pass

        proxy_inputs = []
        for _, r in df_proxy_for_ost.iterrows():
            proxy_inputs.append(
                {
                    "movie_title": r["movie_title"],
                    "genre": r["genre"],
                    "n_support": int(r.get("n", 0)),  # keep LAFT support
                }
            )

        ost_cfg = OSTConfig(
            max_new_tokens=int(getattr(config, "OST_MAX_NEW_TOKENS", 32)),
            do_sample=bool(getattr(config, "OST_DO_SAMPLE", False)),
            temperature=float(getattr(config, "OST_TEMPERATURE", 0.0)),
            top_p=float(getattr(config, "OST_TOP_P", 0.95)),
            one_line_only=bool(getattr(config, "OST_ONE_LINE_ONLY", True)),
        )

        tester = OutputSensitivityTester(
            model=predictor.model,
            tokenizer=predictor.tokenizer,
            device=config.DEVICE,
            cfg=ost_cfg,
            embedder=None,  # keep simple; you can pass sentence-transformers later
        )

        ost_rows = tester.run_batch(proxy_inputs)
        df_ost_proxy = pd.DataFrame(ost_rows)

        # Merge LAFT proxy leakage + OST deltas (proxy-level “hard” evidence)
        # df_proxy has: movie_title, genre, n, mean_log_vdr, median_log_vdr
        df_proxy_laft_ost = df_proxy_for_ost.merge(
            df_ost_proxy,
            on=["movie_title", "genre"],
            how="left",
            suffixes=("_laft", "_ost"),
        )

        # Simple “combined evidence” counters you can report to your professor:
        # e.g., large |mean_log_vdr| AND large delta_MF_edit
        leak_thr = float(getattr(config, "OST_LEAK_THR", 0.05))      # |mean_log_vdr| threshold
        out_thr = float(getattr(config, "OST_OUT_THR_EDIT", 0.30))   # edit-distance threshold

        df_proxy_laft_ost["flag_high_leak"] = df_proxy_laft_ost["mean_log_vdr"].abs() >= leak_thr
        df_proxy_laft_ost["flag_high_out"] = df_proxy_laft_ost["delta_MF_edit"].fillna(0.0) >= out_thr
        df_proxy_laft_ost["flag_both"] = df_proxy_laft_ost["flag_high_leak"] & df_proxy_laft_ost["flag_high_out"]

        n_proxy_eval = int(len(df_proxy_laft_ost))
        n_both = int(df_proxy_laft_ost["flag_both"].sum()) if n_proxy_eval else 0

        ost_summary = {
            "ran": True,
            "n_proxies_tested": n_proxy_eval,
            "leak_thr_abs_mean_logvdr": leak_thr,
            "out_thr_delta_MF_edit": out_thr,
            "n_both_internal_and_output": n_both,
            "pct_both_internal_and_output": (n_both / n_proxy_eval) if n_proxy_eval else 0.0,
        }

        # Save OST outputs
        df_ost_proxy.to_csv(results_dir / "output_sensitivity_proxy_level.csv", index=False)
        df_proxy_laft_ost.to_csv(results_dir / "laft_proxy_join_ost.csv", index=False)

        # Print a few “strongest” proxies by output sensitivity
        show_k = int(getattr(config, "OST_SHOW_TOPK", 10))
        if n_proxy_eval:
            top_out = df_proxy_laft_ost.sort_values(
                by=["delta_MF_edit", "delta_MF_jaccard"],
                ascending=[False, False],
            ).head(show_k)

            print("\nTop proxies by output sensitivity (delta_MF_edit):")
            for _, rr in top_out.iterrows():
                print(
                    f"- ({rr['movie_title']} | {rr['genre']}) "
                    f"delta_MF_edit={rr.get('delta_MF_edit', None):.3f} "
                    f"mean_log_vdr={rr.get('mean_log_vdr', None):.3f}"
                )

    # ---------- Global summary ----------
    summary = {
        "model_name": str(config.MODEL_NAME),
        "layer_idx": int(config.LAYER_IDX),
        "device": str(config.DEVICE),
        "confidence_threshold": float(config.CONFIDENCE_THRESHOLD),
        "n_samples": int(len(df_samples)),
        "mean_logVDR": mean_logvdr,
        "median_logVDR": median_logvdr,
        "WeightedAggregateLeakage": float(weighted_leakage),
        "num_proxies_reported": int(len(df_proxy)),
        "min_proxy_samples": int(min_proxy_samples),
        "proxy_weight_mode": str(weight_mode),
        "OST": ost_summary,
    }
    if proxyP_num_pairs is not None:
        summary["proxyP_num_pairs"] = proxyP_num_pairs

    # Save core LAFT files
    df_samples.to_csv(results_dir / "laft_sample_scores.csv", index=False)
    df_proxy.to_csv(results_dir / "proxy_leakage_by_pair.csv", index=False)

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (LAFT + OST)")
    print("=" * 60)
    print(f"Samples evaluated: {summary['n_samples']}")
    print(f"Mean logVDR:       {summary['mean_logVDR']:.4f}")
    print(f"Median logVDR:     {summary['median_logVDR']:.4f}")
    print(f"Weighted leakage:  {summary['WeightedAggregateLeakage']:.4f}")
    print(f"Proxy pairs saved: {summary['num_proxies_reported']}")
    if proxyP_num_pairs is not None:
        print(f"Proxy-P pairs:     {summary['proxyP_num_pairs']}")

    if summary["OST"]["ran"]:
        print("\nOutput Sensitivity Test (proxy-level):")
        print(f"  Proxies tested:  {summary['OST']['n_proxies_tested']}")
        print(
            f"  BOTH (|mean_log_vdr|>={summary['OST']['leak_thr_abs_mean_logvdr']}, "
            f"delta_MF_edit>={summary['OST']['out_thr_delta_MF_edit']}): "
            f"{summary['OST']['n_both_internal_and_output']} "
            f"({summary['OST']['pct_both_internal_and_output']*100:.1f}%)"
        )
        print("  Saved: output_sensitivity_proxy_level.csv, laft_proxy_join_ost.csv")
    else:
        print("\nOutput Sensitivity Test (OST): skipped.")

    print(f"\nSaved to: {results_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
