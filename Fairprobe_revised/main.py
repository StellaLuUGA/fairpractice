# main.py
# Requirements covered:
# 1) hist of VDR and log-VDR  (from v0/v1/v2 vectors)
# 2) Output similarity        (NO jaccard, NO edit_distance_ratio)
#    -> we use a simple "same-title vs different-title" distance:
#       distance = 0 if outputs match (case-insensitive), else 1
# 3) Movie–movie distance     (cosine distance between v0 movie vectors)
#    - within the same genre vs across different genres
#
# Important fixes included:
# - --model_name is OPTIONAL (has a default) so you won't get "required: --model_name"
# - Ground-truth gender normalization (M/F -> Male/Female) so predictor accuracy won't be 0.000 due to string mismatch
# - If tokenizer has chat_template but jinja2 is too old (<3.1.0), we disable chat_template to avoid ImportError

import os
import argparse
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from data import load_data, build_user_movie_df
from llm_predictor import LLaMAGenderPredictor
from vector_extractor import VectorExtractor, ExtractConfig


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(1.0 - (np.dot(a, b) / denom))


def plot_hist(values: np.ndarray, title: str, outpath: str, bins: int = 30) -> None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def output_match_distance(a: str, b: str) -> float:
    """
    Output similarity WITHOUT edit distance:
      0.0 if the two generated titles match (case-insensitive, stripped)
      1.0 otherwise
    """
    a = "" if a is None else str(a).strip().lower()
    b = "" if b is None else str(b).strip().lower()
    return 0.0 if a == b and a != "" else 1.0


def normalize_gender_str(g: Any) -> str:
    """
    Normalize ground truth from MovieLens users.dat:
      "M"/"F" -> "Male"/"Female"
      already "Male"/"Female" stays
    """
    if g is None:
        return ""
    s = str(g).strip().lower()
    if s in ("m", "male"):
        return "Male"
    if s in ("f", "female"):
        return "Female"
    return str(g).strip()


def maybe_disable_chat_template(tokenizer) -> None:
    """
    If tokenizer.chat_template exists but jinja2<3.1.0, apply_chat_template will crash.
    We disable chat_template to force llm_predictor fallback prompt formatting.
    """
    if not getattr(tokenizer, "chat_template", None):
        return
    try:
        import jinja2  # type: ignore
        from packaging.version import Version  # type: ignore

        if Version(jinja2.__version__) < Version("3.1.0"):
            print(f"[warn] jinja2={jinja2.__version__} < 3.1.0; disabling tokenizer.chat_template to avoid ImportError.")
            tokenizer.chat_template = None
    except Exception:
        # If jinja2 or packaging isn't available, safest is to disable chat_template.
        print("[warn] Could not verify jinja2 version; disabling tokenizer.chat_template to avoid apply_chat_template issues.")
        tokenizer.chat_template = None


@torch.inference_mode()
def generate_one_title(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 16,
) -> str:
    """
    Generate a short continuation and return ONLY the generated part (not the prompt).
    Intended for prompts like: "... recommend 1 movie title only:"
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    prompt_len = int(inputs["input_ids"].shape[1])

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = gen_ids[0, prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # light cleanup
    text = text.strip().split("\n")[0].strip()
    text = text.strip('"').strip("'").strip()
    return text


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # Make model_name OPTIONAL (default provided) so you won't see "required: --model_name"
    ap.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default=None, help="float16|bfloat16|float32 or None")
    ap.add_argument("--seed", type=int, default=42)

    # cost controls
    ap.add_argument("--confidence_threshold", type=float, default=0.80)
    ap.add_argument("--max_pred_rows", type=int, default=400, help="max rows for gender prediction")
    ap.add_argument("--max_vec_samples", type=int, default=80, help="max high-conf samples for v0/v1/v2 extraction")

    # vector extraction config
    ap.add_argument("--layer_idx", type=int, default=-2)
    ap.add_argument("--layer_select", type=str, default="hidden_state", choices=["hidden_state", "self_attention", "MLP"])
    ap.add_argument("--max_new_tokens", type=int, default=32, help="generation cap used inside vector_extractor")
    ap.add_argument("--rec_max_new_tokens", type=int, default=16, help="generation cap for recommendation titles")
    ap.add_argument("--debug_hook", action="store_true")

    ap.add_argument("--out_dir", type=str, default="outputs")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    # 1) Load data and build predictor input df
    movies, ratings, users, target_set = load_data()
    df_um = build_user_movie_df(movies, ratings, users, target_set)

    # Normalize ground-truth gender to avoid "Accuracy: 0.000" due to M/F vs Male/Female mismatch
    df_um = df_um.copy()
    df_um["gender"] = df_um["gender"].apply(normalize_gender_str)

    # sample to control cost
    if len(df_um) > args.max_pred_rows:
        df_pred = df_um.sample(n=args.max_pred_rows, random_state=args.seed).reset_index(drop=True)
    else:
        df_pred = df_um.reset_index(drop=True)

    # 2) Load LLaMA predictor (loads model+tokenizer)
    predictor = LLaMAGenderPredictor(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
    )

    # If chat_template exists but jinja2 is too old, disable it to avoid ImportError
    maybe_disable_chat_template(predictor.tokenizer)

    # 3) High-confidence gender predictions
    preds = predictor.predict_batch(df_pred, confidence_threshold=float(args.confidence_threshold))
    if len(preds) == 0:
        print("No high-confidence predictions. Lower --confidence_threshold or increase --max_pred_rows.")
        return

    # cap vector extraction count
    if len(preds) > args.max_vec_samples:
        preds = preds[: args.max_vec_samples]

    # 4) Vector extractor (reuse predictor.model/tokenizer)
    cfg = ExtractConfig(
        layer_idx=int(args.layer_idx),
        layer_select=str(args.layer_select),
        extract_mode="last_gen",   # current VDR uses generated-token hidden states
        max_new_tokens=int(args.max_new_tokens),
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        debug_hook=bool(args.debug_hook),
    )

    extractor = VectorExtractor(
        model=predictor.model,
        tokenizer=predictor.tokenizer,
        device=args.device,
        cfg=cfg,
    )

    # 5) Extract v0/v1/v2 per sample
    vec_rows = extractor.extract_batch(preds)

    # -----------------------------
    # A) VDR + log-VDR (per sample)
    # -----------------------------
    eps = 1e-12
    records: List[Dict[str, Any]] = []
    for r in vec_rows:
        v0 = r["v0"]
        v1 = r["v1"]
        v2 = r["v2"]

        d01 = cosine_distance(v0, v1, eps=eps)
        d02 = cosine_distance(v0, v2, eps=eps)
        vdr = float((d02 + eps) / (d01 + eps))
        log_vdr = float(np.log(vdr + eps))

        records.append(
            {
                "sample_id": r.get("sample_id"),
                "movie_title": r.get("movie_title"),
                "genre": r.get("genre"),
                "ground_truth": r.get("ground_truth"),
                "predicted_gender": r.get("predicted_gender"),
                "correct": r.get("correct"),
                "a_equals_ahat": r.get("a_equals_ahat"),
                "d_v0_v1": d01,
                "d_v0_v2": d02,
                "VDR": vdr,
                "log_VDR": log_vdr,
                "hidden_dim": r.get("hidden_dim"),
                "layer_idx": int(args.layer_idx),
                "layer_select": str(args.layer_select),
            }
        )

    df_vdr = pd.DataFrame(records)
    df_vdr.to_csv(os.path.join(args.out_dir, "vdr_per_sample.csv"), index=False)
    plot_hist(df_vdr["VDR"].to_numpy(), "Histogram of VDR", os.path.join(args.out_dir, "hist_VDR.png"))
    plot_hist(df_vdr["log_VDR"].to_numpy(), "Histogram of log(VDR)", os.path.join(args.out_dir, "hist_logVDR.png"))

    # -----------------------------
    # B) Movie–movie distance (v0 movie vectors) within vs cross genres
    # -----------------------------
    movies_unique = df_um[["clean_title", "main_genre"]].drop_duplicates().reset_index(drop=True)

    movie_reps: Dict[str, Dict[str, Any]] = {}
    for _, row in movies_unique.iterrows():
        title = str(row["clean_title"])
        g = str(row["main_genre"])
        prompt = extractor._prompt_no_gender(title, g)  # gender={neutral}
        v0_vec, meta = extractor._extract_hidden_vector(prompt)  # uses extract_mode="last_gen"
        movie_reps[title] = {"genre": g, "v0": v0_vec, "meta": meta}

    titles = list(movie_reps.keys())
    within_dists: List[float] = []
    cross_dists: List[float] = []

    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            ti, tj = titles[i], titles[j]
            gi, gj = movie_reps[ti]["genre"], movie_reps[tj]["genre"]
            d = cosine_distance(movie_reps[ti]["v0"], movie_reps[tj]["v0"], eps=eps)
            if gi == gj:
                within_dists.append(d)
            else:
                cross_dists.append(d)

    within_arr = np.array(within_dists, dtype=np.float64)
    cross_arr = np.array(cross_dists, dtype=np.float64)

    pd.DataFrame({"cosine_distance": within_arr}).to_csv(
        os.path.join(args.out_dir, "movie_dist_within_genre.csv"), index=False
    )
    pd.DataFrame({"cosine_distance": cross_arr}).to_csv(
        os.path.join(args.out_dir, "movie_dist_cross_genre.csv"), index=False
    )

    plot_hist(
        within_arr,
        "Movie–movie cosine distance (within genre) [v0]",
        os.path.join(args.out_dir, "hist_movie_within_genre.png"),
    )
    plot_hist(
        cross_arr,
        "Movie–movie cosine distance (cross genre) [v0]",
        os.path.join(args.out_dir, "hist_movie_cross_genre.png"),
    )

    # -----------------------------
    # C) Output similarity (match-distance: 0 if same title else 1)
    # -----------------------------
    movie_outputs: Dict[str, str] = {}
    for title, info in movie_reps.items():
        g = info["genre"]
        prompt = extractor._prompt_no_gender(title, g)
        rec = generate_one_title(
            model=predictor.model,
            tokenizer=predictor.tokenizer,
            device=extractor.device,
            prompt=prompt,
            max_new_tokens=int(args.rec_max_new_tokens),
        )
        movie_outputs[title] = rec

    within_out: List[float] = []
    cross_out: List[float] = []

    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            ti, tj = titles[i], titles[j]
            gi, gj = movie_reps[ti]["genre"], movie_reps[tj]["genre"]
            oi, oj = movie_outputs.get(ti, ""), movie_outputs.get(tj, "")
            d = output_match_distance(oi, oj)
            if gi == gj:
                within_out.append(d)
            else:
                cross_out.append(d)

    within_out_arr = np.array(within_out, dtype=np.float64)
    cross_out_arr = np.array(cross_out, dtype=np.float64)

    pd.DataFrame(
        [{"movie_title": t, "genre": movie_reps[t]["genre"], "v0_recommendation": movie_outputs[t]} for t in titles]
    ).to_csv(os.path.join(args.out_dir, "v0_recommendations_per_movie.csv"), index=False)

    pd.DataFrame({"match_distance": within_out_arr}).to_csv(
        os.path.join(args.out_dir, "output_match_within_genre.csv"), index=False
    )
    pd.DataFrame({"match_distance": cross_out_arr}).to_csv(
        os.path.join(args.out_dir, "output_match_cross_genre.csv"), index=False
    )

    plot_hist(
        within_out_arr,
        "Output match distance (within genre) [0=same,1=diff]",
        os.path.join(args.out_dir, "hist_output_match_within.png"),
    )
    plot_hist(
        cross_out_arr,
        "Output match distance (cross genre) [0=same,1=diff]",
        os.path.join(args.out_dir, "hist_output_match_cross.png"),
    )

    print(f"\nSaved all outputs to: {args.out_dir}\n")

    # Summaries
    print("Movie distance summary (cosine on v0):")
    if len(within_arr) > 0:
        print(f"  within-genre: n={len(within_arr)} mean={within_arr.mean():.4f} median={np.median(within_arr):.4f}")
    else:
        print("  within-genre: n=0 (not enough same-genre pairs)")
    if len(cross_arr) > 0:
        print(f"  cross-genre : n={len(cross_arr)} mean={cross_arr.mean():.4f} median={np.median(cross_arr):.4f}")
    else:
        print("  cross-genre : n=0 (not enough cross-genre pairs)")

    print("\nOutput similarity summary (match-distance; 0=same title, 1=different title):")
    if len(within_out_arr) > 0:
        print(f"  within-genre: n={len(within_out_arr)} frac_different={within_out_arr.mean():.4f}")
    else:
        print("  within-genre: n=0")
    if len(cross_out_arr) > 0:
        print(f"  cross-genre : n={len(cross_out_arr)} frac_different={cross_out_arr.mean():.4f}")
    else:
        print("  cross-genre : n=0")


if __name__ == "__main__":
    main()
