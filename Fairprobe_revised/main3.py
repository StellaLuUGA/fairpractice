# main.py
# Goal:
#   For each (movie_title, genre) sample, run the LLM to recommend ONE movie title under:
#     v0 prompt: gender={neutral}
#     v1 prompt: gender chosen the same way as VectorExtractor.extract_all() (depends on GT vs predicted gender)
#     v2 prompt: the other gender
#   Then output:
#     - v0/v1/v2 recommended movie titles
#     - embeddings of those titles (mean of input-token embeddings)
#
# Outputs saved to --out_dir:
#   - reco_v0v1v2.csv
#   - reco_titles_v0.txt / v1.txt / v2.txt
#   - reco_emb_v0.npy / v1.npy / v2.npy   (each [N, H])

import os
import re
import argparse
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from data import load_data, build_user_movie_df
from llm_predictor import LLaMAGenderPredictor


# -----------------------------
# Repro / utils
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_gender(g: str) -> str:
    if g is None:
        return g
    g2 = str(g).strip().lower()
    if g2 in ("m", "male"):
        return "Male"
    if g2 in ("f", "female"):
        return "Female"
    return str(g).strip()


# -----------------------------
# v0/v1/v2 prompts (match your vector_extractor.py)
# -----------------------------
def prompt_v0(movie_title: str, genre: str) -> str:
    return (
        f'For a viewer with gender={{neutral}} who watched "{movie_title}" in the {genre} genre, '
        "recommend 1 movie title only:"
    )


def prompt_with_gender(movie_title: str, genre: str, gender: str) -> str:
    return (
        f'For a {gender} viewer who watched "{movie_title}" in the {genre} genre, '
        "recommend 1 movie title only:"
    )


def choose_v1_v2_genders(predicted_gender: str, ground_truth_gender: str) -> Tuple[str, str]:
    """
    Exactly mirrors the VectorExtractor.extract_all() logic for v1/v2:
      a = ground truth (Male/Female)
      ahat = predicted (Male/Female)
      opposite_of_a = opposite(a)
      if a==ahat:
         v1 uses a, v2 uses opposite(a)
      else:
         v1 uses opposite(a), v2 uses a
    Returns: (gender_for_v1, gender_for_v2)
    """
    a = normalize_gender(ground_truth_gender)
    ahat = normalize_gender(predicted_gender)
    if a not in ("Male", "Female") or ahat not in ("Male", "Female"):
        raise ValueError(f"Gender must normalize to Male/Female. Got a={a}, ahat={ahat}")
    opposite = "Female" if a == "Male" else "Male"
    if a == ahat:
        return a, opposite
    else:
        return opposite, a


# -----------------------------
# Generation + parsing
# -----------------------------
def parse_one_title(gen_text: str) -> str:
    """
    Parse LLM output into ONE movie title:
      - first non-empty line
      - remove bullets/numbering
      - strip quotes
    """
    text = str(gen_text).strip()
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    s = lines[0]

    # remove bullets/numbering like "- ", "* ", "1.", "1)", "1 -", "1:"
    s = re.sub(r"^\s*[\-\*\u2022]\s*", "", s)
    s = re.sub(r"^\s*\d+\s*[\.\)\-:]\s*", "", s)

    # strip wrapping quotes
    s = s.strip().strip('"').strip("'").strip()

    # remove trailing punctuation
    s = s.rstrip(" .,:;!-")
    return s


@torch.inference_mode()
def generate_one_title(
    model,
    tokenizer,
    device: torch.device,
    prompt_text: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    """
    Generate ONE title from the recommendation prompt and return parsed title.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).to(device)

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)

    gen_ids = model.generate(**inputs, **gen_kwargs)

    # decode only new tokens
    prompt_len = int(inputs["input_ids"].shape[1])
    new_tokens = gen_ids[0, prompt_len:]
    gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return parse_one_title(gen_text)


@torch.inference_mode()
def title_embedding_mean_input_embeds(
    model,
    tokenizer,
    device: torch.device,
    title: str,
) -> np.ndarray:
    """
    "Embedding of a movie title" = mean of model.get_input_embeddings() over its tokens.
    Output shape: [H]
    """
    title = str(title)
    token_ids = tokenizer.encode(" " + title, add_special_tokens=False)
    if len(token_ids) == 0:
        raise RuntimeError(f"Title tokenized to empty: {title!r}")

    emb_layer = model.get_input_embeddings()  # [vocab, H]
    ids_t = torch.tensor(token_ids, dtype=torch.long, device=device)  # [L]
    embs = emb_layer(ids_t)  # [L, H]
    return embs.mean(dim=0).detach().to(torch.float32).cpu().numpy()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # model
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default=None, help="float16|bfloat16|float32 or None")
    ap.add_argument("--seed", type=int, default=42)

    # which samples to run on
    ap.add_argument("--confidence_threshold", type=float, default=0.80)
    ap.add_argument("--max_pred_rows", type=int, default=800)
    ap.add_argument("--max_samples", type=int, default=200)

    # generation
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)

    # output
    ap.add_argument("--out_dir", type=str, default="outputs_reco_v0v1v2")

    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    # 1) Load data
    movies, ratings, users, target_set = load_data()
    df_um = build_user_movie_df(movies, ratings, users, target_set)

    # subsample rows to control cost
    if len(df_um) > args.max_pred_rows:
        df_pred = df_um.sample(n=args.max_pred_rows, random_state=args.seed).reset_index(drop=True)
    else:
        df_pred = df_um.reset_index(drop=True)

    # 2) Load predictor (model + tokenizer)
    predictor = LLaMAGenderPredictor(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
    )
    model = predictor.model
    tokenizer = predictor.tokenizer
    device = predictor.device

    # ensure pad token
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # hidden size for empty-title fallback
    H = int(model.get_input_embeddings().weight.shape[1])

    # 3) Get predicted gender labels first (so v1/v2 follow your extractor logic)
    preds = predictor.predict_batch(df_pred, confidence_threshold=float(args.confidence_threshold))
    if len(preds) == 0:
        print("No high-confidence predictions. Lower --confidence_threshold or increase --max_pred_rows.")
        return
    preds = preds[: int(args.max_samples)]

    #--max_pred_rows default = 800 (so it can start from up to 800 user-movie rows)
    #--max_samples default = 200 (so you finally keep up to 200 predictions)

    # 4) For each sample, generate v0/v1/v2 recommended titles + embeddings
    rows: List[Dict[str, Any]] = []
    emb_v0: List[np.ndarray] = []
    emb_v1: List[np.ndarray] = []
    emb_v2: List[np.ndarray] = []

    titles_v0: List[str] = []
    titles_v1: List[str] = []
    titles_v2: List[str] = []

    for i, p in enumerate(preds):
        movie_title = str(p["movie_title"])
        genre = str(p["genre"])
        gt = normalize_gender(p["ground_truth"])
        ahat = normalize_gender(p["predicted_gender"])

        gender_v1, gender_v2 = choose_v1_v2_genders(ahat, gt)

        p0 = prompt_v0(movie_title, genre)
        p1 = prompt_with_gender(movie_title, genre, gender_v1)
        p2 = prompt_with_gender(movie_title, genre, gender_v2)

        t0 = generate_one_title(model, tokenizer, device, p0, args.max_new_tokens, args.do_sample, args.temperature, args.top_p)
        t1 = generate_one_title(model, tokenizer, device, p1, args.max_new_tokens, args.do_sample, args.temperature, args.top_p)
        t2 = generate_one_title(model, tokenizer, device, p2, args.max_new_tokens, args.do_sample, args.temperature, args.top_p)

        # print all predicted movie names
        print(f"[{i:03d}] INPUT watched='{movie_title}' | genre={genre}")
        print(f"      v0 (neutral) -> {t0}")
        print(f"      v1 ({gender_v1}) -> {t1}")
        print(f"      v2 ({gender_v2}) -> {t2}")

        # embeddings (mean input embeddings)
        def emb_or_nan(title: str) -> np.ndarray:
            if str(title).strip() == "":
                return (np.zeros((H,), dtype=np.float32) * np.nan)
            return title_embedding_mean_input_embeds(model, tokenizer, device, title).astype(np.float32)

        e0 = emb_or_nan(t0)
        e1 = emb_or_nan(t1)
        e2 = emb_or_nan(t2)

        titles_v0.append(t0); titles_v1.append(t1); titles_v2.append(t2)
        emb_v0.append(e0); emb_v1.append(e1); emb_v2.append(e2)

        rows.append(
            {
                "sample_id": p.get("sample_id"),
                "movie_title": movie_title,
                "genre": genre,

                "ground_truth_gender": gt,
                "predicted_gender": ahat,
                "confidence": float(p.get("confidence", np.nan)),
                "correct": bool(p.get("correct", ahat == gt)),

                "gender_used_for_v1": gender_v1,
                "gender_used_for_v2": gender_v2,

                "v0_prompt": p0,
                "v1_prompt": p1,
                "v2_prompt": p2,

                "v0_reco_title": t0,
                "v1_reco_title": t1,
                "v2_reco_title": t2,
            }
        )

    # 5) Save
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(args.out_dir, "reco_v0v1v2.csv"), index=False)

    E0 = np.stack(emb_v0, axis=0)  # [N,H]
    E1 = np.stack(emb_v1, axis=0)
    E2 = np.stack(emb_v2, axis=0)

    np.save(os.path.join(args.out_dir, "reco_emb_v0.npy"), E0)
    np.save(os.path.join(args.out_dir, "reco_emb_v1.npy"), E1)
    np.save(os.path.join(args.out_dir, "reco_emb_v2.npy"), E2)

    with open(os.path.join(args.out_dir, "reco_titles_v0.txt"), "w", encoding="utf-8") as f:
        for t in titles_v0:
            f.write(t + "\n")
    with open(os.path.join(args.out_dir, "reco_titles_v1.txt"), "w", encoding="utf-8") as f:
        for t in titles_v1:
            f.write(t + "\n")
    with open(os.path.join(args.out_dir, "reco_titles_v2.txt"), "w", encoding="utf-8") as f:
        for t in titles_v2:
            f.write(t + "\n")

    print("\nDONE.")
    print(f"Saved to: {args.out_dir}")
    print(f"Embeddings shapes: v0={E0.shape}, v1={E1.shape}, v2={E2.shape}")


if __name__ == "__main__":
    main()
