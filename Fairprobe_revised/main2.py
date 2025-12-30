# main.py
# Extract "decision-time" internal vectors for the GENDER PREDICTION task.
#
# Key idea:
#   We use the same prompt as LLaMAGenderPredictor._build_prompt(movie, genre).
#   Then we run ONE forward pass on the prompt only.
#   We hook a chosen module (whole layer / self_attention / MLP) and capture its output hidden states [T,H].
#   We take the vector at position (prompt_len - 1), i.e., the state right before the model outputs the answer token.
#
# This aligns the internal vector with the actual gender decision, instead of a recommendation prompt.

import os
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


def _pick_tensor_from_output(output) -> Optional[torch.Tensor]:
    """
    Hook outputs vary:
      - tensor
      - tuple(tensor, ...)
      - list(tensor, ...)
    We pick the first tensor, which is almost always the main hidden-state-like output.
    """
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for x in output:
            if torch.is_tensor(x):
                return x
    return None

#Helper: find transformer layers in different model structures
def _get_layers(model):
    # Most HF LLaMA: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Some variants: model.layers
    if hasattr(model, "layers"):
        return model.layers
    # Some wrappers: model.base_model.model.layers
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "layers"):
        return model.base_model.model.layers
    raise AttributeError("Cannot find transformer layers container on this model.")

#Helper: choose which part of the layer to hook
#self-attention (it decides what words/tokens to pay attention to)
#MLP (it does math mixing / transformation on what attention produced)
def _get_hook_module(model, layer_idx: int, layer_select: str):
    layers = _get_layers(model)
    layer = layers[layer_idx]

    if layer_select == "hidden_state":
        return layer
    if layer_select == "self_attention":
        if not hasattr(layer, "self_attn"):
            raise AttributeError("Selected layer has no .self_attn")
        return layer.self_attn
    if layer_select == "MLP":
        if not hasattr(layer, "mlp"):
            raise AttributeError("Selected layer has no .mlp")
        return layer.mlp

    raise ValueError("layer_select must be one of: hidden_state, self_attention, MLP")


def safe_build_gender_prompt(predictor: LLaMAGenderPredictor, movie_title: str, genre: str) -> str:
    """
    Uses predictor._build_prompt if possible.
    If tokenizer has chat_template but jinja2 is too old, apply_chat_template will throw ImportError.
    We catch and fall back to a plain prompt that matches predictor's user_msg + 'Answer:' style.
    """
    try:
        return predictor._build_prompt(movie_title, genre)
    except ImportError:
        # fallback that mirrors the content in predictor._build_prompt
        user_msg = (
            f'A person watched the movie "{movie_title}" which is in the {genre} genre.\n'
            "Based on typical viewing patterns, is this person more likely Male or Female?\n"
            "Answer with exactly one word: Male or Female."
        )
        return user_msg + "\nAnswer:"


@torch.inference_mode()
def extract_decision_vector(
    model,
    tokenizer,
    device: torch.device,
    prompt_text: str,
    layer_idx: int,
    layer_select: str,
    debug_hook: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Returns (vec, meta)

    vec:
      - np.ndarray [hidden_dim]
      - taken at the position right before the model outputs the first answer token
        i.e. position = prompt_len - 1

    meta:
      - prompt_len_ids, hook_shape, used_pos, etc.
    """
    captured: List[torch.Tensor] = []
    printed = False

    def hook_fn(module, inputs, output):
        nonlocal printed
        hs = _pick_tensor_from_output(output)
        if hs is None:
            return
        if debug_hook and not printed:
            print(f"[hook] layer_select={layer_select} shape={tuple(hs.shape)} dtype={hs.dtype}")
            printed = True
        captured.append(hs.detach().to(torch.float32).cpu())

    # IMPORTANT: match predictor scoring style: encode(prompt, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if len(prompt_ids) < 1:
        raise RuntimeError("Prompt tokenized to empty sequence (unexpected).")

    input_ids = torch.tensor([prompt_ids], device=device)
    # input_ids looks like a tensor with shape [1, T]:
    # 1 = batch size (one prompt)
    # T = number of tokens in the prompt
    attn_mask = torch.ones_like(input_ids, device=device)

    hook_module = _get_hook_module(model, layer_idx=layer_idx, layer_select=layer_select)
    h = hook_module.register_forward_hook(hook_fn)
    try:
        _ = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    finally:
        h.remove()

    if not captured:
        raise RuntimeError("No hidden states captured. Hook path/output did not match expectations.")

    hs = captured[0]  # first forward call tensor
    hook_shape = tuple(hs.shape)

    # normalize to [T, H]
    if hs.dim() == 3:
        hs_all = hs[0]          # [T, H]
    elif hs.dim() == 2:
        hs_all = hs             # [T, H]
    else:
        # fallback: flatten
        vec = hs.reshape(-1).numpy()
        return vec, {
            "prompt_len_ids": int(len(prompt_ids)),
            "hook_shape": hook_shape,
            "used_pos": None,
            "hidden_dim": int(vec.shape[0]),
            "note": "unexpected_hook_dim_flattened",
        }

    T = int(hs_all.shape[0])
    H = int(hs_all.shape[1])
    prompt_len = int(len(prompt_ids))
    # T = number of tokens (time steps)
    # H = hidden dimension size
    # decision position = last prompt token (predicts next token)
    desired_pos = prompt_len - 1
    used_pos = min(max(desired_pos, 0), T - 1)
    # last token in the prompt is at index prompt_len - 1
    vec = hs_all[used_pos, :].numpy()
    # vec: the decision-time internal vector

    meta = {
        "prompt_len_ids": prompt_len,
        "hook_shape": hook_shape,
        "T_hook": T,
        "H": H,
        "desired_pos": desired_pos,
        "used_pos": used_pos,
        "layer_idx": layer_idx,
        "layer_select": layer_select,
    }
    return vec, meta


def normalize_gt_gender(raw: str) -> str:
    g = str(raw).strip().lower()
    if g in ("m", "male"):
        return "Male"
    if g in ("f", "female"):
        return "Female"
    return str(raw).strip()


def run_linear_probe(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Dict[str, float]:
    """
    Tiny torch logistic regression probe to test separability.
    No sklearn required.

    X: [N, H] float32
    y: [N] 0/1
    80% train / 20% test split
    """
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)

    n = len(y)
    n_train = max(1, int(0.8 * n))
    tr = idx[:n_train]
    te = idx[n_train:] if n_train < n else idx[:1]  # avoid empty test

    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    # standardize by train stats
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.float32).view(-1, 1)

    model = torch.nn.Linear(Xtr.shape[1], 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(200):
        opt.zero_grad()
        logits = model(Xtr_t)
        loss = loss_fn(logits, ytr_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        te_logits = model(Xte_t)
        te_prob = torch.sigmoid(te_logits)
        te_pred = (te_prob >= 0.5).float()
        acc = float((te_pred.eq(yte_t)).float().mean().item())

    return {"probe_acc": acc, "n_train": float(len(tr)), "n_test": float(len(te))}


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default=None, help="float16|bfloat16|float32 or None")
    ap.add_argument("--seed", type=int, default=42)

    # prediction selection
    ap.add_argument("--confidence_threshold", type=float, default=0.80)
    ap.add_argument("--max_pred_rows", type=int, default=800, help="max user-movie rows to run gender predictor on")
    ap.add_argument("--max_vec_samples", type=int, default=200, help="max high-conf samples to extract vectors for")

    # hook config
    ap.add_argument("--layer_idx", type=int, default=-2)
    ap.add_argument("--layer_select", type=str, default="hidden_state", choices=["hidden_state", "self_attention", "MLP"])
    ap.add_argument("--debug_hook", action="store_true")

    # output
    ap.add_argument("--out_dir", type=str, default="outputs_gender_probe")
    ap.add_argument("--no_probe", action="store_true", help="skip training the linear probe")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    # 1) Load data
    movies, ratings, users, target_set = load_data()
    df_um = build_user_movie_df(movies, ratings, users, target_set)

    # cost control
    if len(df_um) > args.max_pred_rows:
        df_pred = df_um.sample(n=args.max_pred_rows, random_state=args.seed).reset_index(drop=True)
    else:
        df_pred = df_um.reset_index(drop=True)

    # 2) Load predictor (model+tokenizer)
    predictor = LLaMAGenderPredictor(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
    )
    device = predictor.device

    # 3) Run gender prediction and keep high confidence
    preds = predictor.predict_batch(df_pred, confidence_threshold=float(args.confidence_threshold))
    if len(preds) == 0:
        print("No high-confidence predictions. Try lowering --confidence_threshold or increasing --max_pred_rows.")
        return
    if len(preds) > args.max_vec_samples:
        preds = preds[: args.max_vec_samples]

    # 4) Extract decision vectors (gender task)
    rows: List[Dict[str, Any]] = []
    vecs: List[np.ndarray] = []
    labels: List[int] = []

    for p in preds:
        title = str(p["movie_title"])
        genre = str(p["genre"])
        gt = normalize_gt_gender(p["ground_truth"])

        prompt_text = safe_build_gender_prompt(predictor, title, genre)

        vec, meta = extract_decision_vector(
            model=predictor.model,
            tokenizer=predictor.tokenizer,
            device=device,
            prompt_text=prompt_text,
            layer_idx=int(args.layer_idx),
            layer_select=str(args.layer_select),
            debug_hook=bool(args.debug_hook),
        )

        y = 1 if gt == "Male" else 0  # Male=1, Female=0 (consistent label)
        vecs.append(vec.astype(np.float32))
        labels.append(y)

        rows.append(
            {
                "sample_id": p.get("sample_id"),
                "movie_title": title,
                "genre": genre,
                "ground_truth": gt,
                "predicted_gender": p.get("predicted_gender"),
                "confidence": float(p.get("confidence", np.nan)),
                "correct": bool(p.get("correct", False)),
                "male_prob": float(p.get("male_prob", np.nan)),
                "female_prob": float(p.get("female_prob", np.nan)),
                "male_logp": float(p.get("male_logp", np.nan)),
                "female_logp": float(p.get("female_logp", np.nan)),
                "layer_idx": int(args.layer_idx),
                "layer_select": str(args.layer_select),
                "prompt_len_ids": meta.get("prompt_len_ids"),
                "used_pos": meta.get("used_pos"),
                "hook_shape": str(meta.get("hook_shape")),
            }
        )

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(args.out_dir, "gender_task_vectors_meta.csv"), index=False)

    X = np.stack(vecs, axis=0)  # [N, H]
    y = np.array(labels, dtype=np.int64)

    np.save(os.path.join(args.out_dir, "X_vectors.npy"), X)
    np.save(os.path.join(args.out_dir, "y_labels.npy"), y)

    print(f"Saved vectors/meta to {args.out_dir}")
    print(f"X shape: {X.shape} (N samples, H hidden_dim)")
    print(f"Label counts: Male={int((y==1).sum())}, Female={int((y==0).sum())}")

    # 5) Optional: probe separability
    if not args.no_probe and len(y) >= 10 and len(np.unique(y)) == 2:
        stats = run_linear_probe(X, y, seed=args.seed)
        with open(os.path.join(args.out_dir, "probe_result.txt"), "w") as f:
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
        print(f"Linear probe accuracy (quick sanity check): {stats['probe_acc']:.3f}")
    else:
        print("Skipped probe (use --no_probe to silence, or need >=10 samples and both classes).")


if __name__ == "__main__":
    main()
