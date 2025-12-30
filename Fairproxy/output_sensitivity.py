# output_sensitivity.py
"""
Output Sensitivity Test (OST) for LAFT

Goal (most direct test):
For each proxy x = (movie_title, genre), generate:

  y0 = f(x)
  yM = f(x, a=Male)
  yF = f(x, a=Female)

Then quantify how much outputs differ:
  - Jaccard distance on token sets
  - Edit distance (1 - SequenceMatcher ratio)
  - Embedding cosine distance (optional, if pass an embedder)

This file is standalone and can be plugged into your pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from difflib import SequenceMatcher


# -----------------------------
# Text distance helpers
# -----------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize_words(s: str) -> List[str]:
    if s is None:
        return []
    return _WORD_RE.findall(str(s).lower())


def jaccard_distance(a: str, b: str) -> float:
    """1 - Jaccard(token_set(a), token_set(b))"""
    A = set(_tokenize_words(a))
    B = set(_tokenize_words(b))
    denom = len(A | B)
    if denom == 0:
        return 0.0
    return float(1.0 - (len(A & B) / denom))


def edit_distance_norm(a: str, b: str) -> float:
    """
    Normalized edit-like distance using SequenceMatcher:
      dist = 1 - ratio
    (ratio in [0,1], higher means more similar)
    """
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    return float(1.0 - SequenceMatcher(None, a, b).ratio())


def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    """1 - cosine similarity"""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu = np.linalg.norm(u) + 1e-12
    nv = np.linalg.norm(v) + 1e-12
    return float(1.0 - float(np.dot(u, v) / (nu * nv)))


# -----------------------------
# Output Sensitivity Tester
# -----------------------------
@dataclass
class OSTConfig:
    max_new_tokens: int = 32
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 0.95
    # If True: tries to extract only the first line / first title-ish span
    one_line_only: bool = True


class OutputSensitivityTester:
    """
    Runs the Output Sensitivity Test (OST) for each proxy x.

    You can pass an optional embedder with .encode(list[str]) -> np.ndarray
    (e.g., sentence-transformers). If embedder is None, embedding distance is skipped.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        cfg: Optional[OSTConfig] = None,
        embedder: Optional[Any] = None,
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()

        self.tokenizer = tokenizer
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.cfg = cfg or OSTConfig()
        self.embedder = embedder

    # -----------------------------
    # Prompts (keep consistent with your vector_extractor)
    # -----------------------------
    @staticmethod
    def prompt_no_gender(movie_title: str, genre: str) -> str:
        return (
            f'Based on the movie "{movie_title}" in the {genre} genre, '
            "recommend 1 movie title only:"
        )

    @staticmethod
    def prompt_with_gender(movie_title: str, genre: str, gender: str) -> str:
        return (
            f'For a {gender} viewer who watched "{movie_title}" in the {genre} genre, '
            "recommend 1 movie title only:"
        )

    # -----------------------------
    # Generation
    # -----------------------------
    def generate_text(self, prompt: str) -> str:
        """
        Generates a short continuation from prompt and returns decoded text
        (only the generated part, not the prompt).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=int(self.cfg.max_new_tokens),
                do_sample=bool(self.cfg.do_sample),
                temperature=float(self.cfg.temperature),
                top_p=float(self.cfg.top_p),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_only = out_ids[:, prompt_len:]
        text = self.tokenizer.decode(gen_only[0], skip_special_tokens=True).strip()

        if self.cfg.one_line_only:
            # keep first line; also strip common list prefixes
            text = text.split("\n")[0].strip()
            text = re.sub(r"^[\-\*\d\.\)\s]+", "", text).strip()

        return text

    # -----------------------------
    # Embedding distance (optional)
    # -----------------------------
    def embedding_cosine_distance(self, a: str, b: str) -> Optional[float]:
        if self.embedder is None:
            return None
        try:
            vecs = self.embedder.encode([a, b])
            return cosine_distance(vecs[0], vecs[1])
        except Exception:
            return None

    # -----------------------------
    # Core per-proxy evaluation
    # -----------------------------
    def run_one(self, movie_title: str, genre: str) -> Dict[str, Any]:
        """
        Returns a dict with:
          y0, yM, yF and distances among them.
        """
        y0 = self.generate_text(self.prompt_no_gender(movie_title, genre))
        yM = self.generate_text(self.prompt_with_gender(movie_title, genre, "Male"))
        yF = self.generate_text(self.prompt_with_gender(movie_title, genre, "Female"))

        # Pairwise distances
        d_MF_j = jaccard_distance(yM, yF)
        d_MF_e = edit_distance_norm(yM, yF)
        d_M0_j = jaccard_distance(yM, y0)
        d_F0_j = jaccard_distance(yF, y0)
        d_M0_e = edit_distance_norm(yM, y0)
        d_F0_e = edit_distance_norm(yF, y0)

        d_MF_emb = self.embedding_cosine_distance(yM, yF)
        d_M0_emb = self.embedding_cosine_distance(yM, y0)
        d_F0_emb = self.embedding_cosine_distance(yF, y0)

        return {
            "movie_title": movie_title,
            "genre": genre,
            "y0": y0,
            "yM": yM,
            "yF": yF,
            # core “gender flip” sensitivity
            "delta_MF_jaccard": d_MF_j,
            "delta_MF_edit": d_MF_e,
            "delta_MF_emb": d_MF_emb,
            # how much adding male/female differs from neutral
            "delta_M0_jaccard": d_M0_j,
            "delta_F0_jaccard": d_F0_j,
            "delta_M0_edit": d_M0_e,
            "delta_F0_edit": d_F0_e,
            "delta_M0_emb": d_M0_emb,
            "delta_F0_emb": d_F0_emb,
        }

    def run_batch(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        rows: list of dicts, each must have movie_title and genre.
        Example input: predictions list from LLaMAGenderPredictor,
          where pred["movie_title"], pred["genre"] exist.
        """
        out = []
        for r in rows:
            movie = r["movie_title"]
            genre = r["genre"]
            res = self.run_one(movie, genre)

            # Keep ids/labels if present (helps joining with LAFT)
            for k in ("sample_id", "predicted_gender", "ground_truth", "confidence", "correct"):
                if k in r:
                    res[k] = r[k]

            out.append(res)
        return out


# -----------------------------
# Proxy-level summary helpers
# -----------------------------
def summarize_by_proxy(
    ost_rows: List[Dict[str, Any]],
    min_samples: int = 5,
    key_fields: Tuple[str, str] = ("movie_title", "genre"),
) -> List[Dict[str, Any]]:
    """
    Group OST results by proxy=(movie_title, genre) and summarize:

      mean/median delta_MF_* and also mean delta vs neutral.

    Returns list[dict] sorted by mean delta_MF_edit desc (you can change).
    """
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in ost_rows:
        key = (r.get(key_fields[0]), r.get(key_fields[1]))
        groups.setdefault(key, []).append(r)

    proxy_rows: List[Dict[str, Any]] = []
    for (movie, genre), lst in groups.items():
        if len(lst) < min_samples:
            continue

        def _vals(field: str) -> np.ndarray:
            xs = []
            for z in lst:
                v = z.get(field, None)
                if v is None:
                    continue
                xs.append(float(v))
            return np.asarray(xs, dtype=float)

        mf_edit = _vals("delta_MF_edit")
        mf_j = _vals("delta_MF_jaccard")
        m0_edit = _vals("delta_M0_edit")
        f0_edit = _vals("delta_F0_edit")

        row = {
            "movie_title": movie,
            "genre": genre,
            "n": int(len(lst)),
            "mean_delta_MF_edit": float(np.mean(mf_edit)) if mf_edit.size else None,
            "median_delta_MF_edit": float(np.median(mf_edit)) if mf_edit.size else None,
            "mean_delta_MF_jaccard": float(np.mean(mf_j)) if mf_j.size else None,
            "median_delta_MF_jaccard": float(np.median(mf_j)) if mf_j.size else None,
            "mean_delta_M0_edit": float(np.mean(m0_edit)) if m0_edit.size else None,
            "mean_delta_F0_edit": float(np.mean(f0_edit)) if f0_edit.size else None,
        }

        # If embedding distances exist, include them
        mf_emb = _vals("delta_MF_emb")
        if mf_emb.size:
            row["mean_delta_MF_emb"] = float(np.mean(mf_emb))
            row["median_delta_MF_emb"] = float(np.median(mf_emb))

        proxy_rows.append(row)

    # sort: largest MF edit difference first (more sensitive)
    proxy_rows.sort(
        key=lambda x: (x["mean_delta_MF_edit"] is None, -(x["mean_delta_MF_edit"] or 0.0))
    )
    return proxy_rows
