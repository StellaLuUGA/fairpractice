# vector_extractor.py
# Extract v0, v1, v2 from LLaMA (single GPU, matches your v0/v1/v2 definitions)

"""
Vector definitions (same study as LLaMAGenderPredictor):

Let:
- x = non-sensitive input (here: movie_title + genre prompt)
- a = ground-truth gender from dataset  (row["gender"] / pred["ground_truth"])
- â = predicted gender from LLaMAGenderPredictor (pred["predicted_gender"])
- y = downstream output (here: “recommend similar movies”)

We extract hidden vectors as a proxy for p'(y | ·):

v0 = p'(y | x)                 -> prompt WITHOUT any gender
v1 = p'(y | x, a = â)          -> prompt WITH the gender that makes a=â true
v2 = p'(y | x, a != â)         -> prompt WITH the gender that makes a!=â true

Implementation rule per sample:
- Compute v_true  = p'(y | x, a)         (prompt with ground-truth gender)
- Compute v_false = p'(y | x, not a)     (prompt with opposite of ground-truth)
Then:
- if â == a:  v1 = v_true   and v2 = v_false
- if â != a:  v1 = v_false  and v2 = v_true
"""

import torch
import numpy as np


class VectorExtractor:
    def __init__(self, model, tokenizer, layer_idx: int = -2, device: str = "cuda"):
        # single GPU/CPU safety
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()

        self.tokenizer = tokenizer
        self.layer_idx = layer_idx

        # avoid padding issues for LLaMA-family tokenizers
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # -----------------------------
    # Public API (v0/v1/v2)
    # -----------------------------
    def extract_all(self, movie_title: str, genre: str, predicted_gender: str, ground_truth_gender: str):
        """
        Returns:
            {
              'v0': p'(y | x),
              'v1': p'(y | x, a=â),
              'v2': p'(y | x, a!=â),
              'a_equals_ahat': bool,
              'a': ground_truth_gender,
              'ahat': predicted_gender,
              'gender_used_for_v1': 'Male'/'Female',
              'gender_used_for_v2': 'Male'/'Female'
            }
        """
        a = self._normalize_gender(ground_truth_gender)     # ground truth
        ahat = self._normalize_gender(predicted_gender)     # predicted
        if a not in ("Male", "Female") or ahat not in ("Male", "Female"):
            raise ValueError(f"Gender must be 'Male'/'Female'. Got a={a}, ahat={ahat}")

        opposite_of_a = "Female" if a == "Male" else "Male"

        # v0 = p'(y|x): no gender
        v0 = self._extract_hidden_state(self._prompt_no_gender(movie_title, genre))

        # these two are the only two gender-conditioned prompts we ever need
        v_true = self._extract_hidden_state(self._prompt_with_gender(movie_title, genre, a))
        v_false = self._extract_hidden_state(self._prompt_with_gender(movie_title, genre, opposite_of_a))

        a_equals_ahat = (a == ahat)

        # Assign to match your definitions:
        # - v1 corresponds to the "a=â" condition
        # - v2 corresponds to the "a!=â" condition
        if a_equals_ahat:
            v1, v2 = v_true, v_false
            gender_v1, gender_v2 = a, opposite_of_a
        else:
            v1, v2 = v_false, v_true
            gender_v1, gender_v2 = opposite_of_a, a

        return {
            "v0": v0,
            "v1": v1,
            "v2": v2,
            "a_equals_ahat": a_equals_ahat,
            "a": a,
            "ahat": ahat,
            "gender_used_for_v1": gender_v1,
            "gender_used_for_v2": gender_v2,
        }

    def extract_batch(self, predictions):
        """
        predictions: list of dicts from your LLaMAGenderPredictor.predict_batch output, e.g.:
          {
            'sample_id', 'movie_title', 'genre',
            'predicted_gender', 'ground_truth', ...
          }
        """
        results = []
        for pred in predictions:
            vecs = self.extract_all(
                movie_title=pred["movie_title"],
                genre=pred["genre"],
                predicted_gender=pred["predicted_gender"],
                ground_truth_gender=pred["ground_truth"],
            )

            results.append(
                {
                    "sample_id": pred["sample_id"],
                    "movie_title": pred["movie_title"],
                    "genre": pred["genre"],
                    "predicted_gender": pred["predicted_gender"],
                    "ground_truth": pred["ground_truth"],
                    "correct": pred.get("correct", pred["predicted_gender"] == pred["ground_truth"]),
                    **vecs,
                }
            )
        return results

    # -----------------------------
    # Prompt templates (x)
    # -----------------------------
    @staticmethod
    def _prompt_no_gender(movie_title: str, genre: str) -> str:
        # x only
        return f'Based on the movie "{movie_title}" in the {genre} genre, recommend movies:'

    @staticmethod
    def _prompt_with_gender(movie_title: str, genre: str, gender: str) -> str:
        # x plus gender
        return f'For a {gender} viewer who watched "{movie_title}" in the {genre} genre, recommend movies:'

    @staticmethod
    def _normalize_gender(g: str) -> str:
        # Accept common variants; normalize to 'Male'/'Female'
        if g is None:
            return g
        g2 = str(g).strip().lower()
        if g2 in ("m", "male"):
            return "Male"
        if g2 in ("f", "female"):
            return "Female"
        return str(g).strip()

    # -----------------------------
    # Layer access + hidden vector
    # -----------------------------
    def _get_layers_container(self):
        # Common HF layouts for LLaMA-family models
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "layers"):
            return self.model.layers
        if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model") and hasattr(
            self.model.base_model.model, "layers"
        ):
            return self.model.base_model.model.layers
        raise AttributeError(
            "Cannot find transformer layers. Tried: model.model.layers, model.layers, model.base_model.model.layers"
        )

    def _extract_hidden_state(self, prompt: str) -> np.ndarray:
        captured = []

        def hook_fn(module, inputs, output):
            hs = output[0] if isinstance(output, (tuple, list)) else output
            captured.append(hs.detach().to(torch.float32).cpu())  # <-- FIX

        layers = self._get_layers_container()
        target_layer = layers[self.layer_idx]
        hook = target_layer.register_forward_hook(hook_fn)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                _ = self.model(**inputs)
        finally:
            hook.remove()

        if not captured:
            raise RuntimeError("No hidden states captured. Layer path or hook output may not match this model.")

        return captured[0][0, -1, :].numpy()

