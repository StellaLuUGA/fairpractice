"""
A corrected VectorExtractor that cleanly supports three extraction targets:
"self_attention" (attention block output)
"MLP" (FFN output)
"hidden_state" (whole decoder layer output)
"""


# vector_extractor.py
# Extract internal vectors (v0/v1/v2) from a HF LLaMA-family CausalLM.
#
# What you get:
#   - v0 = vector from prompt without gender
#   - v1 = vector from prompt with gender that makes a==ahat true
#   - v2 = vector from prompt with opposite gender
#
# Also returns dimensions + helpful metadata:
#   - hidden_dim
#   - prompt_len / seq_len / gen_len
#
# Supports 3 extraction targets:
#   layer_select="hidden_state"  -> whole decoder layer output
#   layer_select="self_attention"-> attention submodule output
#   layer_select="MLP"           -> MLP/FFN submodule output
#
# Supports token selection over generated continuation only:
#   extract_mode="last_gen" | "first_gen" | "mean_gen"

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class ExtractConfig:
    layer_idx: int = -2
    layer_select: str = "hidden_state"  # "hidden_state" | "self_attention" | "MLP"
    extract_mode: str = "last_gen"      # "last_gen" 
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 0.95
    debug_hook: bool = False


class VectorExtractor:
    def __init__(self, model, tokenizer, device: str = "cuda", cfg: Optional[ExtractConfig] = None):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()

        self.tokenizer = tokenizer
        # avoid padding errors (common for LLaMA tokenizers)
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.cfg = cfg or ExtractConfig()

        if self.cfg.layer_select not in ("hidden_state", "self_attention", "MLP"):
            raise ValueError("cfg.layer_select must be one of: hidden_state, self_attention, MLP")
    # -----------------------------
    # Public API
    # -----------------------------
    def extract_all(
        self,
        movie_title: str,
        genre: str,
        predicted_gender: str,
        ground_truth_gender: str,
    ) -> Dict[str, Any]:
        """
        Returns dict containing:
          v0, v1, v2 (np.ndarray shape [hidden_dim])
          hidden_dim (int)
          plus metadata about which gender was used.
        """
        a = self._normalize_gender(ground_truth_gender)
        ahat = self._normalize_gender(predicted_gender)
        if a not in ("Male", "Female") or ahat not in ("Male", "Female"):
            raise ValueError(f"Gender must normalize to 'Male'/'Female'. Got a={a}, ahat={ahat}")

        opposite_of_a = "Female" if a == "Male" else "Male"

        # v0: no gender
        v0, meta0 = self._extract_hidden_vector(self._prompt_no_gender(movie_title, genre))
        # v_true: with ground-truth gender
        v_true, meta_true = self._extract_hidden_vector(self._prompt_with_gender(movie_title, genre, a))
        # v_false: with opposite gender
        v_false, meta_false = self._extract_hidden_vector(self._prompt_with_gender(movie_title, genre, opposite_of_a))

        a_equals_ahat = (a == ahat)
        if a_equals_ahat:
            v1, v2 = v_true, v_false
            gender_v1, gender_v2 = a, opposite_of_a
            meta1, meta2 = meta_true, meta_false
        else:
            v1, v2 = v_false, v_true
            gender_v1, gender_v2 = opposite_of_a, a
            meta1, meta2 = meta_false, meta_true

        # hidden_dim should be same for all, but we take from v0
        hidden_dim = int(v0.shape[0])

        return {
            "v0": v0,
            "v1": v1,
            "v2": v2,
            "hidden_dim": hidden_dim,
            "a_equals_ahat": a_equals_ahat,
            "a": a,
            "ahat": ahat,
            "gender_used_for_v1": gender_v1,
            "gender_used_for_v2": gender_v2,
        }

    def extract_batch(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        predictions: list of dicts like your gender predictor output:
          {
            'sample_id', 'movie_title', 'genre',
            'predicted_gender', 'ground_truth', ...
          }
        """
        out = []
        for pred in predictions:
            vecs = self.extract_all(
                movie_title=str(pred["movie_title"]),
                genre=str(pred["genre"]),
                predicted_gender=str(pred["predicted_gender"]),
                ground_truth_gender=str(pred["ground_truth"]),
            )
            out.append(
                {
                    "sample_id": pred.get("sample_id"),
                    "movie_title": pred["movie_title"],
                    "genre": pred["genre"],
                    "predicted_gender": pred["predicted_gender"],
                    "ground_truth": pred["ground_truth"],
                    "correct": pred.get("correct", pred["predicted_gender"] == pred["ground_truth"]),
                    **vecs,
                }
            )
        return out

    # -----------------------------
    # Prompt templates
    # -----------------------------
    @staticmethod
    def _prompt_no_gender(movie_title: str, genre: str) -> str:
        return (
            f'For a viewer with gender={{neutral}} who watched "{movie_title}" in the {genre} genre, '
            "recommend 1 movie title only:"
        )

    @staticmethod
    def _prompt_with_gender(movie_title: str, genre: str, gender: str) -> str:
        return (
            f'For a {gender} viewer who watched "{movie_title}" in the {genre} genre, '
            "recommend 1 movie title only:"
        )

    @staticmethod
    def _normalize_gender(g: str) -> str:
        if g is None:
            return g
        g2 = str(g).strip().lower()
        if g2 in ("m", "male"):
            return "Male"
        if g2 in ("f", "female"):
            return "Female"
        return str(g).strip()

    # -----------------------------
    # Model layer / module selection
    # -----------------------------
    def _get_layers(self):
        # Most HF LLaMA: model.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        # Some variants: model.layers
        if hasattr(self.model, "layers"):
            return self.model.layers
        # Some wrappers: model.base_model.model.layers
        if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model") and hasattr(
            self.model.base_model.model, "layers"
        ):
            return self.model.base_model.model.layers
        raise AttributeError("Cannot find transformer layers container on this model.")

    def _get_hook_module(self):
        layers = self._get_layers()
        layer = layers[self.cfg.layer_idx]

        # layers is the list of transformer blocks.
        # self.cfg.layer_idx picks which one (e.g. -2 second last).
        # layer is now a single transformer block object (e.g., the 30th block in a 32-layer model).

        # A decoder block (simplified) is roughly:
        # input hidden states x
        # self-attention transforms x → produces an attention output
        # MLP/FFN transforms x → produces an MLP output
        # residual connections + layer norms combine these


        if self.cfg.layer_select == "hidden_state":
            return layer  # whole decoder layer
        if self.cfg.layer_select == "self_attention":
            if not hasattr(layer, "self_attn"):
                raise AttributeError("Selected layer has no .self_attn")
            return layer.self_attn
        #layer.self_attn is the attention module inside the block.
        # Hooking this module captures the output right after attention, before it gets mixed into the residual stream (exact point depends on architecture).
        if self.cfg.layer_select == "MLP":
            if not hasattr(layer, "mlp"):
                raise AttributeError("Selected layer has no .mlp")
            return layer.mlp
        # Many “concept features” or “attribute steering” effects show up strongly in MLP layers.
        # If gender causes changes mostly here, that suggests the effect is more about internal feature composition rather than token-to-token attention.

        raise RuntimeError("Unreachable")

    # -----------------------------
    # Hook output normalization
    # -----------------------------
    # If output is already a tensor, return it immediately.
    # Example: output is hidden states tensor [B, T, H].
   
    # EAttention might return (attn_output, attn_weights)
    # attn_output is tensor [B, T, H] → gets returned
    # attn_weights is tensor [B, heads, T, T] → ignored because it’s second
    # Some models return (hidden_states, past_key_values)
    # hidden_states tensor returned, past_key_values ignored.
    # the first tensor is usually the main “hidden state” result you care about
    # and the later items are often extras (like attention weights, caches, etc.).

    @staticmethod
    def _pick_tensor_from_output(output) -> Optional[torch.Tensor]:
        """
        Hooks can return:
          - tensor
          - tuple(tensor, ...)
          - list(tensor, ...)
        We pick the first tensor-like element.
        """
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)):
            for x in output:
                if torch.is_tensor(x):
                    return x
        return None

    # -----------------------------
    # Core extraction: vector + metadata
    # -----------------------------
    @torch.inference_mode()
    def _extract_hidden_vector(self, prompt: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Returns (vec, meta)
          vec: np.ndarray [hidden_dim]
          meta: dict with prompt_len/seq_len/gen_len/hidden_dim/hook_shape
        """
        captured: List[torch.Tensor] = []
        printed = False

        def hook_fn(module, inputs, output):
            nonlocal printed
            hs = self._pick_tensor_from_output(output)
            if hs is None:
                return

            if self.cfg.debug_hook and not printed:
                print(f"[hook] layer_select={self.cfg.layer_select} got shape={tuple(hs.shape)} dtype={hs.dtype}")
                printed = True

            captured.append(hs.detach().to(torch.float32).cpu())

        # 1) tokenize prompt
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
        prompt_len = int(prompt_inputs["input_ids"].shape[1])

        # 2) generate continuation tokens
        gen_ids = self.model.generate(
            **prompt_inputs,
            max_new_tokens=int(self.cfg.max_new_tokens),
            do_sample=bool(self.cfg.do_sample),
            temperature=float(self.cfg.temperature),
            top_p=float(self.cfg.top_p),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        seq_len = int(gen_ids.shape[1]) #2nd dim is sequence length
        gen_len = seq_len - prompt_len

        # 3) run forward on full sequence with a hook
        full_inputs = {
            "input_ids": gen_ids,
            "attention_mask": torch.ones_like(gen_ids, device=self.device),
        }

        hook_module = self._get_hook_module()
        h = hook_module.register_forward_hook(hook_fn)
        try:
            _ = self.model(**full_inputs)
        finally:
            h.remove()

        if not captured:
            raise RuntimeError("No hidden states captured. Hook path/output did not match expectations.")

        hs = captured[0]
        hook_shape = tuple(hs.shape)

        # Normalize hs into [T, H]
        if hs.dim() == 3:          # [B, T, H]
            hs_all = hs[0]
        elif hs.dim() == 2:        # [T, H]
            hs_all = hs
        else:
            # unexpected; flatten to vector
            vec = hs.reshape(-1)
            return vec.numpy(), {
                "prompt_len": prompt_len,
                "seq_len": seq_len,
                "gen_len": gen_len,
                "hook_shape": hook_shape,
                "hidden_dim": int(vec.numel()),
                "layer_idx": self.cfg.layer_idx,
                "layer_select": self.cfg.layer_select,
                "extract_mode": self.cfg.extract_mode,
            }

        # If model generated nothing new, fall back to last prompt token
        if gen_len <= 0:
            vec_t = hs_all[-1, :]
        else:
            gen_start = prompt_len
            gen_end = seq_len

            if self.cfg.extract_mode == "last_gen":
                vec_t = hs_all[seq_len - 1, :]
            elif self.cfg.extract_mode == "first_gen":
                vec_t = hs_all[gen_start, :]
            else:  # mean_gen
                vec_t = hs_all[gen_start:gen_end, :].mean(dim=0)

        vec = vec_t.numpy()
        hidden_dim = int(vec.shape[0])

        meta = {
            "prompt_len": prompt_len,
            "seq_len": seq_len,
            "gen_len": gen_len,
            "hook_shape": hook_shape,
            "hidden_dim": hidden_dim,
            "layer_idx": self.cfg.layer_idx,
            "layer_select": self.cfg.layer_select,
            "extract_mode": self.cfg.extract_mode,
        }
        return vec, meta


