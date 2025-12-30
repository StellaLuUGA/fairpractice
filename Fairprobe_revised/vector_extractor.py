"""
A corrected VectorExtractor that cleanly supports three extraction targets:
"self_attention" (attention block output)
"MLP" (FFN output)
"hidden_state" (whole decoder layer output)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class ExtractConfig:
    layer_idx: int = -2
    layer_select: str = "hidden_state"  # "hidden_state" | "self_attention" | "MLP"

    # generation length cap (needed so extractor doesn't explode cost/time)
    max_new_tokens: int = 32

    # token selection over generated continuation only
    extract_mode: str = "last_gen"      # "last_gen" | "first_gen" | "mean_gen"

    # generation mode
    do_sample: bool = False
    temperature: float = 0.7            # used only if do_sample=True
    top_p: float = 0.95                 # used only if do_sample=True

    debug_hook: bool = False


class VectorExtractor:
    def __init__(self, model, tokenizer, device: str = "cuda", cfg: Optional[ExtractConfig] = None):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()

        self.tokenizer = tokenizer
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.cfg = cfg or ExtractConfig()

        if self.cfg.layer_select not in ("hidden_state", "self_attention", "MLP"):
            raise ValueError("cfg.layer_select must be one of: hidden_state, self_attention, MLP")
        if self.cfg.extract_mode not in ("last_gen", "first_gen", "mean_gen"):
            raise ValueError("cfg.extract_mode must be one of: last_gen, first_gen, mean_gen")

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

        # v0: neutral gender in prompt
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
            "meta_v0": meta0,
            "meta_v1": meta1,
            "meta_v2": meta2,
        }

    def extract_batch(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        # NOTE: v0 uses explicit neutral attribute in the prompt
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
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "layers"):
            return self.model.layers
        if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model") and hasattr(
            self.model.base_model.model, "layers"
        ):
            return self.model.base_model.model.layers
        raise AttributeError("Cannot find transformer layers container on this model.")

    def _get_hook_module(self):
        layers = self._get_layers()
        layer = layers[self.cfg.layer_idx]

        if self.cfg.layer_select == "hidden_state":
            return layer
        if self.cfg.layer_select == "self_attention":
            if not hasattr(layer, "self_attn"):
                raise AttributeError("Selected layer has no .self_attn")
            return layer.self_attn
        if self.cfg.layer_select == "MLP":
            if not hasattr(layer, "mlp"):
                raise AttributeError("Selected layer has no .mlp")
            return layer.mlp

        raise RuntimeError("Unreachable")

    # -----------------------------
    # Hook output normalization
    # -----------------------------
    @staticmethod
    def _pick_tensor_from_output(output) -> Optional[torch.Tensor]:
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
        prompt_len_ids = int(prompt_inputs["input_ids"].shape[1])

        # 2) generate continuation tokens
        gen_kwargs = dict(
            max_new_tokens=int(self.cfg.max_new_tokens),
            do_sample=bool(self.cfg.do_sample),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # Only pass temperature/top_p if sampling is enabled (avoids warning)
        if self.cfg.do_sample:
            gen_kwargs["temperature"] = float(self.cfg.temperature)
            gen_kwargs["top_p"] = float(self.cfg.top_p)

        gen_ids = self.model.generate(**prompt_inputs, **gen_kwargs)

        seq_len_ids = int(gen_ids.shape[1])
        gen_len_ids = seq_len_ids - prompt_len_ids

        # 3) run forward on full sequence with a hook
        full_inputs = {
            "input_ids": gen_ids,
            "attention_mask": torch.ones_like(gen_ids, device=self.device),
        }

        hook_module = self._get_hook_module()
        h = hook_module.register_forward_hook(hook_fn)
        try:
            # Force non-cached full forward to reduce “short sequence” hook outputs
            _ = self.model(**full_inputs, use_cache=False)
        finally:
            h.remove()

        if not captured:
            raise RuntimeError("No hidden states captured. Hook path/output did not match expectations.")

        hs = captured[0]
        hook_shape = tuple(hs.shape)

        # Normalize hs into [T, H]
        if hs.dim() == 3:      # [B, T, H]
            hs_all = hs[0]
        elif hs.dim() == 2:    # [T, H]
            hs_all = hs
        else:
            vec = hs.reshape(-1)
            return vec.numpy(), {
                "prompt_len_ids": prompt_len_ids,
                "seq_len_ids": seq_len_ids,
                "gen_len_ids": gen_len_ids,
                "seq_len_hs": int(vec.numel()),
                "hook_shape": hook_shape,
                "hidden_dim": int(vec.numel()),
                "layer_idx": self.cfg.layer_idx,
                "layer_select": self.cfg.layer_select,
                "extract_mode": self.cfg.extract_mode,
            }

        # IMPORTANT FIX:
        # Use captured tensor length, not seq_len_ids, for indexing
        T = int(hs_all.shape[0])
        H = int(hs_all.shape[1])

        # prompt_len in the captured tensor may be <= prompt_len_ids if hook output is shorter
        prompt_len_eff = min(prompt_len_ids, T)

        # If nothing beyond prompt is available in hs_all, fall back to last available token
        if T <= 0:
            raise RuntimeError("Captured hidden states have zero length.")
        if prompt_len_eff >= T:
            vec_t = hs_all[T - 1, :]
            used_mode = "fallback_last_available"
        else:
            gen_start = prompt_len_eff
            gen_end = T
            if self.cfg.extract_mode == "last_gen":
                vec_t = hs_all[T - 1, :]
                used_mode = "last_gen"
            elif self.cfg.extract_mode == "first_gen":
                vec_t = hs_all[gen_start, :]
                used_mode = "first_gen"
            else:  # mean_gen
                vec_t = hs_all[gen_start:gen_end, :].mean(dim=0)
                used_mode = "mean_gen"

        vec = vec_t.numpy()
        hidden_dim = int(vec.shape[0])

        meta = {
            "prompt_len_ids": prompt_len_ids,
            "seq_len_ids": seq_len_ids,
            "gen_len_ids": gen_len_ids,
            "seq_len_hs": T,
            "hook_shape": hook_shape,
            "hidden_dim": hidden_dim,
            "layer_idx": self.cfg.layer_idx,
            "layer_select": self.cfg.layer_select,
            "extract_mode": self.cfg.extract_mode,
            "used_mode": used_mode,
            "H": H,
        }
        return vec, meta
