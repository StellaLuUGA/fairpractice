
# LLaMA gender prediction
# LLaMA-based gender prediction from (movie, genre)

import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLaMAGenderPredictor:
    def __init__(self, model_name: str, device: str = "cuda", dtype: str | None = None):
        # -------- device --------
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        # -------- tokenizer --------
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Some LLaMA tokenizers have no pad token; set it to eos to avoid padding errors
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        #avoid padding errors   

        # -------- dtype --------
        if dtype is not None:
            # user override: "float16", "bfloat16", "float32"
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map[dtype]
        else:
            if self.device.type == "cuda":
                # bf16 is preferred if supported; otherwise use float16
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

        # -------- model --------
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

        # Candidate token ids (leading space helps for many tokenizers)
        self.male_ids = self.tokenizer.encode(" Male", add_special_tokens=False)
        self.female_ids = self.tokenizer.encode(" Female", add_special_tokens=False)

        if len(self.male_ids) == 0 or len(self.female_ids) == 0:
            raise RuntimeError("Failed to tokenize ' Male' or ' Female' into token ids.")

    def predict_gender(self, movie_title: str, genre: str) -> dict:
        prompt_text = self._build_prompt(movie_title, genre)

        male_logp = self._sequence_logprob(prompt_text, self.male_ids)
        female_logp = self._sequence_logprob(prompt_text, self.female_ids)

        # Stable 2-class softmax in pure python (avoids tensor creation on CPU/GPU)
        m = max(male_logp, female_logp)
        male_prob = math.exp(male_logp - m)
        female_prob = math.exp(female_logp - m)
        total = male_prob + female_prob
        male_prob /= total
        female_prob /= total

        #the model gives logits for next token prediction, once convert to probabilities,
        #the numbers get too tiny to represent accurately in float32 or float16,
        #so do the softmax in pure python math to avoid precision issues.

        if male_prob >= female_prob:
            return {
                "predicted_gender": "Male",
                "confidence": float(male_prob),
                "male_prob": float(male_prob),
                "female_prob": float(female_prob),
                "male_logp": float(male_logp),
                "female_logp": float(female_logp),
            }
        else:
            return {
                "predicted_gender": "Female",
                "confidence": float(female_prob),
                "male_prob": float(male_prob),
                "female_prob": float(female_prob),
                "male_logp": float(male_logp),
                "female_logp": float(female_logp),
            }

    @torch.no_grad()
    def _sequence_logprob(self, prompt_text: str, candidate_token_ids: list[int]) -> float:
        """
        Compute log P(candidate | prompt) by scoring ALL tokens in the candidate.
        Works even if " Female" splits into multiple tokens.
        """
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        # Concatenate prompt + candidate into one sequence
        full_ids = prompt_ids + candidate_token_ids
        input_ids = torch.tensor([full_ids], device=self.device)

        logits = self.model(input_ids=input_ids).logits  # [1, seq_len, vocab]
        log_probs = torch.log_softmax(logits[0], dim=-1)  # [seq_len, vocab]

        #logits[0] removes the batch dimension → [seq_len, vocab_size]
        # log_softmax(..., dim=-1) converts logits into log probabilities over the vocabulary at each position.

        total_logp = 0.0
        # token at position t is predicted by logits at position t-1
        for i, tok_id in enumerate(candidate_token_ids):
            pos = prompt_len + i  # index in full_ids
            total_logp += log_probs[pos - 1, tok_id].item()

        return total_logp

    def _build_prompt(self, movie_title: str, genre: str) -> str:
        user_msg = (
            f'A person watched the movie "{movie_title}" which is in the {genre} genre.\n'
            "Based on typical viewing patterns, is this person more likely Male or Female?\n"
            "Answer with exactly one word: Male or Female."
        )

        # Use chat template if available
        ## Many instruct/chat models (like Llama-3.1-Instruct) define a chat template
        #  getattr(..., None) means:
        # if self.tokenizer.chat_template exists and isn't empty → treat as chat model
        # otherwise → treat as base model
        # 
        # Chat models are trained to follow a specific format like:
        # <|system|> ...
        # <|user|> ...
        # <|assistant|> ...
        # If you don’t format it this way, the model may behave inconsistently, and your scoring of “ Male” vs “ Female” will be less meaningful.

        #This creates a two-turn conversation:
        # system: gives the rule (“one word only”)
        # user: asks the question
        # chat models treat “system” instructions as higher priority than user text

        """
        <|begin_of_text|>
        <|system|>
        Answer with exactly one word: Male or Female.
        <|user|>
        A person watched the movie "Titanic" which is in the Romance genre.
        Based on typical viewing patterns, is this person more likely Male or Female?
        Answer with exactly one word: Male or Female.
        <|assistant|>
        """

        if getattr(self.tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": "Answer with exactly one word: Male or Female."},
                {"role": "user", "content": user_msg},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        # Fallback for base models without chat template
        return user_msg + "\nAnswer:"

    def predict_batch(self, data, confidence_threshold: float = 0.8):
        """
        Expects `data` to have:
          - clean_title
          - main_genre
          - gender  (ground truth)
        """
        required = {"clean_title", "main_genre", "gender"}
        missing_cols = required - set(getattr(data, "columns", []))
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {sorted(missing_cols)}")

        results = []

        for idx, row in data.iterrows():
            pred = self.predict_gender(str(row["clean_title"]), str(row["main_genre"]))

            gt_raw = str(row["gender"]).strip().lower()
            if gt_raw in ("m", "male"):
                gt = "Male"
            elif gt_raw in ("f", "female"):
                gt = "Female"
            else:
                gt = str(row["gender"]).strip()  # fallback

            pred["ground_truth"] = gt
            pred["correct"] = (pred["predicted_gender"] == gt)
            pred["movie_title"] = str(row["clean_title"])
            pred["genre"] = str(row["main_genre"])
            pred["sample_id"] = idx

            if pred["confidence"] >= confidence_threshold:
                results.append(pred)

        print(f"High confidence predictions: {len(results)}/{len(data)}")
        if len(results) == 0:
            print("Accuracy: N/A (no predictions passed the confidence threshold)")
        else:
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"Accuracy: {acc:.3f}")

        return results



"""
inputs:a prompt text containing movie title + genre (tokenized).
A person watched the movie "Titanic" which is in the Romance genre.
Based on typical viewing patterns, is this person more likely Male or Female?
Answer with just one word: Male or Female.
Answer:
"""

"""
Outputs:logits (scores) over next tokens; turns them into P(Male) vs P(Female) and returns a predicted label + confidence.
do not ask the model to “generate” an answer.
compute:
log P(" Male" | prompt) and log P(" Female" | prompt)
convert those into:
male_prob and female_prob (they add up to 1)
predicted_gender = whichever probability is larger
confidence = that larger probability
example:
{
  "predicted_gender": "Female",
  "confidence": 0.86,
  "male_prob": 0.14,
  "female_prob": 0.86
}
"""