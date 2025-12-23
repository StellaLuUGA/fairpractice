# LLaMA gender prediction

"""LLaMA-based gender prediction from (movie, genre)"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLaMAGenderPredictor:
    def __init__(self, model_name, device="cuda"):
        # Device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Some LLaMA tokenizers have no pad token; set it to eos to avoid padding errors
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Dtype for single GPU
        # Prefer bfloat16 on GPU (common for Llama-3.1), else float16; CPU uses float32
        if self.device.type == "cuda":
            dtype = torch.bfloat16
            # if your GPU doesn't support bf16 well, switch to float16:
            # dtype = torch.float16
        else:
            dtype = torch.float32
        # Load model fully onto ONE device (no device_map="auto")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

        # Label token IDs (leading space helps tokenization for many LLM tokenizers)
        self.male_ids = self.tokenizer.encode(" Male", add_special_tokens=False)
        self.female_ids = self.tokenizer.encode(" Female", add_special_tokens=False)

        
    def predict_gender(self, movie_title, genre):
        # Build prompt text (chat template if available)
        prompt_text = self._build_prompt(movie_title, genre)

        male_logp = self._sequence_logprob(prompt_text, self.male_ids)
        female_logp = self._sequence_logprob(prompt_text, self.female_ids)

        # Convert two log-probs into normalized probabilities (2-class softmax)
        m = max(male_logp, female_logp)
        male_prob = float(torch.exp(torch.tensor(male_logp - m)))
        female_prob = float(torch.exp(torch.tensor(female_logp - m)))
        total = male_prob + female_prob
        male_prob /= total
        female_prob /= total
    
        if male_prob >= female_prob:
            return {
                "predicted_gender": "Male",
                "confidence": male_prob,
                "male_prob": male_prob,
                "female_prob": female_prob,
            }
        else:
            return {
                "predicted_gender": "Female",
                "confidence": female_prob,
                "male_prob": male_prob,
                "female_prob": female_prob,
            }

    def _sequence_logprob(self, prompt_text, candidate_token_ids):
        """
        Compute log P(candidate | prompt) by scoring ALL tokens in the candidate.
        Fixes: "Female" may be multiple tokens.
        """
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        input_ids = torch.tensor([prompt_ids + candidate_token_ids], device=self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits  # [1, seq_len, vocab]
            log_probs = torch.log_softmax(logits[0], dim=-1)  # [seq_len, vocab]

            total_logp = 0.0
            for i, tok_id in enumerate(candidate_token_ids):
                pos = prompt_len + i
                total_logp += log_probs[pos - 1, tok_id].item()

        return total_logp
    
    def _build_prompt(self, movie_title, genre):

        user_msg = (
            f'A person watched the movie "{movie_title}" which is in the {genre} genre.\n'
            "Based on typical viewing patterns, is this person more likely Male or Female?\n"
            "Answer with exactly one word: Male or Female."
        )

        # Use chat template ONLY if it exists and is non-empty
        if getattr(self.tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": "Answer with exactly one word: Male or Female."},
                {"role": "user", "content": user_msg},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        # Base-model fallback (no chat template)
        return user_msg + "\nAnswer:"

    
    def predict_batch(self, data, confidence_threshold=0.8):
        results = []

        for idx, row in data.iterrows():
            pred = self.predict_gender(row["clean_title"], row["main_genre"])

            pred["ground_truth"] = row["gender"]
            pred["correct"] = (pred["predicted_gender"] == pred["ground_truth"])
            pred["movie_title"] = row["clean_title"]
            pred["genre"] = row["main_genre"]
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