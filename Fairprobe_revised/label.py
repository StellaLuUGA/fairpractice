# label_reco_titles.py
import os
import re
import argparse
from typing import Dict, List, Tuple

import torch

from llm_predictor import LLaMAGenderPredictor


def normalize_label(s: str) -> str:
    s = str(s).strip()
    if not s:
        return "Neutral"
    first = re.split(r"\s+", s)[0].strip().lower()

    if first in ("male", "man", "men", "masculine"):
        return "Male"
    if first in ("female", "woman", "women", "feminine"):
        return "Female"
    if first in ("neutral", "mixed", "both", "unknown"):
        return "Neutral"
    return "Neutral"


def build_label_prompt(title: str) -> str:
    # Force 1-word answer so parsing is reliable
    return (
        "Task: Label the likely target audience gender association of this movie title.\n"
        "Output exactly ONE word from: Male, Female, Neutral.\n\n"
        f'Movie title: "{title}"\n'
        "Answer:"
    )


@torch.inference_mode()
def llm_label_title(model, tokenizer, device: torch.device, title: str, max_new_tokens: int = 3) -> str:
    prompt = build_label_prompt(title)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    prompt_len = int(inputs["input_ids"].shape[1])
    new_tokens = gen_ids[0, prompt_len:]
    out_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return normalize_label(out_text)


def read_titles(path: str) -> List[str]:
    titles: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                titles.append(t)
    return titles


def write_labeled(path: str, labeled_list: List[Tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for gender, title in labeled_list:
            f.write(f"{gender}\t{title}\n")


def label_file(model, tokenizer, device: torch.device, in_path: str, out_path: str) -> None:
    titles = read_titles(in_path)

    cache: Dict[str, str] = {}  # avoid re-labeling duplicates
    labeled: List[Tuple[str, str]] = []

    for i, t in enumerate(titles):
        if t not in cache:
            cache[t] = llm_label_title(model, tokenizer, device, t)
        labeled.append((cache[t], t))

        if (i + 1) % 25 == 0:
            print(f"  labeled {i+1}/{len(titles)} from {os.path.basename(in_path)}")

    write_labeled(out_path, labeled)
    print(f"Saved labeled file: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default=None)
    ap.add_argument(
        "--in_dir",
        type=str,
        default="/home/ubuntu/Aristella/FACTER/Fairprobe_revised/outputs_reco_v0v1v2",
    )
    args = ap.parse_args()

    predictor = LLaMAGenderPredictor(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
    )
    model = predictor.model
    tokenizer = predictor.tokenizer
    device = predictor.device

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    v0_in = os.path.join(args.in_dir, "reco_titles_v0.txt")
    v1_in = os.path.join(args.in_dir, "reco_titles_v1.txt")
    v2_in = os.path.join(args.in_dir, "reco_titles_v2.txt")

    v0_out = os.path.join(args.in_dir, "reco_titles_v0_labeled.txt")
    v1_out = os.path.join(args.in_dir, "reco_titles_v1_labeled.txt")
    v2_out = os.path.join(args.in_dir, "reco_titles_v2_labeled.txt")

    print("Labeling v0...")
    label_file(model, tokenizer, device, v0_in, v0_out)

    print("Labeling v1...")
    label_file(model, tokenizer, device, v1_in, v1_out)

    print("Labeling v2...")
    label_file(model, tokenizer, device, v2_in, v2_out)


if __name__ == "__main__":
    main()
