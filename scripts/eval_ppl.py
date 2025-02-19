import os
import argparse
import gc
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def parge_args():
    """
    add argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ema", action="store_true",
        help="""if set, then limit the records when there are matched EMA"""
    )
    parser.add_argument(
        "--avh", action="store_true",
        help="""indicator for using AVH dataset""")
    parser.add_argument(
        "--ellen", action="store_true",
        help="""indicator for using Ellen's dataset""")
    return parser.parse_args()


def setup_model_and_tokenizer(model_card, device="cuda"):
    """Setup the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_card)
    model = AutoModelForCausalLM.from_pretrained(model_card)
    
    # Handle special tokens if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def calculate_perplexity(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int = 2048
) -> float:
    """Calculate perplexity score for a given text"""
    encodings = tokenizer(
        text,
        return_tensors="pt",
        return_attention_mask=True,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    with torch.no_grad():
        outputs = model(**encodings, labels=encodings['input_ids'])
        
    return torch.exp(outputs[0]).item()


def process_transcripts(
        trans_df, output_file, col):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cards = ("EleutherAI/pythia-70m-deduped","EleutherAI/pythia-160m-deduped",
                   "EleutherAI/pythia-410m-deduped","EleutherAI/pythia-1b-deduped",
                   "EleutherAI/pythia-1.4b-deduped","EleutherAI/pythia-2.8b-deduped",
                   "EleutherAI/pythia-6.9b-deduped","EleutherAI/pythia-12b-deduped")
    if not os.path.exists(output_file):
        for model_card in model_cards:
            model_name = model_card.split("/")[1]
            model, tokenizer = setup_model_and_tokenizer(model_card=model_card, device=device)
            print(f"Calculating {model_card} for PPL")
            raw_ppls = []
            for _, row in tqdm(trans_df.iterrows(), 
                            total=len(trans_df), desc="Processing transcriptions"):
                raw_ppl = calculate_perplexity(
                    row[col].lower().strip(),
                    model,
                    tokenizer,
                    device
                )
                raw_ppls.append(raw_ppl)
            trans_df[f"{model_name}_ppl"] = raw_ppls
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        trans_df.to_json(output_file, orient="records", lines=True)
    else:
        ppl_df = pd.read_json(output_file, lines=True)


if __name__ == "__main__":
    pargs = parge_args()
    if pargs.avh:
        col = "text"
        output_path = f"../data/no-ema/pythia.jsonl"
        label_df = pd.read_json(
            path_or_buf="../data/avh_tald.jsonl", lines=True)
    if pargs.ellen:
        col = "answer"
        output_path = f"../data/no-ema/pythia_ellen.jsonl"
        label_df = pd.read_json("../data/ellen.jsonl", lines=True)
    process_transcripts(label_df, output_path, col)
    