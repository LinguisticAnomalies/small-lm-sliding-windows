'''
Evalaute the sliding perplexity for AVH transcripts
'''
import argparse
import os
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
        "--avh", action="store_true",
        help="""indicator for using AVH dataset""")
    parser.add_argument(
        "--ellen", action="store_true",
        help="""indicator for using Ellen's dataset""")
    return parser.parse_args()


@dataclass
class TokenizerOutput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


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


def encode_transcripts(transcripts: List[str], tokenizer, batch_size=64) -> List[TokenizerOutput]:
    """
    Encode transcripts using batched processing.
    
    Args:
        transcripts: List of transcript strings
        tokenizer: HuggingFace tokenizer
        batch_size: Number of transcripts to process at once
    
    Returns:
        List of TokenizerOutput objects
    """
    encodings_list = []
    
    # Process transcripts in batches
    for i in tqdm(range(0, len(transcripts), batch_size), desc="Encoding transcripts"):
        batch_transcripts = transcripts[i:i + batch_size]
        batch_encodings = tokenizer(
            batch_transcripts,
            truncation=True,
            max_length=1024,
            padding=False,  # We'll handle padding separately if needed
            return_tensors=None  # Return list of lists instead of tensors
        )
        for j in range(len(batch_transcripts)):
            # Convert to tensors efficiently
            input_ids = torch.tensor([batch_encodings['input_ids'][j]], dtype=torch.long)
            attention_mask = torch.tensor([batch_encodings['attention_mask'][j]], dtype=torch.long)
            
            output = TokenizerOutput(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            encodings_list.append(output)
    
    return encodings_list



def calculate_batch_perplexity(
        model, encodings_list, device, window_size, batch_size=8) -> Tuple[List[float], List[List[float]]]:
    """
    Calculate both average and per-window perplexities for multiple transcripts.
    
    Returns:
        Tuple containing:
        - List of average perplexities for each transcript
        - List of lists containing all window perplexities for each transcript
    """
    all_perplexities = []
    all_window_perplexities = []  # Store per-window perplexities for each transcript
    
    for batch_start in tqdm(range(0, len(encodings_list), batch_size), desc="Processing transcripts"):
        batch_end = min(batch_start + batch_size, len(encodings_list))
        batch_encodings = encodings_list[batch_start:batch_end]
        batch_perplexities = []
        batch_window_perplexities = []  # Store window perplexities for current batch
        
        # Each transcript
        for encodings in batch_encodings:
            seq_len = encodings.input_ids.size(1)
            
            # Handle short sequences with global perplexity
            if seq_len < window_size:
                input_ids = encodings.input_ids.to(device)
                target_ids = input_ids.clone()
                
                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs.loss
                    
                ppl = torch.exp(neg_log_likelihood)
                batch_perplexities.append(ppl.item())
                batch_window_perplexities.append([ppl.item()])  # Single window for short sequences
                continue
            
            # Process longer sequences with sliding windows
            window_nlls = []
            window_ppls = []  # Store perplexity for each window
            window_batch_size = 16
            window_positions = list(range(0, seq_len - window_size + 1))
            
            for window_batch_start in range(0, len(window_positions), window_batch_size):
                window_batch_end = min(window_batch_start + window_batch_size, len(window_positions))
                current_windows = window_positions[window_batch_start:window_batch_end]
                
                batch_input_ids = []
                batch_target_ids = []
                
                for begin_loc in current_windows:
                    end_loc = begin_loc + window_size
                    window_input_ids = encodings.input_ids[:, begin_loc:end_loc]
                    batch_input_ids.append(window_input_ids)
                    batch_target_ids.append(window_input_ids.clone())
                
                batch_input_ids = torch.cat(batch_input_ids, dim=0).to(device)
                batch_target_ids = torch.cat(batch_target_ids, dim=0).to(device)
                
                with torch.no_grad():
                    outputs = model(batch_input_ids, labels=batch_target_ids)
                    neg_log_likelihoods = outputs.loss.view(-1)
                    
                    # Calculate perplexity for each window
                    window_perplexities = torch.exp(neg_log_likelihoods)
                    window_ppls.extend(window_perplexities.cpu().tolist())
                    window_nlls.extend(neg_log_likelihoods.cpu().tolist())
            
            # Calculate average perplexity for the transcript
            avg_nll = sum(window_nlls) / len(window_nlls)
            avg_ppl = torch.exp(torch.tensor(avg_nll))
            
            batch_perplexities.append(avg_ppl.item())
            batch_window_perplexities.append(window_ppls)
        
        all_perplexities.extend(batch_perplexities)
        all_window_perplexities.extend(batch_window_perplexities)
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return all_perplexities, all_window_perplexities


def process_transcripts(
        trans_df, model_card, col, device, suffix, window, output_dir,
        encoding_batch_size=64, perplexity_batch_size=8):
    """Process transcripts with optimized batch processing."""
    model_name = model_card.split("/")[1]
    output_path = f"{output_dir}{model_name}-{suffix}.jsonl"
    if not os.path.exists(output_path):
        model, tokenizer = setup_model_and_tokenizer(model_card=model_card, device=device)
        
        transcripts = trans_df[col].values.tolist()
        encodings_list = encode_transcripts(
            transcripts, 
            tokenizer, 
            batch_size=encoding_batch_size
        )
        
        avg_ppls, window_ppls = calculate_batch_perplexity(
            model=model,
            encodings_list=encodings_list,
            device=device,
            batch_size=perplexity_batch_size,
            window_size=window
        )
        
        trans_df[f"{model_name}_avg_ppl"] = avg_ppls
        trans_df[f"{model_name}_window_ppls"] = window_ppls
        trans_df.to_json(output_path, orient="records", lines=True)
    else:
        pass


def convert_to_linux_encoding(text):
    # Dictionary of common Windows special characters to Linux equivalents
    char_map = {
        '\u2019': "'",  # right single quotation
        '\u2018': "'",  # left single quotation
        '\u201c': '"',  # left double quotation
        '\u201d': '"',  # right double quotation
        '\u2013': '-',  # en dash
        '\u2014': '--', # em dash
        '\u2026': '...' # ellipsis
    }
    
    # Replace special characters
    for win_char, lin_char in char_map.items():
        text = text.replace(win_char, lin_char)
    
    # Convert to utf-8 and back to ensure Linux compatibility
    try:
        text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    
    return text


if __name__ == "__main__":
    pargs = parge_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cards = ("EleutherAI/pythia-70m-deduped","EleutherAI/pythia-160m-deduped",
                   "EleutherAI/pythia-410m-deduped","EleutherAI/pythia-1b-deduped",
                   "EleutherAI/pythia-1.4b-deduped","EleutherAI/pythia-2.8b-deduped",
                   "EleutherAI/pythia-6.9b-deduped","EleutherAI/pythia-12b-deduped")
    window_sizes = (8, 16, 32, 64, 128)
    if pargs.ellen:
        suffix = "ellen"
        col = "answer"
        if os.path.exists("../data/ellen.jsonl"):
            label_df = pd.read_json("../data/ellen.jsonl", lines=True)
        else:
            label_df = pd.read_pickle('/edata/george/cleanedporqQA_reduced.pkl')
            label_df = label_df[['file', 'answer', 'disorganization_subscale']]
            label_df['answer'] = (label_df['answer']
                        .apply(convert_to_linux_encoding)
                        .str.replace(r'<s>(.*?)</s>', r'\1', regex=True)
                        .str.replace(r'\s+', ' ', regex=True)
                        .str.strip())
            label_df.to_json("../data/ellen.jsonl", orient="records", lines=True)
    if pargs.avh:
        suffix = "label"
        col = "text"
        label_df = pd.read_json("../data/avh_tald.jsonl", lines=True)
    for window_size in window_sizes:
        for model_card in model_cards:
            output_dir = f"../data/no-ema/sliding_windows/{window_size}/"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Calculating {model_card} for {window_size} PPL sliding windows")
            process_transcripts(label_df, model_card, col, device, 
                                suffix, window_size, output_dir)
