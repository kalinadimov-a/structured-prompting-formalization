import argparse
import gc
from pathlib import Path
from typing import List, Sequence, Tuple
import warnings

import nltk
import pandas as pd
import torch
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score


# Constants describing dataset column names
TEXT_COL_INFORMAL = "Style 1"
TEXT_COL_REFERENCE = "Style 2"

# Prompt template
ORIGINAL_PROMPT_TEMPLATE = ("""
    You are an expert in text refinement.
    Your task is to convert informal text into professional, grammatically correct, and well-structured English while preserving meaning.

    Follow these steps:
    1. Identify informal words, slang, abbreviations, and grammatical errors.
    2. Replace informal words with their formal equivalents.
    3. Correct any grammar, punctuation, and spelling mistakes.
    4. Ensure proper capitalization and sentence structure.
    5. Maintain the original intent of the message.

    **Example Input:** "Familt Guy, The Simpons, Futurama, and South Park!!!!"
    **Example Output:** "The best cartoons are ""The Family Guy"", ""The Simpsons"", ""Futurama"", and ""South Park""."

    **Example Input:** "yahoo music, you pay like $10 a month for unlimited downloads"
    **Example Output:** "On Yahoo Music, you pay approximately $10 for unlimited downloads."

    **Example Input:** "(or w/e) p.s gurl how old r u ?"
    **Example Output:** "How old are you?"

    **Now process this sentence:** "{informal_text}"
"""
)

# Pre-split template to avoid rebuilding a large string each time
_prompt_prefix, _prompt_suffix = ORIGINAL_PROMPT_TEMPLATE.split('{informal_text}')

# Metrics setup
rouge_metric = Rouge()
nltk.download('wordnet', quiet=True)

# Utilities
def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def set_seed(seed: int, deterministic: bool = True):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_data(files: Sequence[str]) -> pd.DataFrame:
    splits = ["train", "val", "test"]
    dfs = []
    for split_name, fp in zip(splits, files):
        if not Path(fp).exists():
            raise FileNotFoundError(f"Missing data file: {fp}")
        df = pd.read_csv(fp, delimiter="	")
        if TEXT_COL_INFORMAL not in df.columns or TEXT_COL_REFERENCE not in df.columns:
            raise ValueError(
                f"Data file {fp} must contain columns '{TEXT_COL_INFORMAL}' and '{TEXT_COL_REFERENCE}'."
            )
        df["__split__"] = split_name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def build_prompts(batch: Sequence[str]) -> List[str]:
    return [f"{_prompt_prefix}{txt}{_prompt_suffix}" for txt in batch]


def generate_batch(
    model,
    tokenizer,
    batch_texts: Sequence[str],
    *,
    max_new_tokens: int,
    deterministic: bool,
) -> List[str]:
    """Generate outputs for a batch of inputs."""
    prompts = build_prompts(batch_texts)

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=not deterministic,
        num_beams=1,  # greedy when do_sample=False
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        use_cache=True,
    )
    # Only set temperature when sampling is enabled to avoid warnings
    if gen_kwargs["do_sample"]:
        gen_kwargs["temperature"] = 0.7

    with torch.inference_mode():
        outputs = model.generate(**enc, **gen_kwargs)

    # Decoder-only: generated includes prompt; need to cut input length
    if not getattr(model.config, "is_encoder_decoder", False):
        generated_texts = []
        input_ids = enc["input_ids"]
        for inp_ids, out_ids in zip(input_ids, outputs):
            gen_part = out_ids[len(inp_ids):]
            text = tokenizer.decode(gen_part, skip_special_tokens=True)
            generated_texts.append(text.strip())
    else:
        # Encoder-decoder: whole decode is target side only
        generated_texts = [tokenizer.decode(seq, skip_special_tokens=True).strip() for seq in outputs]

    return generated_texts


# Compute BLEU, ROUGE-1/2/L, METEOR, BERTScore
def compute_metrics(
    preds: Sequence[str],
    refs: Sequence[str],
    *,
    bert_batch_size: int = 64,
) -> pd.DataFrame:

    # Tokenizers
    def tok(x: str) -> List[str]:
        # Whitespace tokenization
        return x.split()

    sf = SmoothingFunction().method1

    bleu_scores = []
    rouge_scores = []
    meteor_scores_list = []
    kept_preds = []
    kept_refs = []

    for p, r in zip(preds, refs):
        if not p.strip():
            # Skip empty predictions
            bleu_scores.append(0.0)
            rouge_scores.append({"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}})
            meteor_scores_list.append(0.0)
        else:
            bleu_scores.append(sentence_bleu([tok(r)], tok(p), smoothing_function=sf))
            rouge_scores.append(rouge_metric.get_scores(p, r)[0])
            meteor_scores_list.append(meteor_score([tok(r)], tok(p)))
            kept_preds.append(p)
            kept_refs.append(r)

    # BERTScore (only on non-empty predictions)
    if kept_preds:
        P, R, F1 = bert_score(
            kept_preds,
            kept_refs,
            lang="en",
            model_type="bert-base-uncased",
            batch_size=bert_batch_size,
        )
        bert_f1 = F1.tolist()
    else:
        bert_f1 = [0.0] * len(preds)

    # Fill BERT scores back into correct positions (0 for empty preds)
    bert_iter = iter(bert_f1)
    final_bert_scores = [next(bert_iter) if p.strip() else 0.0 for p in preds]

    return pd.DataFrame(
        {
            "BLEU": bleu_scores,
            "ROUGE-1": [s["rouge-1"]["f"] for s in rouge_scores],
            "ROUGE-2": [s["rouge-2"]["f"] for s in rouge_scores],
            "ROUGE-L": [s["rouge-l"]["f"] for s in rouge_scores],
            "METEOR": meteor_scores_list,
            "BERTScore": final_bert_scores,
        }
    )


# Model loading
def infer_model_family(name: str, config) -> str:
    lname = name.lower()
    if "t5" in lname:
        return "t5"
    if "falcon" in lname:
        return "falcon"
    if "opt" in lname:
        return "opt"
    # fallback: inspect config
    return getattr(config, "model_type", "")


def build_device_map(device_index: int):
    if torch.cuda.is_available():
        return {"": f"cuda:{device_index}"}
    return {"": "cpu"}


def pick_attn_impl(model_name: str) -> str:
    lname = model_name.lower()
    # Force eager for models known to not support flash and where SDPA may be safest
    if "falcon" in lname or "opt" in lname:
        return "eager"
    # Prefer PyTorch SDPA when available; fallback to eager
    try:
        import torch.nn.functional as F
        return "sdpa" if hasattr(F, "scaled_dot_product_attention") else "eager"
    except Exception:
        return "eager"


def load_model_and_tokenizer(model_name: str, device_index: int, dtype: torch.dtype):
    print(f"=== Loading {model_name} ===")

    # Config first for arch detection
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
    family = infer_model_family(model_name, config)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Special handling for OPT
    if family == "opt":
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Padding side: left for decoder-only, right for encoder-decoder
    elif family == "t5" or getattr(config, "is_encoder_decoder", False):
        tokenizer.padding_side = "right"
    else:
        tokenizer.padding_side = "left"

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # Model
    device_map = build_device_map(device_index)
    attn_impl = pick_attn_impl(model_name)

    if family == "t5" or getattr(config, "is_encoder_decoder", False):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_impl,
            trust_remote_code=False,
        )

    # If we added a new pad token, ensure embeddings are resized
    if hasattr(model, "resize_token_embeddings") and len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    model.eval()
    return model, tokenizer


# Main logic
def main():
    parser = argparse.ArgumentParser(description="Plan-and-Solve formalization evaluation")
    parser.add_argument("--train", default="train_en.txt", help="Train TSV file")
    parser.add_argument("--val", default="val_en.txt", help="Val TSV file")
    parser.add_argument("--test", default="test_en.txt", help="Test TSV file")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["tiiuae/falcon-7b-instruct", "google/flan-t5-large", "facebook/opt-1.3b"],
        help="HF model names",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", help="Enable PyTorch deterministic algorithms")
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--out_prefix", default="ps_prompting_results", help="Output file prefix (without extension)")
    args = parser.parse_args()

    dtype = pick_dtype()
    set_seed(args.seed, deterministic=args.deterministic)

    # Load data
    print("Loading data ...")
    data = load_data([args.train, args.val, args.test])
    informal_texts: List[str] = data[TEXT_COL_INFORMAL].astype(str).tolist()
    references: List[str] = data[TEXT_COL_REFERENCE].astype(str).tolist()

    all_results = data.copy()

    for model_name in args.models:
        model, tokenizer = load_model_and_tokenizer(model_name, args.device_index, dtype)

        predictions: List[str] = []
        print("Generating (batched)...")
        for i in tqdm(range(0, len(informal_texts), args.batch_size), desc="Batches"):
            batch = informal_texts[i : i + args.batch_size]
            batch_preds = generate_batch(
                model,
                tokenizer,
                batch,
                max_new_tokens=args.max_new_tokens,
                deterministic=True,  # keep greedy by default
            )
            predictions.extend(batch_preds)

        col_pred = f"Pred_{model_name}"
        all_results[col_pred] = predictions

        print("Computing metrics ...")
        metrics_df = compute_metrics(predictions, references, bert_batch_size=args.batch_size)
        for metric_col in metrics_df.columns:
            all_results[f"{metric_col}_{model_name}"] = metrics_df[metric_col]

        # Print summary
        summary = metrics_df.mean()
        print(f"--- Averages for {model_name} ---")
        for m_name, val in summary.items():
            print(f"{m_name}: {val:.4f}")

        # Cleanup before next model
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Per-split aggregation
    print("Aggregating per-split metrics ...")
    summary_rows = []
    for model_name in args.models:
        for split_name in ["train", "val", "test"]:
            mask = all_results["__split__"] == split_name
            row = {"model": model_name, "split": split_name}
            for metric in ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR", "BERTScore"]:
                row[metric] = all_results.loc[mask, f"{metric}_{model_name}"].mean()
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    per_split_path = f"{args.out_prefix}_per_split.csv"
    full_path = f"{args.out_prefix}.csv"

    summary_df.to_csv(per_split_path, index=False)
    all_results.to_csv(full_path, index=False)

    print(f"Saved: {full_path} and {per_split_path}")
    print("Done.")


if __name__ == "__main__":
    # Suppress noisy but harmless warnings
    warnings.filterwarnings(
        "ignore",
        message=".*_register_pytree_node is deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*resume_download.*",
        category=FutureWarning,
    )

    print("--------------- Running PS prompting sript ---------------")

    main()
