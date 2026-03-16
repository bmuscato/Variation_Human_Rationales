"""
Usage:
    python compute_soft_faith.py \
        --model_dir ./models/best_model \
        --csv_path predictions.csv \
        --device cuda:0 \
        --batch_size 16 \
        --output_csv soft_faith_results.csv \
        --output_json soft_faith_results.json
"""

import os
import csv
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm



@dataclass
class Explanation:

    text: str
    scores: List[float]                 # model_rationales (importance per token position)
    hard_label_human: int = 0
    hard_label_pred: int = 0
    soft_labels_human: List[float] = field(default_factory=list)
    soft_labels_pred: List[float] = field(default_factory=list)
    human_rationales: List[float] = field(default_factory=list)


def parse_float_list(raw: str, sep: str = ",") -> List[float]:

    if not raw or raw.strip() in ("", "nan", "None"):
        return []
    # Auto-detect separator
    if ";" in raw:
        sep = ";"
    parts = raw.strip().split(sep)
    return [float(x.strip()) for x in parts if x.strip()]


def load_explanations_from_csv(csv_path: str) -> List[Explanation]:
 
    explanations = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_rat = parse_float_list(row.get("model_rationales", ""))
            if not model_rat:
                continue  # skip rows without rationales

            exp = Explanation(
                text=row["text"],
                scores=model_rat,
                hard_label_human=int(row.get("hard_label_human", 0)),
                hard_label_pred=int(row.get("hard_label_pred", 0)),
                soft_labels_human=parse_float_list(row.get("soft_labels_human", "")),
                soft_labels_pred=parse_float_list(row.get("soft_labels_pred", "")),
                human_rationales=parse_float_list(row.get("human_rationales", "")),
            )
            explanations.append(exp)
    return explanations


# =============================================================================
# Soft perturbation helpers
# =============================================================================

def soft_perturb_comprehensiveness(
    embeddings: torch.Tensor,
    importance_scores: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Soft perturbation for COMPREHENSIVENESS:
    Mask OUT important tokens — higher importance → higher chance of being zeroed.
    mask = Bernoulli(1 - normalised_importance) * attention_mask
    """
    importance_scores = importance_scores.unsqueeze(-1)
    attention_mask = attention_mask.unsqueeze(-1).float()

    # normalization 
    normalized_importance_scores = (importance_scores - importance_scores.min()) / (importance_scores.max() - importance_scores.min())

    # Bernoulli(1 - importance): important tokens likely zeroed
    mask = torch.bernoulli(1.0 - normalized_importance_scores)
    mask = mask * attention_mask
    return embeddings * mask


def soft_perturb_sufficiency(
    embeddings: torch.Tensor,
    importance_scores: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Soft perturbation for SUFFICIENCY:
    KEEP important tokens — higher importance → higher chance of being kept.
    mask = Bernoulli(normalised_importance) * attention_mask
    """
    importance_scores = importance_scores.unsqueeze(-1)
    attention_mask = attention_mask.unsqueeze(-1).float()

    normalized_importance_scores = (importance_scores - importance_scores.min()) / (importance_scores.max() - importance_scores.min())
        

    # Bernoulli(importance): important tokens likely kept
    mask = torch.bernoulli(normalized_importance_scores)
    mask = mask * attention_mask
    return embeddings * mask




def align_scores_to_tokens(
    scores: List[float],
    seq_len: int,
) -> torch.Tensor:
    """
    Align variable-length saliency scores to the tokenised sequence length.
    Pads with 0 or truncates as needed. Special tokens (CLS/SEP) get 0.
    """
    n = len(scores)
    if n >= seq_len:
        aligned = scores[:seq_len]
    else:
        aligned = scores + [0.0] * (seq_len - n)
    t = torch.tensor(aligned, dtype=torch.float32)
    # Zero out CLS (pos 0) and SEP (last non-pad, but we don't know exactly
    # where it is — the attention mask handles padding anyway)
    t[0] = 0.0
    return t


def compute_soft_comprehensiveness_batch(
    model, tokenizer, explanations: List[Explanation],
    device: str, max_len: int = 512,
) -> List[float]:
    """Compute soft comprehensiveness for a batch of explanations."""
    texts = [
        exp.text[0] if isinstance(exp.text, list) else exp.text
        for exp in explanations
    ]

    encoding = tokenizer(
        texts, padding=True, truncation=True,
        max_length=max_len, return_tensors="pt",
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}
    seq_len = encoding["input_ids"].shape[1]

    # Align saliency scores
    scores_batch = torch.stack([
        align_scores_to_tokens(exp.scores, seq_len)
        for exp in explanations
    ]).to(device)

    with torch.no_grad():
        # Original prediction
        orig_out = model(**encoding)
        orig_probs = F.softmax(orig_out.logits, dim=-1).cpu().numpy()
        full_probs = orig_probs.max(axis=-1)
        full_class = orig_probs.argmax(axis=-1)
        rows = np.arange(len(explanations))

        # Get embeddings from last hidden state
        orig_hidden = model(**encoding, output_hidden_states=True)
        embeddings = orig_hidden.hidden_states[-1]  # [B, seq_len, hidden]

        # Soft perturb: mask out important tokens
        perturbed_emb = soft_perturb_comprehensiveness(
            embeddings, scores_batch, encoding["attention_mask"].to(device)
        )

        # Convert perturbed embeddings back to token ids (nearest neighbour)
        perturbed_input = {k: v.clone() for k, v in encoding.items()}
        perturbed_input["input_ids"] = perturbed_emb.argmax(dim=-1)

        # Perturbed prediction
        pert_out = model(**perturbed_input)
        pert_probs = F.softmax(pert_out.logits, dim=-1).cpu().numpy()
        reduced_probs = pert_probs[rows, full_class]

    # Comprehensiveness = max(0, full - reduced)
    comp = np.maximum(0, full_probs - reduced_probs)
    return comp.tolist()


def compute_soft_sufficiency_batch(
    model, tokenizer, explanations: List[Explanation],
    device: str, max_len: int = 512,
) -> List[float]:
    """Compute soft normalised sufficiency for a batch of explanations."""
    texts = [
        exp.text[0] if isinstance(exp.text, list) else exp.text
        for exp in explanations
    ]

    encoding = tokenizer(
        texts, padding=True, truncation=True,
        max_length=max_len, return_tensors="pt",
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}
    seq_len = encoding["input_ids"].shape[1]

    scores_batch = torch.stack([
        align_scores_to_tokens(exp.scores, seq_len)
        for exp in explanations
    ]).to(device)

    with torch.no_grad():
        # Original prediction
        orig_out = model(**encoding)
        orig_probs = F.softmax(orig_out.logits, dim=-1).cpu().numpy()
        full_probs = orig_probs.max(axis=-1)
        full_class = orig_probs.argmax(axis=-1)
        rows = np.arange(len(explanations))

        # Get embeddings
        orig_hidden = model(**encoding, output_hidden_states=True)
        embeddings = orig_hidden.hidden_states[-1]

        # --- Perturbed (keep important tokens) ---
        perturbed_emb = soft_perturb_sufficiency(
            embeddings, scores_batch, encoding["attention_mask"].to(device)
        )
        perturbed_input = {k: v.clone() for k, v in encoding.items()}
        perturbed_input["input_ids"] = perturbed_emb.argmax(dim=-1)

        pert_out = model(**perturbed_input)
        pert_probs = F.softmax(pert_out.logits, dim=-1).cpu().numpy()
        reduced_probs = pert_probs[rows, full_class]

        # --- Baseline (all tokens zeroed) ---
        baseline_input = {k: v.clone() for k, v in encoding.items()}
        baseline_input["input_ids"] = torch.zeros_like(encoding["input_ids"]).to(device)

        base_out = model(**baseline_input)
        base_probs = F.softmax(base_out.logits, dim=-1).cpu().numpy()
        baseline_reduced = base_probs[rows, full_class]

    # Raw sufficiency = 1 - max(0, full - reduced)
    sufficiency = 1.0 - np.maximum(0, full_probs - reduced_probs)
    baseline_suff = 1.0 - np.maximum(0, full_probs - baseline_reduced)

    # Normalise: (suff - baseline) / (1 - baseline)
    baseline_suff -= 1e-4  # avoid division by zero
    normalised = np.maximum(0, (sufficiency - baseline_suff) / (1.0 - baseline_suff))
    normalised = np.clip(normalised, 0.0, 1.0)

    return normalised.tolist()


#main
def run_evaluation(
    model_dir: str,
    csv_path: str,
    device: str = "cpu",
    batch_size: int = 16,
    max_len: int = 512,
    n_runs: int = 5,
    output_csv: Optional[str] = None,
    output_json: Optional[str] = None,
):
    """
    Load model + CSV, compute soft comprehensiveness and soft sufficiency.

    Because the soft perturbation uses Bernoulli sampling, results vary
    between runs. We average over `n_runs` for stability.
    """

    # ── Load model and tokenizer ──────────────────────────────────────────
    print(f"Loading model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    print(f"  Model loaded on {device}")
    print(f"  Num labels: {model.config.num_labels}")

    # ── Load CSV ──────────────────────────────────────────────────────────
    print(f"\nLoading explanations from {csv_path}")
    explanations = load_explanations_from_csv(csv_path)
    print(f"  Loaded {len(explanations)} examples with rationales")

    if not explanations:
        print("No valid explanations found. Exiting.")
        return

    # ── Compute metrics (averaged over n_runs) ────────────────────────────
    all_comp_runs = []
    all_suff_runs = []

    for run in range(n_runs):
        print(f"\n── Run {run+1}/{n_runs} ──")
        comp_scores = []
        suff_scores = []

        for i in tqdm(range(0, len(explanations), batch_size),
                       desc="  Batches"):
            batch = explanations[i : i + batch_size]

            c = compute_soft_comprehensiveness_batch(
                model, tokenizer, batch, device, max_len
            )
            comp_scores.extend(c)

            s = compute_soft_sufficiency_batch(
                model, tokenizer, batch, device, max_len
            )
            suff_scores.extend(s)

        run_comp = np.mean(comp_scores)
        run_suff = np.mean(suff_scores)
        all_comp_runs.append(run_comp)
        all_suff_runs.append(run_suff)
        print(f"  Soft Comp ↑: {run_comp:.4f}  |  Soft Suff ↓: {run_suff:.4f}")

    # ── Aggregate ─────────────────────────────────────────────────────────
    mean_comp = np.mean(all_comp_runs)
    std_comp = np.std(all_comp_runs)
    mean_suff = np.mean(all_suff_runs)
    std_suff = np.std(all_suff_runs)

    print(f"\n{'='*60}")
    print(f"RESULTS  ({n_runs} runs)")
    print(f"{'='*60}")
    print(f"  Soft Comprehensiveness ↑:  {mean_comp:.4f} ± {std_comp:.4f}")
    print(f"  Soft Sufficiency ↓:        {mean_suff:.4f} ± {std_suff:.4f}")
    print(f"{'='*60}")

    # ── Save per-example results (from last run) ──────────────────────────
    if output_csv:
        print(f"\nSaving per-example results → {output_csv}")
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "text", "hard_label_human", "hard_label_pred",
                "soft_comp", "soft_suff",
            ])
            for idx, exp in enumerate(explanations):
                w.writerow([
                    exp.text,
                    exp.hard_label_human,
                    exp.hard_label_pred,
                    f"{comp_scores[idx]:.6f}",
                    f"{suff_scores[idx]:.6f}",
                ])
        print(f"  {len(explanations)} rows saved.")

    # ── Save summary JSON ─────────────────────────────────────────────────
    if output_json:
        summary = {
            "n_examples": len(explanations),
            "n_runs": n_runs,
            "soft_comprehensiveness": {
                "mean": float(mean_comp),
                "std": float(std_comp),
                "per_run": [float(x) for x in all_comp_runs],
            },
            "soft_sufficiency": {
                "mean": float(mean_suff),
                "std": float(std_suff),
                "per_run": [float(x) for x in all_suff_runs],
            },
        }
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved → {output_json}")

    return {
        "soft_comprehensiveness": mean_comp,
        "soft_sufficiency": mean_suff,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute Soft Comprehensiveness & Soft Sufficiency"
    )
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to HuggingFace model dir "
                             "(contains model.safetensors, config.json, tokenizer.json)")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to predictions CSV")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu, cuda:0, etc.)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Number of Bernoulli sampling runs to average")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Save per-example results to CSV")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save summary to JSON")
    args = parser.parse_args()

    run_evaluation(
        model_dir=args.model_dir,
        csv_path=args.csv_path,
        device=args.device,
        batch_size=args.batch_size,
        max_len=args.max_len,
        n_runs=args.n_runs,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
