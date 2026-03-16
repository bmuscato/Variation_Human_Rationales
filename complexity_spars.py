

import numpy as np
import torch
from typing import Optional


# =============================================================================
# Complexity  (↓ better)
# =============================================================================

def compute_complexity(
    attributions: np.ndarray,
    attention_masks: Optional[np.ndarray] = None,
) -> float:
    """
    Compute average complexity (entropy of fractional contributions)
    across all instances.

    For each instance, complexity = -(1/L) * sum_j( f_j * log(f_j) )
    where f_j = |a_j| / sum(|a|) is the fractional contribution of token j,
    and L is the number of valid tokens.

    Lower complexity means the model focuses on fewer tokens (more interpretable).

    Args:
        attributions:   [N, seq_len] raw attention scores (e.g. CLS attention)
        attention_masks: [N, seq_len] 1 for real tokens, 0 for padding.
                         If None, all positions are treated as valid.

    Returns:
        Average complexity score across all instances (float).
    """
    attributions = torch.tensor(attributions, dtype=torch.float32)
    if attention_masks is not None:
        attention_masks = torch.tensor(attention_masks, dtype=torch.float32)
    else:
        attention_masks = torch.ones_like(attributions)

    N, seq_len = attributions.shape
    eps = 1e-8
    scores = []

    for i in range(N):
        mask = attention_masks[i]
        valid = mask.bool()
        n_valid = valid.sum().item()
        if n_valid == 0:
            continue

        # Absolute attributions for valid tokens only
        abs_attr = torch.abs(attributions[i]) * mask
        total = abs_attr.sum() + eps

        # Fractional contribution per token
        frac = abs_attr / total  # [seq_len]

        # Entropy: -sum( f * log(f) ), only over valid positions
        log_frac = torch.log(frac + eps)
        entropy = -torch.sum(frac * log_frac * mask)

        # Normalise by number of valid tokens
        complexity = entropy / n_valid
        scores.append(complexity.item())

    return float(np.mean(scores)) if scores else 0.0


# =============================================================================
# Sparseness  (↑ better)
# =============================================================================

def compute_sparseness(
    attributions: np.ndarray,
    attention_masks: Optional[np.ndarray] = None,
) -> float:
    """
    Compute average sparseness (Gini index) across all instances.

    For each instance, sparseness = 1 - 2 * sum_j((L - j + 0.5) * |a_j|_sorted) / (L * sum(|a|))
    where attributions are sorted by absolute value.

    Higher sparseness means the model concentrates attribution on fewer tokens.

    Args:
        attributions:   [N, seq_len] raw attention scores
        attention_masks: [N, seq_len] 1 for real tokens, 0 for padding.
                         If None, all positions are treated as valid.

    Returns:
        Average sparseness score across all instances (float).
    """
    attributions = torch.tensor(attributions, dtype=torch.float32)
    if attention_masks is not None:
        attention_masks = torch.tensor(attention_masks, dtype=torch.float32)
    else:
        attention_masks = torch.ones_like(attributions)

    N = attributions.shape[0]
    eps = 1e-8
    scores = []

    for i in range(N):
        mask = attention_masks[i]
        valid_idx = mask.bool()
        n_valid = valid_idx.sum().item()
        if n_valid == 0:
            continue

        # Extract valid tokens and sort by absolute value
        abs_attr = torch.abs(attributions[i][valid_idx])
        sorted_attr, _ = torch.sort(abs_attr)  # ascending

        total = sorted_attr.sum() + eps

        # Weight for index j: (n_valid - j + 0.5)
        # j=0 → L+0.5, j=1 → L-0.5, ..., j=L-1 → 1.5
        # Matches original: (n_features - j + 0.5)
        weights = n_valid - torch.arange(n_valid, dtype=torch.float32) + 0.5
        weighted_sum = (weights * sorted_attr).sum()

        sparsity = 1.0 - 2.0 * weighted_sum / (total * n_valid)
        scores.append(sparsity.item())

    return float(np.mean(scores)) if scores else 0.0


#saving 
def compute_complexity_sparseness_from_csv(
    csv_path: str,
    attention_col: str = "model_rationales",
    separator: str = None,
) -> dict:
   
    import csv

    print(f"\nLoading attributions from {csv_path} ...")

    attributions = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get(attention_col, '').strip()
            if not raw:
                continue

            # Auto-detect separator from first non-empty row
            if separator is None:
                if ';' in raw:
                    sep = ';'
                else:
                    sep = ','
            else:
                sep = separator

            scores = [float(x) for x in raw.split(sep)]
            attributions.append(scores)

    if not attributions:
        print(f"  ⚠ No attributions found in column '{attention_col}'")
        return {'complexity': 0.0, 'sparseness': 0.0}

    # Pad to same length
    max_len = max(len(a) for a in attributions)
    padded = [a + [0.0] * (max_len - len(a)) for a in attributions]
    attr_np = np.array(padded)

    # Build attention mask (non-zero = valid)
    mask_np = (attr_np != 0.0).astype(float)
    # But also mark the original non-padded positions as valid even if score is 0
    for i, a in enumerate(attributions):
        mask_np[i, :len(a)] = 1.0

    c = compute_complexity(attr_np, mask_np)
    s = compute_sparseness(attr_np, mask_np)

    print(f"  Complexity (↓ better): {c:.4f}")
    print(f"  Sparseness (↑ better): {s:.4f}")
    print(f"  Instances: {len(attributions)}")

    return {'complexity': c, 'sparseness': s}



def compute_complexity_sparseness(
    attention_scores: np.ndarray,
    attention_masks: np.ndarray,
) -> dict:
    
    c = compute_complexity(attention_scores, attention_masks)
    s = compute_sparseness(attention_scores, attention_masks)
    return {'complexity': c, 'sparseness': s}



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute complexity and sparseness from a predictions CSV"
    )
    parser.add_argument("csv_path", type=str, help="Path to predictions CSV")
    parser.add_argument("--col", type=str, default="model_rationales",
                        help="Column with attention scores")
    args = parser.parse_args()

    results = compute_complexity_sparseness_from_csv(args.csv_path, attention_col=args.col)
    print(f"\nComplexity: {results['complexity']:.4f}")
    print(f"Sparseness: {results['sparseness']:.4f}")
