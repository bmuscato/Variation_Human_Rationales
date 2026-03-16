

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from typing import Dict, List, Tuple, Optional
import json
import random
from transformers import PreTrainedTokenizer
from collections import Counter


def set_seed(seed: int):
   
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
   
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='macro'),
        'precision': precision_score(labels, predictions, average='macro', zero_division=0),
        'recall': recall_score(labels, predictions, average='macro', zero_division=0)
    }


    cm = confusion_matrix(labels, predictions)
    if len(cm) == 2:
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
    else:
        metrics['confusion_matrix'] = cm.tolist()

    return metrics




def soft_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
 
    return float(np.mean(np.sum(np.minimum(y_true, y_pred), axis=1)))


def soft_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    n_classes = y_true.shape[1]
    f1s = []
    for c in range(n_classes):
        tp = np.sum(np.minimum(y_true[:, c], y_pred[:, c]))
        fp = np.sum(np.maximum(y_pred[:, c] - y_true[:, c], 0))
        fn = np.sum(np.maximum(y_true[:, c] - y_pred[:, c], 0))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1s.append(2 * prec * rec / (prec + rec + 1e-12))
    return float(np.mean(f1s))


def soft_jsd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
 
    eps = 1e-12
    p = np.clip(y_true, eps, 1.0)
    q = np.clip(y_pred, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m), axis=1)
    kl_qm = np.sum(q * np.log(q / m), axis=1)
    return float(np.mean(0.5 * kl_pm + 0.5 * kl_qm))




def load_hatexplain_data(filepath: str) -> Dict[str, Dict]:
    """Load HateXplain dataset from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_majority_label(annotators: List[Dict], label_mapping: Dict[str, int]) -> Tuple[int, float]:
   
    labels = [ann['label'] for ann in annotators]
    label_counts = Counter(labels)

    # Get the most common label
    majority_label_str, count = label_counts.most_common(1)[0]
    majority_label = label_mapping[majority_label_str]

    # Calculate agreement score
    agreement_score = count / len(annotators)

    return majority_label, agreement_score


def merge_rationales(rationales: List[List[int]]) -> List[int]:
    
    if not rationales:
        return []

   
    rationales_array = np.array(rationales)

    # Use majority voting: a token is a rationale if >50% of annotators marked it
    merged = (rationales_array.mean(axis=0) >= 0.5).astype(int)

    return merged.tolist()


def create_data_splits(data: Dict[str, Dict], train_ratio: float, val_ratio: float,
                      test_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
   
    data_list = []
    for post_id, item in data.items():
        item['post_id'] = post_id
        data_list.append(item)


    random.seed(seed)
    random.shuffle(data_list)


    n = len(data_list)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)


    train_data = data_list[:train_end]
    val_data = data_list[train_end:val_end]
    test_data = data_list[val_end:]

    return train_data, val_data, test_data


def create_rationale_mask_from_tokens(binary_rationales: List[int], tokenizer: PreTrainedTokenizer,
                                    post_tokens: List[str], max_length: int) -> torch.Tensor:
   

    text = ' '.join(post_tokens)

    # Tokenize with BERT tokenizer
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors='pt'
    )

    # Initialize mask with zeros
    mask = torch.zeros(max_length)

    # Map original tokens to character positions
    char_pos = 0
    token_char_spans = []
    for token in post_tokens:
        start = text.find(token, char_pos)
        if start == -1:
            start = char_pos
        end = start + len(token)
        token_char_spans.append((start, end))
        char_pos = end + 1  # +1 for space

    # Get character offsets for BERT tokens
    offset_mapping = encoding['offset_mapping'][0]

    # Map rationales to BERT tokens
    for i, is_rationale in enumerate(binary_rationales):
        if is_rationale == 1 and i < len(token_char_spans):
            token_start, token_end = token_char_spans[i]

            # Find all BERT tokens that overlap with this original token
            for j, (bert_start, bert_end) in enumerate(offset_mapping):
                # Skip special tokens
                if bert_start == 0 and bert_end == 0:
                    continue

                # Check if BERT token overlaps with original token
                if bert_start < token_end and bert_end > token_start:
                    mask[j] = 1.0

    return mask


def save_checkpoint(model, optimizer, epoch: int, step: int, best_metric: float, filepath: str):

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'best_metric': best_metric
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, model, optimizer=None):

    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint.get('epoch', 0), checkpoint.get('step', 0), checkpoint.get('best_metric', 0.0)


def format_time(seconds: float) -> str:

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def print_metrics(metrics: Dict[str, float], prefix: str = ""):

    if prefix:
        print(f"\n{prefix} Metrics:")
    else:
        print("\nMetrics:")

    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")

    if 'confusion_matrix' in metrics:
        print(f"\nConfusion Matrix:")
        for i, row in enumerate(metrics['confusion_matrix']):
            print(f"  {row}")


def compute_supervised_attention_loss(attention_weights: torch.Tensor,
                                    rationale_masks: torch.Tensor,
                                    padding_mask: torch.Tensor) -> torch.Tensor:
    """Compute loss between attention weights and rationale masks."""
    # Handle different attention weight formats
    if attention_weights.dim() == 4:
        # Average over heads and take attention to [CLS] token
        attention_weights = attention_weights.mean(dim=1)[:, 0, :]
    elif attention_weights.dim() == 3:
        # Take attention to [CLS] token
        attention_weights = attention_weights[:, 0, :]

    # Apply padding mask
    attention_weights = attention_weights * padding_mask
    rationale_masks = rationale_masks * padding_mask

    # Normalize attention weights
    attention_sum = attention_weights.sum(dim=1, keepdim=True) + 1e-10
    attention_weights_norm = attention_weights / attention_sum

 
    loss = ((attention_weights_norm - rationale_masks) ** 2) * padding_mask

    # Average over non-padded tokens
    num_tokens = padding_mask.sum(dim=1, keepdim=True) + 1e-10
    mse_loss = (loss.sum(dim=1) / num_tokens.squeeze()).mean()

    return loss


def extract_attention_weights(outputs, layer: int = -1, head: Optional[int] = None) -> torch.Tensor:
    """Extract attention weights from model outputs."""
    if not hasattr(outputs, 'attentions') or outputs.attentions is None:
        raise ValueError("Model outputs do not contain attention weights")

    attention = outputs.attentions[layer]

    if head is not None:
        attention = attention[:, head, :, :]
    else:
        attention = attention.mean(dim=1)

    return attention


