"""
Usage:
    python evaluate_classifier.py --rationale_type soft --attention_layer 8 --attention_head 7
    python evaluate_classifier.py --rationale_type union --attention_method average
"""

import os
import sys
import json
import csv
import argparse
import warnings
from itertools import groupby

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

warnings.filterwarnings('ignore', message='networkx backend defined more than once')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils import (
    set_seed,
    calculate_metrics,
    print_metrics,
    extract_attention_weights,
    soft_accuracy,
    soft_macro_f1,
    soft_jsd,
)

from train_classifier_multi_rationale import HateSpeechDataset, collate_fn, _get_attention_scores


#eraser-based

def find_consecutive_spans(binary_mask):
    spans   = []
    indices = np.where(binary_mask == 1)[0]
    if len(indices) == 0:
        return spans
    for k, g in groupby(enumerate(indices), lambda x: x[1] - x[0]):
        group = list(g)
        spans.append((group[0][1], group[-1][1] + 1))
    return spans


def compute_span_iou(span1, span2):
    start1, end1 = span1
    start2, end2 = span2
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = (end1 - start1) + (end2 - start2) - intersection
    return intersection / union if union > 0 else 0.0


def compute_span_iou_f1(model_spans, human_spans, iou_threshold=0.5):
    if not model_spans and not human_spans:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    if not model_spans or not human_spans:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    matched_pred = sum(
        1 for m in model_spans
        if any(compute_span_iou(m, h) >= iou_threshold for h in human_spans)
    )
    matched_gt = sum(
        1 for h in human_spans
        if any(compute_span_iou(h, m) >= iou_threshold for m in model_spans)
    )
    precision = matched_pred / len(model_spans)
    recall    = matched_gt  / len(human_spans)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1}


#inference

def get_prediction_details(
    model, dataloader, device,
    attention_layer: int,
    attention_head,
    attention_method: str,
    rationale_type: str,
    return_texts: bool = False,
) -> dict:
    
    model.eval()

    all_predictions     = []
    all_hard_labels     = []
    all_probabilities   = []
    all_attention       = []
    all_texts           = []
    all_soft_labels     = []
    all_rationale_masks = []
    all_target_cats     = []

    rationale_key = f'rationale_mask_{rationale_type}'

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            hard_labels    = batch['hard_labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            logits        = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions   = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_hard_labels.extend(hard_labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            scores = _get_attention_scores(
                outputs,
                attention_layer=attention_layer,
                attention_head=attention_head,
                method=attention_method,
            )
            all_attention.extend(scores.cpu().numpy())

            if return_texts:
                start = batch_idx * dataloader.batch_size
                end   = min(start + dataloader.batch_size, len(dataloader.dataset))
                for idx in range(start, end):
                    item = dataloader.dataset.data[idx]
                    all_texts.append(item['text'])
                    all_target_cats.append(item.get('final_target_category', None))

                    if 'soft_label' in item:
                        all_soft_labels.append(item['soft_label'].cpu().numpy())
                    else:
                        hard = item['label'].item() if hasattr(item['label'], 'item') else item['label']
                        one_hot        = np.zeros(config.NUM_LABELS)
                        one_hot[hard]  = 1.0
                        all_soft_labels.append(one_hot)

                    if rationale_key in item:
                        all_rationale_masks.append(item[rationale_key].cpu().numpy())

    results = {
        'predictions':   np.array(all_predictions),
        'labels':        np.array(all_hard_labels),
        'probabilities': np.array(all_probabilities),
        'attention':     np.array(all_attention),
    }
    if return_texts:
        results['texts']       = all_texts
        results['target_cats'] = all_target_cats
        results['soft_labels'] = np.array(all_soft_labels)
        if all_rationale_masks:
            results['rationale_masks'] = np.array(all_rationale_masks)

    return results




def compute_soft_metrics(results: dict) -> dict:
    print(f"\n{'='*80}")
    print("Soft Metrics")
    print('='*80)

    y_true = results['soft_labels']
    y_pred = results['probabilities']

    s_acc = soft_accuracy(y_true, y_pred)
    s_f1  = soft_macro_f1(y_true, y_pred)
    jsd   = soft_jsd(y_true, y_pred)

    y_true_t = torch.tensor(y_true, dtype=torch.float32)
    y_pred_t = torch.tensor(y_pred, dtype=torch.float32)
    kl = F.kl_div(torch.log(y_pred_t + 1e-12), y_true_t, reduction='batchmean').item()

    metrics = {
        'soft_accuracy': float(s_acc),
        'soft_macro_f1': float(s_f1),
        'jsd':           float(jsd),
        'kl_divergence': float(kl),
    }

    print(f"Soft Accuracy:  {s_acc:.4f}")
    print(f"Soft Macro F1:  {s_f1:.4f}")
    print(f"JSD (↓ better): {jsd:.4f}")
    print(f"KL Divergence:  {kl:.4f}")

    return metrics




def save_predictions_to_csv(results: dict, output_file: str):
    print(f"\nSaving predictions → {output_file}")
    n = len(results['texts'])

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'text', 'soft_labels_human', 'soft_labels_pred',
            'hard_label_human', 'hard_label_pred',
            'model_rationales', 'human_rationales',
        ])
        for i in range(n):
            sl_human = ','.join(f"{x:.3f}" for x in results['soft_labels'][i])
            sl_pred  = ','.join(f"{x:.3f}" for x in results['probabilities'][i])

            attn = results['attention'][i]
            s    = attn.sum()
            norm_attn = attn / s if s > 0 else attn
            mr_str = ','.join(f"{x:.6f}" for x in norm_attn)

            hr_str = ''
            if 'rationale_masks' in results and i < len(results['rationale_masks']):
                hr_str = ','.join(f"{x:.6f}" for x in results['rationale_masks'][i])

            writer.writerow([
                results['texts'][i],
                sl_human, sl_pred,
                int(results['labels'][i]),
                int(results['predictions'][i]),
                mr_str, hr_str,
            ])

    print(f"  Saved {n} rows to {output_file}")


#error analysis

def save_error_cases(predictions, labels, probabilities, texts,
                     output_file="error_cases.json"):
    label_names = ['Normal', 'Offensive', 'Hate speech']
    errors = []
    for idx in range(len(predictions)):
        pred, true = int(predictions[idx]), int(labels[idx])
        if pred != true:
            prob = [float(x) for x in probabilities[idx]]
            errors.append({
                'text':               texts[idx],
                'true_label':         label_names[true],
                'predicted_label':    label_names[pred],
                'true_label_id':      true,
                'predicted_label_id': pred,
                'confidence':         float(max(prob)),
                'predicted_probs':    {'normal': prob[0], 'offensive': prob[1],
                                       'hate_speech': prob[2]},
                'error_type':         f"{label_names[true]}_as_{label_names[pred]}",
            })
    with open(output_file, 'w') as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(errors)} error cases → {output_file}")


def analyze_errors(predictions, labels, probabilities, texts=None, num_examples=20):
    label_names = ['Normal', 'Offensive', 'Hate speech']
    error_idx = np.where(predictions != labels)[0]
    print(f"\nError Analysis: {len(error_idx)}/{len(predictions)} "
          f"({len(error_idx)/len(predictions)*100:.2f}%)")

    err_mat = np.zeros((3, 3), dtype=int)
    for i in range(len(predictions)):
        if predictions[i] != labels[i]:
            err_mat[labels[i], predictions[i]] += 1

    print("\nError confusion (true → predicted):")
    print(f"{'':>12} {'Normal':>8} {'Offensive':>10} {'Hate speech':>12}")
    for i in range(3):
        print(f"{label_names[i]:>12} {err_mat[i,0]:>8} {err_mat[i,1]:>10} {err_mat[i,2]:>12}")

    max_probs = np.max(probabilities, axis=1)
    print(f"\nHigh confidence errors (>80%): {np.sum(max_probs[error_idx] > 0.8)}")

    if texts is not None and len(error_idx) > 0:
        print(f"\n{'='*80}\nExample Misclassifications (up to {num_examples}):\n{'='*80}")
        sorted_err = error_idx[np.argsort(-max_probs[error_idx])]
        for k, idx in enumerate(sorted_err[:num_examples]):
            print(f"\n[{k+1}] \"{texts[idx][:200]}{'...' if len(texts[idx])>200 else ''}\"")
            print(f"  True: {label_names[labels[idx]]}  |  "
                  f"Predicted: {label_names[predictions[idx]]}  |  "
                  f"Confidence: {max_probs[idx]:.2%}")
            print(f"  Probs: N={probabilities[idx][0]:.2%}  "
                  f"O={probabilities[idx][1]:.2%}  H={probabilities[idx][2]:.2%}")




def compute_explainability_metrics(
    model, test_loader, device, tokenizer,
    attention_threshold: float = 0.5,
    rationale_type: str = 'soft',
    attention_method: str = 'cls',
    attention_layer: int = 8,
    attention_head=None,
) -> dict:
    """
    Plausibility + faithfulness metrics.
    Soft rationales binarized (> 0 → 1) for IOU/token/span/AUPRC.
    Uses _get_attention_scores — identical to train.py.
    """
    print("\n" + "="*80)
    print("Explainability Metrics")
    print(f"  rationale_type={rationale_type}  |  "
          f"method={attention_method}  |  layer={attention_layer}")
    print("="*80)

    model.eval()
    is_soft = (rationale_type == 'soft')

    iou_match_scores = []
    token_precisions, token_recalls, token_f1s = [], [], []
    span_prec, span_rec, span_f1s = [], [], []
    comprehensiveness_scores, sufficiency_scores = [], []
    all_attn_scores, all_human_rat = [], []
    n_examples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Explainability"):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            hard_labels    = batch['hard_labels'].to(device)

            rat_keys = [k for k in batch if k.startswith('rationale_mask_')]
            if not rat_keys:
                continue
            rationale_masks = batch[rat_keys[0]].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            batch_attn = _get_attention_scores(
                outputs,
                attention_layer=attention_layer,
                attention_head=attention_head,
                method=attention_method,
            )
            probs = torch.softmax(outputs.logits, dim=-1)

            for i in range(len(hard_labels)):
                label_i = hard_labels[i].long()
                if rationale_masks[i].sum() == 0:
                    continue

                n_examples    += 1
                per_token_attn = batch_attn[i].cpu().numpy()
                human_rat      = rationale_masks[i].cpu().numpy()
                valid_mask     = attention_mask[i].cpu().numpy()
                valid_idx      = np.where(valid_mask == 1)[0]

                if len(valid_idx) == 0:
                    continue

                valid_attn      = per_token_attn[valid_idx]
                valid_attn_norm = valid_attn / (valid_attn.sum() + 1e-10)

                human_rat_hard = (human_rat > 0).astype(int) if is_soft else human_rat.astype(int)

                threshold  = valid_attn.mean() + attention_threshold * valid_attn.std()
                model_rat  = np.zeros_like(human_rat_hard)
                model_rat[valid_idx] = (valid_attn > threshold).astype(int)
                model_rat[0] = 0
                sep_idx = valid_idx[-1]
                if sep_idx < len(model_rat):
                    model_rat[sep_idx] = 0

                human_rat_clean    = human_rat_hard.copy()
                human_rat_clean[0] = 0
                if sep_idx < len(human_rat_clean):
                    human_rat_clean[sep_idx] = 0

                orig_pred = probs[i, label_i].item()

                # ── Token-set metrics ────────────────────────────────────────
                M = set(np.where(model_rat == 1)[0])
                H = set(np.where(human_rat_clean == 1)[0])
                if M and H:
                    inter = M & H
                    union = M | H
                    iou   = len(inter) / len(union) if union else 0
                    iou_match_scores.append(1 if iou >= 0.5 else 0)
                    prec = len(inter) / len(M) if M else 0
                    rec  = len(inter) / len(H) if H else 0
                    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0
                    token_precisions.append(prec)
                    token_recalls.append(rec)
                    token_f1s.append(f1)

                # ── Span IOU F1 ──────────────────────────────────────────────
                sm = compute_span_iou_f1(
                    find_consecutive_spans(model_rat),
                    find_consecutive_spans(human_rat_clean),
                )
                span_prec.append(sm['precision'])
                span_rec.append(sm['recall'])
                span_f1s.append(sm['f1'])

                # ── AUPRC ────────────────────────────────────────────────────
                for vi in valid_idx[1:-1]:
                    all_attn_scores.append(
                        valid_attn_norm[np.where(valid_idx == vi)[0][0]]
                    )
                    all_human_rat.append(int(human_rat_clean[vi] > 0))

                # ── Comprehensiveness ────────────────────────────────────────
                masked_inp = input_ids[i].clone()
                for mi in M:
                    if mi < len(masked_inp):
                        masked_inp[mi] = tokenizer.mask_token_id
                comp_out  = model(masked_inp.unsqueeze(0), attention_mask[i].unsqueeze(0))
                comp_pred = torch.softmax(comp_out.logits, dim=-1)[0, label_i].item()
                comprehensiveness_scores.append(orig_pred - comp_pred)

                # ── Sufficiency ──────────────────────────────────────────────
                rat_inp  = torch.full_like(input_ids[i], tokenizer.pad_token_id)
                rat_attn = torch.zeros_like(attention_mask[i])
                rat_inp[0]  = tokenizer.cls_token_id
                rat_attn[0] = 1
                for mi in M:
                    if mi < len(input_ids[i]):
                        rat_inp[mi]  = input_ids[i][mi]
                        rat_attn[mi] = 1
                last = max(M) + 1 if M else 1
                if last < len(rat_inp):
                    rat_inp[last]  = tokenizer.sep_token_id
                    rat_attn[last] = 1
                suf_out  = model(rat_inp.unsqueeze(0), rat_attn.unsqueeze(0))
                suf_pred = torch.softmax(suf_out.logits, dim=-1)[0, label_i].item()
                sufficiency_scores.append(orig_pred - suf_pred)

    if n_examples == 0:
        print("No examples with rationales found.")
        return None

    auprc = 0.0
    if all_attn_scores and all_human_rat:
        try:
            auprc = average_precision_score(
                np.array(all_human_rat), np.array(all_attn_scores)
            )
        except Exception:
            auprc = 0.0

    results = {
        'plausibility': {
            'iou_f1_span':        np.mean(span_f1s)         if span_f1s        else 0.0,
            'iou_precision_span': np.mean(span_prec)        if span_prec       else 0.0,
            'iou_recall_span':    np.mean(span_rec)         if span_rec        else 0.0,
            'iou_match_rate':     np.mean(iou_match_scores) if iou_match_scores else 0.0,
            'token_precision':    np.mean(token_precisions) if token_precisions else 0.0,
            'token_recall':       np.mean(token_recalls)    if token_recalls   else 0.0,
            'token_f1':           np.mean(token_f1s)        if token_f1s       else 0.0,
            'auprc':              float(auprc),
        },
        'faithfulness': {
            'comprehensiveness': np.mean(comprehensiveness_scores) if comprehensiveness_scores else 0.0,
            'sufficiency':       np.mean(sufficiency_scores)       if sufficiency_scores       else 0.0,
        },
        'num_examples': n_examples,
    }

    p = results['plausibility']
    f = results['faithfulness']
    print(f"\nExamples evaluated: {n_examples}")
    print(f"Span IOU F1:       {p['iou_f1_span']:.3f}")
    print(f"Span IOU Prec:     {p['iou_precision_span']:.3f}")
    print(f"Span IOU Rec:      {p['iou_recall_span']:.3f}")
    print(f"Token-set Match:   {p['iou_match_rate']:.3f}")
    print(f"Token Precision:   {p['token_precision']:.3f}")
    print(f"Token Recall:      {p['token_recall']:.3f}")
    print(f"Token F1:          {p['token_f1']:.3f}")
    print(f"AUPRC:             {p['auprc']:.3f}")
    print(f"Comprehensiveness: {f['comprehensiveness']:.3f}")
    print(f"Sufficiency:       {f['sufficiency']:.3f}")

    return results


def evaluate_model_comprehensive(
    model, test_loader, device,
    attention_layer, attention_head, attention_method,
    rationale_type,
    use_soft_labels: bool,
    show_examples: bool = True,
) -> tuple:
    results = get_prediction_details(
        model, test_loader, device,
        attention_layer=attention_layer,
        attention_head=attention_head,
        attention_method=attention_method,
        rationale_type=rationale_type,
        return_texts=True,
    )

    predictions   = results['predictions']
    labels        = results['labels']
    probabilities = results['probabilities']

    metrics = calculate_metrics(predictions, labels)
    try:
        labels_bin   = label_binarize(labels, classes=[0, 1, 2])
        auroc_scores = [
            roc_auc_score(labels_bin[:, i], probabilities[:, i])
            for i in range(3) if len(np.unique(labels_bin[:, i])) > 1
        ]
        metrics['auroc'] = float(np.mean(auroc_scores)) if auroc_scores else 0.5
    except Exception:
        metrics['auroc'] = 0.5

    print("\n" + "="*80)
    print("CLASSIFICATION METRICS")
    print("="*80)
    print_metrics(metrics)
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print("\n" + classification_report(
        labels, predictions,
        labels=[0, 1, 2],
        target_names=['Normal', 'Offensive', 'Hate speech'],
        digits=4,
    ))

    if show_examples:
        analyze_errors(predictions, labels, probabilities, results.get('texts'))

    return metrics, results



def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hate speech classifier - SUPERVISED ATTENTION"
    )
    parser.add_argument('--model_path',  type=str, default=config.BEST_MODEL_PATH)
    parser.add_argument('--batch_size',  type=int, default=config.BATCH_SIZE)

    # rationale_type → single source of truth for label mode
    parser.add_argument('--rationale_type', type=str, default='soft',
                        choices=['soft', 'union', 'random', 'summed'])

    # Attention config — mirrors train.py
    parser.add_argument('--attention_method', type=str, default='cls',
                        choices=['cls', 'average'])
    parser.add_argument('--attention_layer', type=int, default=config.ATTENTION_LAYER)
    parser.add_argument('--attention_head',
                        type=lambda x: None if x == 'None' else int(x),
                        default=config.ATTENTION_HEAD,
                        help="Head index, or 'None' to average all heads.")

    # Explainability
    parser.add_argument('--attention_threshold', type=float, default=0.5)
    parser.add_argument('--skip_explainability', action='store_true')

    # Bias
    parser.add_argument('--skip_bias_metrics', action='store_true')

    # Output
    parser.add_argument('--no_examples',     action='store_true')
    parser.add_argument('--predictions_csv', type=str, default='predictions.csv')
    parser.add_argument('--save_results',    type=str, default=None)

    args = parser.parse_args()
    set_seed(config.SEED)

    # Derived — same logic as train.py and evaluate_classifier_baseline.py
    use_soft_labels = (args.rationale_type == 'soft')

    print("=" * 80)
    print("SUPERVISED ATTENTION EVALUATION")
    print(f"  rationale_type={args.rationale_type}  |  "
          f"mode={'soft' if use_soft_labels else 'hard'}")
    print(f"  attn_method={args.attention_method}  |  "
          f"layer={args.attention_layer}  |  head={args.attention_head}")
    print("=" * 80)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\nLoading test data...")
    test_data = torch.load(config.TEST_FILE)
    print(f"Test samples: {len(test_data)}")

    has_target_cats = any(
        item.get('final_target_category') is not None for item in test_data[:10]
    )

    test_dataset = HateSpeechDataset(test_data, rationale_type=args.rationale_type)
    test_loader  = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    print(f"\nLoading model: {config.MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=config.NUM_LABELS
    )
    if args.model_path.endswith('.pt'):
        state_dict = torch.load(args.model_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        key = 'bert.embeddings.word_embeddings.weight'
        if key in state_dict:
            saved = state_dict[key].shape[0]
            curr  = model.bert.embeddings.word_embeddings.weight.shape[0]
            if saved != curr:
                raise ValueError(f"Vocab mismatch: saved={saved}, model={curr}.")
        model.load_state_dict(state_dict)

    model.to(config.DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_FILE)

    # ── Step 1 : Classification ───────────────────────────────────────────────
    metrics, results = evaluate_model_comprehensive(
        model, test_loader, config.DEVICE,
        attention_layer=args.attention_layer,
        attention_head=args.attention_head,
        attention_method=args.attention_method,
        rationale_type=args.rationale_type,
        use_soft_labels=use_soft_labels,
        show_examples=not args.no_examples,
    )

    if 'texts' in results:
        save_error_cases(
            results['predictions'], results['labels'],
            results['probabilities'], results['texts'],
        )

    # ── Step 2 : Save CSV ─────────────────────────────────────────────────────
    if 'texts' in results and 'soft_labels' in results:
        save_predictions_to_csv(results, args.predictions_csv)

    # ── Step 3 : Soft metrics (soft mode only) ────────────────────────────────
    soft_metrics = None
    if use_soft_labels and 'soft_labels' in results:
        soft_metrics = compute_soft_metrics(results)

    # ── Step 4 : Explainability ───────────────────────────────────────────────
    explainability_metrics = None
    rat_key  = f'rationale_mask_{args.rationale_type}'
    has_rats = any(
        rat_key in item and item[rat_key].sum() > 0
        for item in test_data[:10]
    )

    if not args.skip_explainability:
        if has_rats:
            explainability_metrics = compute_explainability_metrics(
                model, test_loader, config.DEVICE, tokenizer,
                attention_threshold=args.attention_threshold,
                rationale_type=args.rationale_type,
                attention_method=args.attention_method,
                attention_layer=args.attention_layer,
                attention_head=args.attention_head,
            )
        else:
            print(f"\nSkipping explainability: no '{rat_key}' data found.")

    # ── Step 5 : Bias ─────────────────────────────────────────────────────────
    bias_metrics = None
    if not args.skip_bias_metrics:
        if has_target_cats:
            bias_metrics = compute_bias_metrics(model, test_loader, config.DEVICE)
        else:
            print("\nSkipping bias metrics: final_target_category not found.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Mode: {'soft' if use_soft_labels else 'hard'}  |  "
          f"Rationale: {args.rationale_type}")
    print(f"\nHard Metrics:")
    print(f"  F1={metrics['f1']:.4f}  |  Acc={metrics['accuracy']:.4f}  |  "
          f"AUROC={metrics['auroc']:.4f}")

    if soft_metrics:
        print(f"\nSoft Metrics:")
        print(f"  Soft Acc={soft_metrics['soft_accuracy']:.4f}  |  "
              f"Soft F1={soft_metrics['soft_macro_f1']:.4f}  |  "
              f"JSD={soft_metrics['jsd']:.4f}  |  "
              f"KL={soft_metrics['kl_divergence']:.4f}")

    if explainability_metrics:
        p = explainability_metrics['plausibility']
        f = explainability_metrics['faithfulness']
        print(f"\nExplainability ({explainability_metrics['num_examples']} examples):")
        print(f"  Plausibility: Span IOU F1={p['iou_f1_span']:.3f}  "
              f"Token F1={p['token_f1']:.3f}  AUPRC={p['auprc']:.3f}")
        print(f"  Faithfulness: Comp={f['comprehensiveness']:.3f}  "
              f"Suff={f['sufficiency']:.3f}")

    if bias_metrics:
        g = bias_metrics['gmb_metrics']
        print(f"\nBias: Overall AUC={bias_metrics['overall_auc']:.4f}  |  "
              f"GMB Sub={g['gmb_subgroup_auc']:.4f}  "
              f"BPSN={g['gmb_bpsn_auc']:.4f}  BNSP={g['gmb_bnsp_auc']:.4f}")

    print("="*80)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if args.save_results:
        save_data = {
            'mode':           'soft' if use_soft_labels else 'hard',
            'rationale_type': args.rationale_type,
            'metrics':        metrics,
            'predictions':    results['predictions'].tolist(),
            'labels':         results['labels'].tolist(),
            'probabilities':  results['probabilities'].tolist(),
        }
        if soft_metrics:
            save_data['soft_metrics'] = soft_metrics
        if explainability_metrics:
            save_data['explainability_metrics'] = explainability_metrics
        if bias_metrics:
            save_data['bias_metrics'] = bias_metrics

        with open(args.save_results, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved → {args.save_results}")


if __name__ == "__main__":
    main()