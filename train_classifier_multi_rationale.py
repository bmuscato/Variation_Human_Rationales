

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import time
import numpy as np
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils import (
    set_seed,
    calculate_metrics,
    save_checkpoint,
    load_checkpoint,
    format_time,
    print_metrics,
    compute_supervised_attention_loss,
    extract_attention_weights,
)



class KLRegular(nn.Module):
    """KL(P_targ || Q_pred) — standard, target drives the divergence."""
    def __init__(self):
        super().__init__()

    def forward(self, Q_pred: torch.Tensor, P_targ: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sum(P_targ * torch.log2(P_targ / Q_pred), dim=1))


class KLInverse(nn.Module):
    """KL(Q_pred || P_targ) — inverse, prediction drives the divergence."""
    def __init__(self):
        super().__init__()

    def forward(self, Q_pred: torch.Tensor, P_targ: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sum(Q_pred * torch.log2(Q_pred / P_targ), dim=1))


_kl_regular = KLRegular()
_kl_inverse = KLInverse()


#label loss
def compute_label_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    use_soft_labels: bool,
    loss_fn: str,
) -> torch.Tensor:
    if not use_soft_labels:
        if loss_fn != 'ce':
            raise ValueError(f"Hard labels require loss_fn='ce', got '{loss_fn}'")
        return F.cross_entropy(logits, labels)

    probs   = F.softmax(logits, dim=-1).clamp(min=1e-10)
    targets = labels.clamp(min=1e-10)

    if loss_fn == 'kl':
        return _kl_regular(probs, targets)
    elif loss_fn == 'kl_inverse':
        return _kl_inverse(probs, targets)
    elif loss_fn == 'soft_loss':
        log_probs = F.log_softmax(logits, dim=-1)
        return -(targets * log_probs).sum(dim=-1).mean()
    elif loss_fn == 'mse':
        return F.mse_loss(probs, targets)
    else:
        raise ValueError(
            f"Unknown soft label loss: '{loss_fn}'. "
            "Choose from: 'kl', 'kl_inverse', 'soft_loss', 'mse'."
        )


#attention extraction

def _get_attention_scores(
    outputs,
    attention_layer: int,
    attention_head,
    method: str = 'cls',
) -> torch.Tensor:
    attn = extract_attention_weights(outputs, layer=attention_layer, head=attention_head)

    if method == 'cls':
        return attn[:, 0, :]
    elif method == 'average':
        return attn.mean(dim=1)
    else:
        raise ValueError(f"Unknown attention method: '{method}'. Use 'cls' or 'average'.")


#rationale loss

def compute_rationale_loss(
    attention_scores: torch.Tensor,
    rationale_masks: torch.Tensor,
    padding_mask: torch.Tensor,
    loss_type: str = 'mse',
    soft_rationales: bool = False,
) -> torch.Tensor:
    if loss_type == 'soft_loss':
        masked_attn = attention_scores * padding_mask
        masked_rat  = rationale_masks  * padding_mask

        attn_sum  = masked_attn.sum(dim=1, keepdim=True).clamp(min=1e-10)
        norm_attn = masked_attn / attn_sum

        if soft_rationales:
            rat_sum  = masked_rat.sum(dim=1, keepdim=True).clamp(min=1e-10)
            norm_rat = masked_rat / rat_sum
        else:
            norm_rat = masked_rat

        log_attn = F.log_softmax(norm_attn, dim=-1)
        return -(norm_rat * log_attn).sum(dim=-1).mean()

    elif loss_type == 'kl_regular':
        masked_attn = (attention_scores * padding_mask).clamp(min=1e-10)
        masked_rat  = (rationale_masks  * padding_mask).clamp(min=1e-10)
        masked_attn = masked_attn / masked_attn.sum(dim=1, keepdim=True)
        masked_rat  = masked_rat  / masked_rat.sum(dim=1, keepdim=True)
        return _kl_regular(masked_attn, masked_rat)

    elif loss_type == 'kl_inverse':
        masked_attn = (attention_scores * padding_mask).clamp(min=1e-10)
        masked_rat  = (rationale_masks  * padding_mask).clamp(min=1e-10)
        masked_attn = masked_attn / masked_attn.sum(dim=1, keepdim=True)
        masked_rat  = masked_rat  / masked_rat.sum(dim=1, keepdim=True)
        return _kl_inverse(masked_attn, masked_rat)

    return compute_supervised_attention_loss(
        attention_weights=attention_scores,
        rationale_masks=rationale_masks,
        padding_mask=padding_mask,
        loss_type=loss_type,
        soft_rationales=soft_rationales,
    )


#dataset

class HateSpeechDataset(Dataset):

    _SOFT_TYPES = {'soft'}
    _HARD_TYPES = {'union', 'random', 'summed'}

    def __init__(self, data: list, rationale_type: str = 'soft'):
        self.data            = data
        self.rationale_type  = rationale_type
        self.rationale_key   = f'rationale_mask_{rationale_type}'
        self.use_soft_labels = rationale_type in self._SOFT_TYPES

        if rationale_type not in self._SOFT_TYPES | self._HARD_TYPES:
            raise ValueError(
                f"rationale_type '{rationale_type}' not recognised. "
                f"Choose from: {sorted(self._SOFT_TYPES | self._HARD_TYPES)}"
            )
        if self.rationale_key not in data[0]:
            available = [k for k in data[0] if k.startswith('rationale_mask_')]
            raise ValueError(f"Key '{self.rationale_key}' not in data. Available: {available}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item   = self.data[idx]
        labels = item['soft_label'] if self.use_soft_labels else item['label']
        return {
            'input_ids':        item['input_ids'],
            'attention_mask':   item['attention_mask'],
            'labels':           labels,
            'hard_label':       item['label'],
            self.rationale_key: item[self.rationale_key],
            'text':             item['text'],
            'post_id':          item['post_id'],
        }


def collate_fn(batch: list) -> dict:
    rationale_keys = [k for k in batch[0] if k.startswith('rationale_mask_')]
    batch_dict = {
        'input_ids':      torch.stack([b['input_ids']      for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels':         torch.stack([b['labels']         for b in batch]),
        'hard_labels':    torch.stack([b['hard_label']     for b in batch]),
    }
    for key in rationale_keys:
        batch_dict[key] = torch.stack([b[key] for b in batch])
    return batch_dict


#training

def train_epoch(
    model, train_loader, optimizer, scheduler, device, epoch,
    use_soft_labels: bool = True,
    label_loss_fn: str = 'kl',
    use_rationale_loss: bool = False,
    rationale_loss_fn: str = 'mse',
    soft_rationales: bool = False,
    attention_method: str = 'cls',
    attention_layer: int = -1,
    attention_head=0,
    log_interval: int = 100,
):
    model.train()
    total_loss = total_label_loss = total_rationale_loss = 0.0
    rationale_computed_count = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(progress_bar):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)
        hard_labels    = batch['hard_labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=use_rationale_loss,
        )
        logits = outputs.logits

        label_loss = compute_label_loss(logits, labels, use_soft_labels, label_loss_fn)
        loss = label_loss

        if use_rationale_loss:
            rat_keys = [k for k in batch if k.startswith('rationale_mask_')]
            if rat_keys:
                rationale_masks = batch[rat_keys[0]].to(device)
                has_rationale   = rationale_masks.sum(dim=1) > 0
                is_offensive    = hard_labels > 0
                compute_mask    = has_rationale & is_offensive

                if compute_mask.any():
                    attention_scores = _get_attention_scores(
                        outputs,
                        attention_layer=attention_layer,
                        attention_head=attention_head,
                        method=attention_method,
                    )
                    rationale_loss = compute_rationale_loss(
                        attention_scores=attention_scores[compute_mask],
                        rationale_masks=rationale_masks[compute_mask],
                        padding_mask=attention_mask[compute_mask].float(),
                        loss_type=rationale_loss_fn,
                        soft_rationales=soft_rationales,
                    )
                    loss = label_loss + config.ATTENTION_ALPHA * rationale_loss  # BERT-HateXplain: supervised attention with alpha weighting
                    total_rationale_loss    += rationale_loss.item()
                    rationale_computed_count += 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss       += loss.item()
        total_label_loss += label_loss.item()

        postfix = {'loss': f'{loss.item():.4f}', 'lbl': f'{label_loss.item():.4f}'}
        if rationale_computed_count > 0:
            postfix['rat'] = f'{total_rationale_loss / rationale_computed_count:.4f}'
        progress_bar.set_postfix(postfix)

        if (step + 1) % log_interval == 0:
            msg = (f"Step {step+1}/{len(train_loader)} | "
                   f"loss={total_loss/(step+1):.4f} | "
                   f"lbl={total_label_loss/(step+1):.4f}")
            if rationale_computed_count > 0:
                msg += f" | rat={total_rationale_loss/rationale_computed_count:.4f}"
            msg += f" | lr={scheduler.get_last_lr()[0]:.2e}"
            print(msg)

    n = len(train_loader)
    return (
        total_loss / n,
        total_label_loss / n,
        total_rationale_loss / rationale_computed_count if rationale_computed_count > 0 else 0.0,
    )


#evaluation

def evaluate_model(
    model, dataloader, device,
    use_soft_labels: bool = True,
    label_loss_fn: str = 'kl',
    use_rationale_loss: bool = False,
    rationale_loss_fn: str = 'mse',
    soft_rationales: bool = False,
    attention_method: str = 'cls',
    attention_layer: int = -1,
    attention_head=0,
    save_predictions: bool = False,
):
    model.eval()
    total_loss = total_label_loss = total_rationale_loss = 0.0
    rationale_computed_count = 0

    all_predictions, all_hard_labels, all_probabilities = [], [], []
    all_attention_scores = [] if save_predictions else None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            hard_labels    = batch['hard_labels'].to(device)

            need_attentions = use_rationale_loss or save_predictions
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=need_attentions,
            )
            logits        = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

            label_loss = compute_label_loss(logits, labels, use_soft_labels, label_loss_fn)
            loss = label_loss

            if use_rationale_loss:
                rat_keys = [k for k in batch if k.startswith('rationale_mask_')]
                if rat_keys:
                    rationale_masks = batch[rat_keys[0]].to(device)
                    has_rationale   = rationale_masks.sum(dim=1) > 0
                    is_offensive    = hard_labels > 0
                    compute_mask    = has_rationale & is_offensive

                    if compute_mask.any():
                        attention_scores = _get_attention_scores(
                            outputs,
                            attention_layer=attention_layer,
                            attention_head=attention_head,
                            method=attention_method,
                        )
                        rationale_loss = compute_rationale_loss(
                            attention_scores=attention_scores[compute_mask],
                            rationale_masks=rationale_masks[compute_mask],
                            padding_mask=attention_mask[compute_mask].float(),
                            loss_type=rationale_loss_fn,
                            soft_rationales=soft_rationales,
                        )
                        loss = label_loss + config.ATTENTION_ALPHA * rationale_loss  
                        total_rationale_loss    += rationale_loss.item()
                        rationale_computed_count += 1

            if save_predictions and outputs.attentions is not None:
                scores = _get_attention_scores(
                    outputs,
                    attention_layer=attention_layer,
                    attention_head=attention_head,
                    method=attention_method,
                )
                all_attention_scores.extend(scores.cpu().numpy())

            total_loss       += loss.item()
            total_label_loss += label_loss.item()

            preds = torch.argmax(logits, dim=-1)
            all_predictions.extend(preds.cpu().numpy())
            all_hard_labels.extend(hard_labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    n = len(dataloader)
    avg_loss           = total_loss / n
    avg_label_loss     = total_label_loss / n
    avg_rationale_loss = (total_rationale_loss / rationale_computed_count
                          if rationale_computed_count > 0 else 0.0)

    metrics = calculate_metrics(np.array(all_predictions), np.array(all_hard_labels))
    metrics['total_loss']  = avg_loss
    metrics['label_loss']  = avg_label_loss
    if use_rationale_loss:
        metrics['rationale_loss'] = avg_rationale_loss

    if save_predictions:
        predictions_dict = {
            'predictions':    np.array(all_predictions),
            'labels':         np.array(all_hard_labels),
            'probabilities':  np.array(all_probabilities),
            'attention_scores': (np.array(all_attention_scores)
                                 if all_attention_scores else None),
        }
        return metrics, avg_loss, predictions_dict

    return metrics, avg_loss


#predicition saving

def save_predictions(predictions_dict, output_path, dataset,
                     tokenizer=None, rationale_type='soft'):
    results       = []
    rationale_key = f'rationale_mask_{rationale_type}'

    for idx in range(len(predictions_dict['predictions'])):
        item = dataset.data[idx]
        result = {
            'post_id':         item['post_id'],
            'text':            item['text'],
            'predicted_label': int(predictions_dict['predictions'][idx]),
            'true_label':      int(predictions_dict['labels'][idx]),
            'probabilities':   predictions_dict['probabilities'][idx].tolist(),
        }

        if rationale_key in item:
            result['human_rationales'] = item[rationale_key].numpy().tolist()

        if predictions_dict['attention_scores'] is not None:
            attention      = predictions_dict['attention_scores'][idx]
            attention_mask = item['attention_mask'].numpy()
            masked         = attention * attention_mask
            s              = masked.sum()
            normalized     = masked / s if s > 0 else masked
            result['model_rationales'] = normalized.tolist()

            if tokenizer is not None:
                valid_idx    = np.where(attention_mask == 1)[0]
                valid_attn   = normalized[valid_idx]
                valid_tokens = item['input_ids'].numpy()[valid_idx]
                top_k        = min(10, len(valid_attn))
                top_idx      = np.argsort(valid_attn)[-top_k:][::-1]
                result['model_rationale_tokens'] = [
                    {'token': tokenizer.decode([valid_tokens[i]]),
                     'score': float(valid_attn[i])}
                    for i in top_idx
                ]

        results.append(result)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nPredictions saved to {output_path}")
    print(f"  - {len(results)} samples")
    print(f"  - Human rationales: {rationale_type}")
    print(f"  - Model rationales: normalized soft distribution (sum=1)")




#main

def main():
    parser = argparse.ArgumentParser(
        description="Train hate speech classifier - BASELINE"
    )
    parser.add_argument('--epochs',        type=int,   default=config.NUM_EPOCHS)
    parser.add_argument('--batch_size',    type=int,   default=config.BATCH_SIZE)
    parser.add_argument('--learning_rate', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--model_name',    type=str,   default=config.MODEL_NAME)

    parser.add_argument('--rationale_type', type=str, default='soft',
                        choices=['soft', 'union', 'random', 'summed'])

    parser.add_argument('--label_loss', type=str, default=None,
                    choices=['ce', 'kl', 'kl_inverse', 'soft_loss', 'mse'],
                    help=("Defaults to 'kl' for soft, 'ce' for hard. "
                          "Soft labels: 'kl' (KLRegular), 'kl_inverse', 'soft_loss', 'mse'. "
                          "Hard labels: 'ce' only."))

    parser.add_argument('--use_rationale_loss', action='store_true')
    parser.add_argument('--rationale_loss', type=str, default='mse',
                        choices=['mse', 'kl', 'kl_regular', 'kl_inverse', 'soft_loss'])
    parser.add_argument('--soft_rationales', action='store_true')

    parser.add_argument('--attention_method', type=str, default='cls',
                        choices=['cls', 'average'])
    parser.add_argument('--attention_layer', type=int, default=config.ATTENTION_LAYER)
    parser.add_argument('--attention_head',
                        type=lambda x: None if x == 'None' else int(x),
                        default=config.ATTENTION_HEAD,
                        help="Head index, or 'None' to average all heads.")

    parser.add_argument('--resume',           action='store_true')
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--predictions_file', type=str, default='predictions.json')

    parser.add_argument('--push_to_hub',  action='store_true')
    parser.add_argument('--hub_repo_id',  type=str, default=None)
    parser.add_argument('--hub_private',  action='store_true', default=True)
    parser.add_argument('--hub_token',    type=str, default=None)

    args = parser.parse_args()
    set_seed(config.SEED)

    if args.push_to_hub and not args.hub_repo_id:
        parser.error("--push_to_hub requires --hub_repo_id to be set.")


    use_soft_labels = (args.rationale_type == 'soft')

  
    if args.label_loss is None:
        args.label_loss = 'kl' if use_soft_labels else 'ce'

    if use_soft_labels and args.label_loss == 'ce':
        raise ValueError(
            f"rationale_type='soft' requires a soft label loss "
            f"('kl', 'kl_inverse', 'soft_loss', 'mse'), got 'ce'."
    )
    if not use_soft_labels and args.label_loss != 'ce':
        raise ValueError(
            f"rationale_type='{args.rationale_type}' uses hard majority labels, "
            f"which requires label_loss='ce', got '{args.label_loss}'."
        )

    print("=" * 80)
    print("BASELINE TRAINING")
    print(f"  rationale_type={args.rationale_type}  |  "
          f"mode={'soft' if use_soft_labels else 'hard'}  |  "
          f"label_loss={args.label_loss}")
    print(f"  rationale_loss={args.use_rationale_loss}  |  "
          f"rat_loss={args.rationale_loss}  |  soft_rat={args.soft_rationales}")
    print(f"  attn_method={args.attention_method}  |  "
          f"layer={args.attention_layer}  |  head={args.attention_head}")
    if args.push_to_hub:
        print(f"  push_to_hub=True  |  repo={args.hub_repo_id}  |  "
              f"private={args.hub_private}")

    print(f"  checkpoint → {config.BEST_MODEL_PATH}  (overwritten on each F1 improvement)")
    print("=" * 80)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_data = torch.load(config.TRAIN_FILE)
    val_data   = torch.load(config.VAL_FILE)
    print(f"Train: {len(train_data)}  |  Val: {len(val_data)}")

    if args.use_rationale_loss:
        key = f'rationale_mask_{args.rationale_type}'
        if key not in train_data[0]:
            raise ValueError(f"No '{key}' found in data.")

    train_dataset = HateSpeechDataset(train_data, args.rationale_type)
    val_dataset   = HateSpeechDataset(val_data,   args.rationale_type)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=config.NUM_LABELS,
        hidden_dropout_prob=config.DROPOUT_RATE,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_FILE)
    model.to(config.DEVICE)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate,
                      weight_decay=config.WEIGHT_DECAY)
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    # Runs AFTER args.label_loss is resolved — loads from best_baseline.pt
    start_epoch, best_f1 = 0, 0.0
    best_val_metrics     = {}

    if args.resume:
        if os.path.exists(config.BEST_MODEL_PATH):
            start_epoch, _, best_f1 = load_checkpoint(
                config.BEST_MODEL_PATH, model, optimizer
            )
            print(f"Resumed from {config.BEST_MODEL_PATH} | "
                  f"epoch {start_epoch} | best F1={best_f1:.4f}")
        else:
            print(f"No checkpoint found at {config.BEST_MODEL_PATH}, starting from scratch.")

    # ── Training loop ────────────────────────────────────────────────────────
    training_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_lbl, train_rat = train_epoch(
            model, train_loader, optimizer, scheduler, config.DEVICE, epoch + 1,
            use_soft_labels=use_soft_labels,
            label_loss_fn=args.label_loss,
            use_rationale_loss=args.use_rationale_loss,
            rationale_loss_fn=args.rationale_loss,
            soft_rationales=args.soft_rationales,
            attention_method=args.attention_method,
            attention_layer=args.attention_layer,
            attention_head=args.attention_head,
            log_interval=config.LOG_INTERVAL,
        )

        print("\nEvaluating on validation set...")
        val_metrics, val_loss = evaluate_model(
            model, val_loader, config.DEVICE,
            use_soft_labels=use_soft_labels,
            label_loss_fn=args.label_loss,
            use_rationale_loss=args.use_rationale_loss,
            rationale_loss_fn=args.rationale_loss,
            soft_rationales=args.soft_rationales,
            attention_method=args.attention_method,
            attention_layer=args.attention_layer,
            attention_head=args.attention_head,
        )

        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"train={train_loss:.4f} (lbl={train_lbl:.4f}, rat={train_rat:.4f}) | "
              f"val={val_loss:.4f} | time={format_time(time.time()-t0)}")
        print_metrics(val_metrics, prefix="Validation")
        print('=' * 80)

        # ── Save best checkpoint — always to best_baseline.pt (overwrite) ────
        if val_metrics['f1'] > best_f1:
            best_f1          = val_metrics['f1']
            best_val_metrics = val_metrics
            save_checkpoint(model, optimizer, epoch + 1, 0, best_f1,
                            config.BEST_MODEL_PATH)
            print(f"  ✓ Best model saved (F1={best_f1:.4f}) → {config.BEST_MODEL_PATH}")

    print(f"\nDone. Total time: {format_time(time.time() - training_start)}")
    print(f"Best validation F1: {best_f1:.4f}")

    # ── Final eval ───────────────────────────────────────────────────────────
    print("\nFinal evaluation on validation set...")
    if args.save_predictions:
        final_metrics, _, pred_dict = evaluate_model(
            model, val_loader, config.DEVICE,
            use_soft_labels=use_soft_labels,
            label_loss_fn=args.label_loss,
            use_rationale_loss=args.use_rationale_loss,
            rationale_loss_fn=args.rationale_loss,
            soft_rationales=args.soft_rationales,
            attention_method=args.attention_method,
            attention_layer=args.attention_layer,
            attention_head=args.attention_head,
            save_predictions=True,
        )
        print_metrics(final_metrics, prefix="Final Validation")
        save_predictions(pred_dict, args.predictions_file, val_dataset,
                         tokenizer, args.rationale_type)
    else:
        final_metrics, _ = evaluate_model(
            model, val_loader, config.DEVICE,
            use_soft_labels=use_soft_labels,
            label_loss_fn=args.label_loss,
            use_rationale_loss=args.use_rationale_loss,
            rationale_loss_fn=args.rationale_loss,
            soft_rationales=args.soft_rationales,
            attention_method=args.attention_method,
            attention_layer=args.attention_layer,
            attention_head=args.attention_head,
        )
        print_metrics(final_metrics, prefix="Final Validation")


if __name__ == "__main__":
    main()
