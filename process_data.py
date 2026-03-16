

import os
import sys
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
from collections import Counter, defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils import (
    set_seed, load_hatexplain_data, create_data_splits,
    get_majority_label, merge_rationales, create_rationale_mask_from_tokens,
    filter_by_agreement
)


def extract_final_target_category(item: dict) -> list:
   
    targets_all = []

    if 'annotators' in item:
        for annotator in item['annotators']:
            if isinstance(annotator, dict) and 'target' in annotator:
                targets = annotator['target']
                if isinstance(targets, list):
                    targets_all.extend(targets)
                elif isinstance(targets, str) and targets != 'None':
                    targets_all.append(targets)

    for i in range(1, 4):
        target_field = f'target{i}'
        if target_field in item:
            targets = item[target_field]
            if isinstance(targets, list):
                targets_all.extend(targets)
            elif isinstance(targets, str) and targets != 'None':
                targets_all.append(targets)

    if not targets_all:
        return None

    community_counts = Counter(targets_all)
    if 'None' in community_counts:
        del community_counts['None']
    if 'Other' in community_counts:
        del community_counts['Other']

    final_communities = [c for c, count in community_counts.items() if count >= 2]
    return final_communities if final_communities else None


def get_soft_label(annotators: list, label_mapping: dict, num_labels: int) -> torch.Tensor:
    
    dist = torch.zeros(num_labels, dtype=torch.float32)
    for ann in annotators:
        label_str = ann.get('label', 'normal')
        if label_str in label_mapping:
            dist[label_mapping[label_str]] += 1.0
    s = dist.sum()
    if s > 0:
        dist = dist / s
    else:
        dist[0] = 1.0
    return dist


def process_single_item(item: dict, tokenizer, max_length: int) -> dict:
    """Process a single HateXplain data item."""
    post_tokens = item.get('post_tokens', [])
    annotators = item.get('annotators', [])
    rationales = item.get('rationales', [])

    # Get majority label and agreement score
    label, agreement = get_majority_label(annotators, config.LABEL_MAPPING)

    # Get soft label distribution
    soft_label = get_soft_label(annotators, config.LABEL_MAPPING, config.NUM_LABELS)

    # Extract final target categories
    final_target_category = extract_final_target_category(item)

    # Reconstruct text from tokens
    text = ' '.join(post_tokens)

    # Tokenize
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )

    # Merge rationales from multiple annotators
    if rationales:
        merged_rationales = merge_rationales(rationales)
    else:
        merged_rationales = [0] * len(post_tokens)

    # Create processed item
    processed_item = {
        'input_ids': encoding['input_ids'].squeeze(0),
        'attention_mask': encoding['attention_mask'].squeeze(0),
        'label': torch.tensor(label, dtype=torch.long),
        'soft_label': soft_label,
        'text': text,
        'post_id': item.get('post_id', ''),
        'agreement_score': agreement,
        'original_tokens': post_tokens,
        'binary_rationales': merged_rationales,
        'final_target_category': final_target_category,
    }

    # Add token type ids if available
    if 'token_type_ids' in encoding:
        processed_item['token_type_ids'] = encoding['token_type_ids'].squeeze(0)

    # Create rationale mask for attention supervision
    # Always create it for offensive/hatespeech examples with rationales,
    # since the baseline uses attention CE (no flag needed).
    if label > 0 and any(merged_rationales):
        rationale_mask = create_rationale_mask_from_tokens(
            merged_rationales, tokenizer, post_tokens, max_length
        )
        processed_item['rationale_mask'] = rationale_mask
    else:
        processed_item['rationale_mask'] = torch.zeros(max_length)

    return processed_item


def process_dataset(data_list: list, tokenizer, max_length: int,
                    desc: str = "Processing") -> list:
    """Process a list of data items."""
    processed_data = []
    errors = 0
    for item in tqdm(data_list, desc=desc):
        try:
            processed_item = process_single_item(item, tokenizer, max_length)
            processed_data.append(processed_item)
        except Exception as e:
            print(f"Error processing item {item.get('post_id', 'unknown')}: {e}")
            errors += 1
    if errors > 0:
        print(f"  {errors} items failed processing.")
    return processed_data


def save_processed_data(data: list, filepath: str):
    """Save processed data to file."""
    torch.save(data, filepath)
    print(f"Saved {len(data)} items to {filepath}")


def analyze_target_communities(data: list) -> dict:
    """Analyze distribution of target communities in the dataset."""
    all_communities = []
    posts_with_targets = 0

    for item in data:
        if item['final_target_category'] is not None:
            posts_with_targets += 1
            all_communities.extend(item['final_target_category'])

    community_counts = Counter(all_communities)

    return {
        'total_posts': len(data),
        'posts_with_targets': posts_with_targets,
        'community_counts': community_counts,
        'unique_communities': len(community_counts)
    }


def main():
    """Main data processing pipeline."""
    parser = argparse.ArgumentParser(description="Process HateXplain dataset")
    parser.add_argument('--data_path', type=str, default=config.DATA_PATH,
                        help='Path to dataset.json file')
    parser.add_argument('--max_length', type=int, default=config.MAX_LENGTH,
                        help='Maximum sequence length')
    parser.add_argument('--model_name', type=str, default=config.MODEL_NAME,
                        help='Pretrained model name for tokenizer')
    args = parser.parse_args()

    # Set random seed
    set_seed(config.SEED)

    print(f"Loading data from {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    # Load raw data
    raw_data = load_hatexplain_data(args.data_path)
    print(f"Loaded {len(raw_data)} items from dataset")

    # Initialize tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Save tokenizer
    tokenizer.save_pretrained(config.TOKENIZER_FILE)
    print(f"Saved tokenizer to {config.TOKENIZER_FILE}")

    # Create data splits
    print("\nCreating train/val/test splits...")
    train_data, val_data, test_data = create_data_splits(
        raw_data,
        config.TRAIN_RATIO,
        config.VAL_RATIO,
        config.TEST_RATIO,
        config.SEED
    )

    # Filter by agreement if needed
    if config.MIN_ANNOTATOR_AGREEMENT > 1:
        print(f"\nFiltering by minimum agreement: {config.MIN_ANNOTATOR_AGREEMENT}")
        train_data = filter_by_agreement(train_data, config.MIN_ANNOTATOR_AGREEMENT)
        val_data = filter_by_agreement(val_data, config.MIN_ANNOTATOR_AGREEMENT)
        test_data = filter_by_agreement(test_data, config.MIN_ANNOTATOR_AGREEMENT)

    print(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Process each split
    print("\nProcessing data splits...")
    train_processed = process_dataset(train_data, tokenizer, args.max_length, "Processing train")
    val_processed = process_dataset(val_data, tokenizer, args.max_length, "Processing val")
    test_processed = process_dataset(test_data, tokenizer, args.max_length, "Processing test")

    # Label distribution
    print("\nLabel distribution:")
    for name, data in [("Train", train_processed), ("Val", val_processed), ("Test", test_processed)]:
        total = len(data)
        if total == 0:
            print(f"  {name}: EMPTY (0 items)")
            continue

        labels = [item['label'].item() for item in data]
        label_counts = Counter(labels)

        if config.NUM_LABELS == 2:
            non_off = label_counts.get(0, 0)
            off = label_counts.get(1, 0)
            print(f"  {name}: {non_off} normal ({non_off/total:.2%}), "
                  f"{off} offensive/hatespeech ({off/total:.2%})")
        else:
            normal = label_counts.get(0, 0)
            offensive = label_counts.get(1, 0)
            hatespeech = label_counts.get(2, 0)
            print(f"  {name}: {normal} normal ({normal/total:.2%}), "
                  f"{offensive} offensive ({offensive/total:.2%}), "
                  f"{hatespeech} hatespeech ({hatespeech/total:.2%})")

    # Rationale coverage
    print("\nRationale coverage:")
    for name, data in [("Train", train_processed), ("Val", val_processed), ("Test", test_processed)]:
        total = len(data)
        if total == 0:
            continue
        has_rat = sum(1 for d in data if d['rationale_mask'].sum() > 0)
        print(f"  {name}: {has_rat}/{total} items have non-empty rationale masks "
              f"({has_rat/total:.1%})")

    # Target community analysis
    print("\nTarget community analysis:")
    for name, data in [("Train", train_processed), ("Val", val_processed), ("Test", test_processed)]:
        if not data:
            continue
        analysis = analyze_target_communities(data)
        print(f"\n  {name} set:")
        print(f"    Posts with target communities: {analysis['posts_with_targets']} / "
              f"{analysis['total_posts']} ({analysis['posts_with_targets']/analysis['total_posts']*100:.1f}%)")
        print(f"    Unique communities: {analysis['unique_communities']}")
        if analysis['community_counts']:
            print(f"    Top 10 communities:")
            for community, count in analysis['community_counts'].most_common(10):
                print(f"      {community}: {count}")

    # Save processed data
    print("\nSaving processed data...")
    save_processed_data(train_processed, config.TRAIN_FILE)
    save_processed_data(val_processed, config.VAL_FILE)
    save_processed_data(test_processed, config.TEST_FILE)

    print(f"\nData processing complete!")
    print(f"Processed data saved to: {config.PROCESSED_DATA_DIR}")

    # Sample
    if train_processed:
        print("\nSample processed item:")
        sample = train_processed[0]
        print(f"  Post ID: {sample['post_id']}")
        print(f"  Text: {sample['text'][:100]}...")
        print(f"  Label: {sample['label'].item()}")
        print(f"  Soft label: {sample['soft_label'].tolist()}")
        print(f"  Agreement score: {sample['agreement_score']:.2f}")
        print(f"  Target communities: {sample['final_target_category']}")
        print(f"  Input shape: {sample['input_ids'].shape}")
        if sample['binary_rationales']:
            rationale_indices = [i for i, r in enumerate(sample['binary_rationales']) if r == 1]
            if rationale_indices:
                rationale_tokens = [sample['original_tokens'][i] for i in rationale_indices[:5]]
                print(f"  Sample rationale tokens: {rationale_tokens}")


if __name__ == "__main__":
    main()