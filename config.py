

import os
import torch


SEED = 42
torch.manual_seed(SEED)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset.json")  # HateXplain dataset

PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MAX_LENGTH = 128


NUM_LABELS = 3
LABEL_MAPPING = {'normal': 0, 'offensive': 1, 'hatespeech': 2}

# For binary classification, uncomment:
# NUM_LABELS = 2
# LABEL_MAPPING = {'not offensive': 0, 'offensive': 1}


MODEL_NAME = "bert-base-uncased"
DROPOUT_RATE = 0.1

BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Warning: CUDA not available, using CPU")


LOG_INTERVAL = 100
SAVE_INTERVAL = 500


TRAIN_FILE = os.path.join(PROCESSED_DATA_DIR, "train_data.pt")
VAL_FILE = os.path.join(PROCESSED_DATA_DIR, "val_data.pt")
TEST_FILE = os.path.join(PROCESSED_DATA_DIR, "test_data.pt")
TOKENIZER_FILE = os.path.join(PROCESSED_DATA_DIR, "tokenizer")

# Model save paths
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_hate_speech_model.pt")

METRIC_AVERAGING = "macro" if NUM_LABELS > 2 else "binary"

USE_SUPERVISED_ATTENTION = True 
ATTENTION_ALPHA = 10.0
ATTENTION_LAYER = 8
ATTENTION_HEAD = 7 
RATIONALE_THRESHOLD = 0.5 

ATTENTION_METHOD = 'cls'
# HateXplain specific settings

#MIN_ANNOTATOR_AGREEMENT = 2 SRA original