from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
WEIGHTS_DIR = PROJECT_DIR / "weights"

class Config:
    PROJECT_DIR = PROJECT_DIR
    DATA_DIR = DATA_DIR
    WEIGHTS_DIR = WEIGHTS_DIR
    TRAIN_IMAGE_DIR = DATA_DIR / "ISIC2018_Task3_Training_Input"
    TRAIN_GROUND_TRUTH = (
        DATA_DIR / "ISIC2018_Task3_Training_GroundTruth"
        / "ISIC2018_Task3_Training_GroundTruth.csv"
    )
    # =========================
    # Random Seed
    # =========================
    SEED = 59

    LABELS = [
        "MEL",
        "NV",
        "BCC",
        "AKIEC",
        "BKL",
        "DF",
        "VASC",
    ]

    NUM_CLASSES = len(LABELS)
    CLASS_LABELS = tuple(sorted(LABELS))

    # =========================
    # Image
    # =========================
    IMAGE_SIZE = (224, 224)

    # =========================
    # DataLoader
    # =========================
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    PIN_MEMORY = torch.cuda.is_available()

    # =========================
    # Training
    # =========================
    EPOCHS = 150
    WEIGHT_DECAY = 1e-4
    STUDENT_BACKBONE_LR = 2e-4
    STUDENT_HEAD_LR = 1e-3
    STUDENT_PATIENCE = 30

    # =========================
    # Knowledge Distillation
    # =========================
    TEMPERATURE = 3
    ALPHA = 0.5


cfg = Config()
