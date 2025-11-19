import os
from pathlib import Path


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "plantvillage" / "color"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# === CONFIGURAÇÃO DE RETOMADA DE TREINAMENTO ===

# OPÇÃO 1: Treino do Zero (Padrão)
RESUME_TRAINING = False
RESUME_MODEL_PATH = None
INITIAL_EPOCH = 0
RESUME_VAL_ACCURACY = None

# OPÇÃO 2: Continuar Treinamento (Descomente para usar)
# RESUME_TRAINING = True
# RESUME_MODEL_PATH = MODEL_DIR / "plant_disease_classifier.keras"
# INITIAL_EPOCH = 5
# RESUME_VAL_ACCURACY = 0.7634

NUM_CLASSES = 38

MODEL_NAME = "plant_disease_classifier"
CHECKPOINT_PATH = MODEL_DIR / f"{MODEL_NAME}.keras"

for directory in [DATA_DIR, DATASET_DIR, MODEL_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
