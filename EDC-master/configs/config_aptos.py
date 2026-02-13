import os
import torch

# --------------------------------------------------
# Detect whether running inside Google Colab
# --------------------------------------------------
try:
    from IPython import get_ipython
    ipython = get_ipython()
    IN_COLAB = ipython is not None and "google.colab" in str(ipython)
except ImportError:
    IN_COLAB = False

ENV = "colab" if IN_COLAB else "local"


# --------------------------------------------------
# Select device based on environment
# --------------------------------------------------
if ENV == "colab":
    # In Colab always prefer CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    # Local M1/M2/M3/M4 Mac → use MPS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


# --------------------------------------------------
# Base directory detection
# --------------------------------------------------
if ENV == "colab":
    BASE_DIR = "/content/MoCo_EDC_DIFF"
else:
    # Local: assuming config.py is at:
    #   APTOS_EDC/EDC-master/config_aptos.py
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../."))


# --------------------------------------------------
# Project directories
# --------------------------------------------------

CODE_DIR = os.path.join(BASE_DIR, "EDC-master")

# ---- APTOS dataset ----
# Must contain:
#   APTOS/
#      ├── train/NORMAL
#      └── test/NORMAL / ABNORMAL
if os.path.exists(os.path.join(BASE_DIR, "MoCo_APTOS")):
    DATASET_DIR = os.path.join(BASE_DIR, "MoCo_APTOS")
else:
    DATASET_DIR = os.path.join(BASE_DIR, "APTOS")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR  = os.path.join(DATASET_DIR, "test")

# Model save directory
SAVED_MODELS_DIR = os.path.join(CODE_DIR, "saved_models")


# --------------------------------------------------
# Pretty print configuration
# --------------------------------------------------
def print_config():
    print("===== APTOS CONFIGURATION =====")
    print(f"Environment:     {ENV}")
    print(f"Device:          {device}")
    print(f"Base directory:  {BASE_DIR}")
    print(f"Code directory:  {CODE_DIR}")
    print(f"Dataset root:    {DATASET_DIR}")
    print(f"Train folder:    {TRAIN_DIR}")
    print(f"Test folder:     {TEST_DIR}")
    print(f"Saved models:    {SAVED_MODELS_DIR}")
    print("================================")
