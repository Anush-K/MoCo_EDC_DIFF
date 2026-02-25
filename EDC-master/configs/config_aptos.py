import os
import torch

# --------------------------------------------------
# Detect environment: Colab, Server, or Local Mac
# --------------------------------------------------
try:
    from IPython import get_ipython
    ipython = get_ipython()
    IN_COLAB = ipython is not None and "google.colab" in str(ipython)
except ImportError:
    IN_COLAB = False

IN_SERVER = (not IN_COLAB) and torch.cuda.is_available() and (not torch.backends.mps.is_available())
IN_MAC    = (not IN_COLAB) and torch.backends.mps.is_available()

if IN_COLAB:
    ENV = "colab"
elif IN_SERVER:
    ENV = "server"
elif IN_MAC:
    ENV = "mac"
else:
    ENV = "cpu"

# --------------------------------------------------
# Device
# --------------------------------------------------
if IN_COLAB or IN_SERVER:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif IN_MAC:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# --------------------------------------------------
# Base directory & paths
# --------------------------------------------------
if ENV == "colab":
    BASE_DIR = "/content/MoCo_EDC_DIFF"

elif ENV == "server":
    # College server: /home/cs24d008/EDC_SSL/EDC-master
    BASE_DIR = "/home/cs24d008/EDC_SSL"

else:
    # Local Mac: configs/config_aptos.py → go up two levels
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

CODE_DIR         = os.path.join(BASE_DIR, "EDC-master")
MOCO_WEIGHTS_DIR = os.path.join(BASE_DIR, "EDC_5Dataset_SSL_Weights")
MOCO_WEIGHTS     = os.path.join(MOCO_WEIGHTS_DIR, "moco_all5datasets_allN_200ep.pth")

# ---- APTOS dataset ----
if ENV == "server":
    DATASET_DIR = os.path.join(BASE_DIR, "APTOS")
elif ENV == "colab":
    if os.path.exists(os.path.join(BASE_DIR, "MoCo_APTOS")):
        DATASET_DIR = os.path.join(BASE_DIR, "MoCo_APTOS")
    else:
        DATASET_DIR = os.path.join(BASE_DIR, "APTOS")
else:
    DATASET_DIR = os.path.join(BASE_DIR, "APTOS")

TRAIN_DIR        = os.path.join(DATASET_DIR, "train")
TEST_DIR         = os.path.join(DATASET_DIR, "test")
SAVED_MODELS_DIR = os.path.join(CODE_DIR, "saved_models")


def print_config():
    print("===== APTOS CONFIGURATION =====")
    print(f"Environment:     {ENV}")
    print(f"Device:          {device}")
    print(f"Base directory:  {BASE_DIR}")
    print(f"Code directory:  {CODE_DIR}")
    print(f"Dataset root:    {DATASET_DIR}")
    print(f"Train folder:    {TRAIN_DIR}")
    print(f"Test folder:     {TEST_DIR}")
    print(f"MoCo weights:    {MOCO_WEIGHTS}")
    print(f"Saved models:    {SAVED_MODELS_DIR}")
    print("================================")