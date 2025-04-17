
import os


MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS = os.path.join(MODULE_PATH, "checkpoints/")
TRAIN_RESULTS_PATH = os.path.join(MODULE_PATH, "train_results/")

os.makedirs(CHECKPOINTS, exist_ok=True)
