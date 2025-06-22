import os
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- General Settings ---
AGENT_NAMES = ['Agent1_DQN', 'Agent2_Baseline', 'Agent3_Baseline', 'Agent4_Baseline']
NUM_GAMES_PER_EXPERIMENT = 50 # Number of episodes per hyperparameter set
# TRAINING_MODE_ENABLED is always True for experiments, can be overridden if needed for evaluation runs
DEFAULT_TRAINING_MODE = True 

# --- Paths ---
VIDEO_DIR_BASE = PROJECT_ROOT / "videos_experiments" # Base directory for experiment videos
MODEL_DIR_BASE = PROJECT_ROOT / "models_experiments" # Base directory for experiment models
RESULTS_DIR_BASE = PROJECT_ROOT / "results_experiments" # Base directory for experiment metrics

# --- Environment Specifics (Can be auto-detected but defaults are good) ---
RAW_OBS_SHAPE = (210, 160, 3)
ACTION_SPACE_SIZE = 6

# --- Base DQN Agent Hyperparameters ---
# These are the default values. Experiments will override specific keys.
BASE_DQN_HYPERPARAMS = {
    "buffer_size": 50000,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_steps": 100000,
    "learn_start_steps": 10000,
    "frame_stack_size": 4,
    "img_height": 84,
    "img_width": 84,
    # Hyperparameters to be varied in experiments:
    "learning_rate": 0.0001,
    "batch_size": 32,
    "target_update_frequency": 1000,
    "gamma": 0.99,
}

# --- Experiment Configurations ---
# Each dictionary in this list defines a specific set of hyperparameters to test.
# The 'id' will be used for naming output folders.
# Keys here will override those in BASE_DQN_HYPERPARAMS.
EXPERIMENT_HYPERPARAMETER_SETS = [
    {
        "id": "lr0.00025_bs32_uf1000_g0.99", # Original DQN paper LR
        "learning_rate": 0.00025,
        "batch_size": 32,
        "target_update_frequency": 1000, # or 10000 agent steps, often 4 environment steps for DQN
        "gamma": 0.99,
    },
    {
        "id": "lr0.0001_bs64_uf1000_g0.99",
        "learning_rate": 0.0001,
        "batch_size": 64, # Larger batch size
        "target_update_frequency": 1000,
        "gamma": 0.99,
    },
    {
        "id": "lr0.0001_bs32_uf500_g0.99",
        "learning_rate": 0.0001,
        "batch_size": 32,
        "target_update_frequency": 500, # More frequent target updates
        "gamma": 0.99,
    },
    {
        "id": "lr0.0001_bs32_uf1000_g0.95",
        "learning_rate": 0.0001,
        "batch_size": 32,
        "target_update_frequency": 1000,
        "gamma": 0.95, # Less emphasis on future rewards
    },
    # Add more configurations as needed
    # Example: default values from BASE_DQN_HYPERPARAMS
    {
        "id": "default_params", # This will use the exact values from BASE_DQN_HYPERPARAMS
                                # if the keys below match and values are identical,
                                # or you can just omit the overriding keys to use base.
        "learning_rate": BASE_DQN_HYPERPARAMS["learning_rate"],
        "batch_size": BASE_DQN_HYPERPARAMS["batch_size"],
        "target_update_frequency": BASE_DQN_HYPERPARAMS["target_update_frequency"],
        "gamma": BASE_DQN_HYPERPARAMS["gamma"],
    }
]

# --- Agent Configuration for each tournament ---
# Defines the agent types. The DQN agent will use the hyperparams from the current experiment.
AGENT_SETUP_CONFIG = [
    {"type": "dqn", "name": AGENT_NAMES[0]}, # This DQN will get experimental HPs
    {"type": "baseline", "name": AGENT_NAMES[1]},
    {"type": "baseline", "name": AGENT_NAMES[2]},
    {"type": "baseline", "name": AGENT_NAMES[3]},
]

# Ensure agent names match the length of AGENT_SETUP_CONFIG
if len(AGENT_NAMES) != len(AGENT_SETUP_CONFIG):
    raise ValueError("Length of AGENT_NAMES must match length of AGENT_SETUP_CONFIG.")
for i in range(len(AGENT_NAMES)):
    AGENT_SETUP_CONFIG[i]["name"] = AGENT_NAMES[i] # Assign names from AGENT_NAMES