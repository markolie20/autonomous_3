# autonomous_3 Project Overview

This project implements a multi-agent reinforcement learning (RL) tournament environment for the Atari game Warlords, using the PettingZoo library. It is designed to facilitate experiments comparing a DQN agent against baseline agents, with automated training, evaluation, and analysis workflows.

## Structure and Workflow

### 1. Experiment Configuration

- All experiment settings are defined in `src/config.py`, including:
  - Agent names and types (DQN or baseline).
  - Hyperparameter sets for DQN experiments.
  - Directory paths for saving models, videos, and results.

### 2. Running Experiments

- The main experiment loop is in `src/main.py`:
  - Dynamically loads configuration.
  - Initializes the Warlords environment and agents.
  - Iterates over each experiment configuration, running a tournament for each.
  - Saves per-episode metrics, trained models, and gameplay videos for each experiment.

- The core tournament/gameplay logic is in `src/game_loop.py`:
  - Runs multiple episodes (games) per experiment.
  - Collects detailed metrics for each agent (scores, points won, DQN epsilon, etc.).
  - Saves results to CSV and videos to disk.

### 3. Agents

- `src/agents/dqn_agent.py`: Implements a DQN agent with frame stacking, experience replay, and target network updates.
- `src/agents/baseline_agent.py`: Implements a simple baseline agent that acts randomly.

### 4. Neural Network and Replay Buffer

- `src/networks/dqn_cnn.py`: Defines the convolutional neural network used by the DQN agent.
- `src/utils/replay_buffer.py`: Implements a replay buffer for experience sampling.

### 5. Results Analysis

- `src/analysis/plot_results.py`:
  - Loads experiment results from CSV files.
  - Generates learning curves and comparison plots for agent performance.
  - Saves plots to the results directory for easy review.

### 6. Notebooks

- `notebooks/experiment_runner.ipynb`: Example notebook for running experiments, especially in environments like Google Colab.
- `notebooks/evaluation_notebook.ipynb`: Loads results and generates plots for analysis.

## Data and Outputs

- **Models:** Saved in `models_experiments` per experiment.
- **Videos:** Saved in `videos_experiments` per experiment.
- **Results:** Per-episode metrics in `results_experiments`, with analysis plots in subfolders.

## Customization

- Add new experiment configurations in `src/config.py` under `EXPERIMENT_HYPERPARAMETER_SETS`.
- Implement new agent types by extending the agent classes in `src/agents`.
