# (In src/main.py)
import os
import sys
from pathlib import Path
import importlib # To dynamically import config module

from pettingzoo.atari import warlords_v3
import numpy as np

from .agents.baseline_agent import BaselineAgent
from .agents.dqn_agent import DQNAgent
# from . import config # Default config
from . import game_loop

def initialize_agents(agent_setup_config, env_action_space_size, env_raw_obs_shape, current_dqn_hyperparams, cfg_module):
    """
    Initializes agents based on cfg_module.AGENT_SETUP_CONFIG,
    using current_dqn_hyperparams for any DQN agent.
    """
    agent_instances = []
    print("\n--- Initializing Agents for Current Experiment ---")
    for agent_cfg in agent_setup_config: # Use passed agent_setup_config
        agent_name = agent_cfg["name"]
        agent_type = agent_cfg["type"].lower()

        if agent_type == "dqn":
            try:
                dqn_params_for_run = {
                    **current_dqn_hyperparams,
                    "player_name": agent_name,
                    "input_shape_raw": env_raw_obs_shape, # Use dynamically fetched shape
                    "num_actions": env_action_space_size  # Use dynamically fetched size
                }
                dqn_params_for_run.pop('id', None)
                
                # Check for reduced image sizes from fast_test_config
                # These are part of current_dqn_hyperparams if from fast_test
                h = dqn_params_for_run.get("img_height", cfg_module.BASE_DQN_HYPERPARAMS.get("img_height", 84))
                w = dqn_params_for_run.get("img_width", cfg_module.BASE_DQN_HYPERPARAMS.get("img_width", 84))
                dqn_params_for_run["img_height"] = h
                dqn_params_for_run["img_width"] = w

                agent = DQNAgent(**dqn_params_for_run)
                print(f"Instantiated '{agent_name}' as DQNAgent with params: { {k: dqn_params_for_run[k] for k in ['learning_rate', 'batch_size', 'target_update_frequency', 'gamma', 'img_height', 'img_width']} }")
            except Exception as e:
                print(f"Could not initialize DQNAgent '{agent_name}': {e}. Falling back to Baseline.")
                agent = BaselineAgent(player_name=agent_name)
                print(f"Instantiated '{agent_name}' as BaselineAgent (fallback).")
        elif agent_type == "baseline":
            agent = BaselineAgent(player_name=agent_name)
            print(f"Instantiated '{agent_name}' as BaselineAgent.")
        else:
            raise ValueError(f"Unknown agent type '{agent_type}' for agent '{agent_name}'.")
        agent_instances.append(agent)
    return agent_instances

def run_experiments(config_module_name="src.config"):
    """
    Main experiment loop, using a dynamically loaded configuration module.
    """
    try:
        cfg = importlib.import_module(config_module_name)
        print(f"--- Successfully loaded configuration: {config_module_name} ---")
    except ImportError:
        print(f"Error: Could not import configuration module '{config_module_name}'. Falling back to default 'src.config'.")
        cfg = importlib.import_module("src.config")


    print(f"--- Starting Warlords Multi-Experiment Run (Config: {config_module_name}) ---")

    env = warlords_v3.env(render_mode="rgb_array")
    env.reset()
    
    if not env.agents:
        raise RuntimeError("Environment agents list is empty after reset. Cannot proceed.")

    first_agent_id = env.agents[0] 
    # Use environment's actual action space size and obs shape
    # These are independent of config's RAW_OBS_SHAPE which is more like a hint or default
    env_action_space_size = env.action_space(first_agent_id).n
    env_raw_obs_shape = env.observation_space(first_agent_id).shape
    
    print(f"Environment: Warlords_v3")
    print(f"Actual Raw Observation Space Shape: {env_raw_obs_shape}")
    print(f"Actual Action Space Size: {env_action_space_size}")

    for exp_set in cfg.EXPERIMENT_HYPERPARAMETER_SETS:
        exp_id = exp_set.get("id", "unknown_experiment")
        print(f"\n\n>>>>>>>>>> Running Experiment: {exp_id} <<<<<<<<<<")

        current_experiment_dqn_hyperparams = cfg.BASE_DQN_HYPERPARAMS.copy()
        current_experiment_dqn_hyperparams.update(exp_set)

        current_model_dir = cfg.MODEL_DIR_BASE / exp_id
        current_video_dir = cfg.VIDEO_DIR_BASE / exp_id
        current_results_dir = cfg.RESULTS_DIR_BASE / exp_id
        os.makedirs(current_model_dir, exist_ok=True)
        os.makedirs(current_video_dir, exist_ok=True)
        os.makedirs(current_results_dir, exist_ok=True)
        
        print(f"Models for this experiment will be saved to: {current_model_dir}")
        print(f"Videos for this experiment will be saved to: {current_video_dir}")
        print(f"Metrics for this experiment will be saved to: {current_results_dir}")
        
        agent_instances = initialize_agents(
            cfg.AGENT_SETUP_CONFIG, # Use agent setup from the loaded config
            env_action_space_size, 
            env_raw_obs_shape, 
            current_experiment_dqn_hyperparams,
            cfg # Pass the loaded config module to initialize_agents
        )

        game_loop.run_tournament(
            env=env,
            agent_instances=agent_instances,
            num_games=cfg.NUM_GAMES_PER_EXPERIMENT,
            training_mode=cfg.DEFAULT_TRAINING_MODE,
            video_dir=current_video_dir,
            model_dir=current_model_dir,
            results_dir=current_results_dir,
            record_nth_video=1 # Record all for fast test, or adjust
        )
        print(f">>>>>>>>>> Experiment {exp_id} Finished <<<<<<<<<<")

    env.close()
    print(f"\n--- All Warlords Experiments (Config: {config_module_name}) Ended ---")

if __name__ == "__main__":
    # Allows running with a specific config: `python -m src.main src.config_fast_test`
    # Or default: `python -m src.main`
    config_to_use = "src.config" # Default
    if len(sys.argv) > 1:
        config_to_use = sys.argv[1]
    run_experiments(config_module_name=config_to_use)