# (In src/game_loop.py)
import os
import imageio
from collections import defaultdict, Counter
import pandas as pd # For saving metrics
from pathlib import Path # For robust path handling

# DQNAgent class might be needed for isinstance checks if we want agent-specific metrics
# from .agents.dqn_agent import DQNAgent

def run_episode(env, agent_instances_map, agent_name_map, episode_id, training_mode, video_dir, record_video=True):
    # ... (setup as before: env.reset(), agent_instance.reset_episode_state()) ...
    env.reset()

    for agent_instance in agent_instances_map.values():
        if hasattr(agent_instance, 'reset_episode_state'):
            agent_instance.reset_episode_state()

    # Metrics for this single episode
    episode_metrics = {
        'episode_id': episode_id,
        # We will populate scores and points per agent later
    }
    # Initialize agent-specific metrics
    for agent_custom_name in agent_name_map.values():
        episode_metrics[f'{agent_custom_name}_score'] = 0
        episode_metrics[f'{agent_custom_name}_points_won'] = 0
    
    # Store DQN specific metrics if applicable (can be extended)
    dqn_agent_name = None
    for env_id, agent_obj in agent_instances_map.items():
        if hasattr(agent_obj, 'current_epsilon'): # Heuristic for DQNAgent
            dqn_agent_name = agent_name_map[env_id]
            episode_metrics[f'{dqn_agent_name}_epsilon_start_episode'] = agent_obj.current_epsilon
            break # Assuming only one main DQN agent for this metric logging

    agent_prev_s_a = {env_agent_id: {'state': None, 'action': None} for env_agent_id in env.agents}
    frames = []
    # --- Main agent iteration loop ---
    for env_agent_id in env.agent_iter():
        observation_raw, reward, termination, truncation, info = env.last()
        
        agent_obj = agent_instances_map[env_agent_id]
        agent_custom_name = agent_name_map[env_agent_id]

        # Accumulate metrics directly into the episode_metrics dict
        episode_metrics[f'{agent_custom_name}_score'] += reward
        if reward > 0:
            episode_metrics[f'{agent_custom_name}_points_won'] += 1

        done = termination or truncation
        action = None
        is_dqn_agent = hasattr(agent_obj, 'get_stacked_frames')

        if is_dqn_agent:
            current_stacked_state = agent_obj.get_stacked_frames(observation_raw)
            prev_s_dict_entry = agent_prev_s_a[env_agent_id]
            if prev_s_dict_entry['state'] is not None and prev_s_dict_entry['action'] is not None:
                agent_obj.replay_buffer.add(
                    prev_s_dict_entry['state'], prev_s_dict_entry['action'], 
                    reward, current_stacked_state, done
                )
                if training_mode:
                    agent_obj.train_step() # Potential place to get loss
            
            if done:
                action = None 
                agent_prev_s_a[env_agent_id] = {'state': None, 'action': None}
                if hasattr(agent_obj, 'reset_episode_state'):
                    agent_obj.reset_episode_state()
            else:
                action = agent_obj.act(current_stacked_state, training=training_mode)
                agent_prev_s_a[env_agent_id] = {'state': current_stacked_state, 'action': action}
        else: # Baseline
            action = agent_obj.act(observation_raw) if not done else None
        
        env.step(action)
        if record_video:
            try:
                frames.append(env.render())
            except Exception as e:
                print(f"Warning: Could not render frame for video: {e}")
                record_video = False
    
    # Log epsilon at the end of the episode for the DQN agent
    if dqn_agent_name:
         for agent_obj in agent_instances_map.values(): # Find the DQN agent again
            if hasattr(agent_obj, 'current_epsilon') and agent_obj.name == dqn_agent_name:
                episode_metrics[f'{dqn_agent_name}_epsilon_end_episode'] = agent_obj.current_epsilon
                episode_metrics[f'{dqn_agent_name}_total_steps_end_episode'] = agent_obj.total_steps_taken
                break
    
    # ... (video saving as before) ...
    if record_video and frames:
        video_path = Path(video_dir) / f"game_{episode_id}.mp4"
        try:
            imageio.mimsave(video_path, frames, fps=30)
            print(f"Video saved to {video_path}")
        except Exception as e:
            print(f"Error saving video: {e}. Frames: {len(frames)}")

    return episode_metrics # Return the dictionary of metrics for this episode


def run_tournament(env, agent_instances, num_games, training_mode, video_dir, model_dir, results_dir, record_nth_video=10):
    """
    Runs a tournament, collects per-episode metrics, and saves them.
    """
    if not env.agents:
        env.reset()
    if not env.agents:
        raise ValueError("Environment has no agents.")
            
    agent_instances_map = {env.agents[i]: agent_instances[i] for i in range(len(env.agents))}
    agent_name_map = {env.agents[i]: agent_instances[i].name for i in range(len(env.agents))}

    all_episodes_metrics = [] # List to store metrics dictionaries from each episode

    for game_idx in range(num_games):
        print(f"\n--- Running Game {game_idx + 1}/{num_games} ---")
        should_record_video = (game_idx + 1) % record_nth_video == 0 or game_idx == 0 or game_idx == num_games -1

        # Run episode and get detailed metrics
        current_episode_data = run_episode(
            env, agent_instances_map, agent_name_map, game_idx, 
            training_mode, video_dir, record_video=should_record_video
        )
        all_episodes_metrics.append(current_episode_data)

        # Print summary for the current episode
        print(f"Game {game_idx + 1} Summary:")
        for agent_name in agent_name_map.values():
            score = current_episode_data.get(f'{agent_name}_score', 0)
            points = current_episode_data.get(f'{agent_name}_points_won', 0)
            print(f"  {agent_name}: Score = {score}, Points Won = {points}")
        
        # Print DQN agent's Epsilon if available
        for agent_obj in agent_instances:
             if hasattr(agent_obj, 'current_epsilon'):
                print(f"  {agent_obj.name} (DQN): Epsilon = {agent_obj.current_epsilon:.4f}, Total Steps = {agent_obj.total_steps_taken}")
    
    # --- After all games ---
    # Convert list of dicts to DataFrame
    metrics_df = pd.DataFrame(all_episodes_metrics)
    
    # Save metrics to CSV
    results_file_path = Path(results_dir) / "episode_metrics.csv"
    try:
        metrics_df.to_csv(results_file_path, index=False)
        print(f"\nEpisode metrics saved to: {results_file_path}")
    except Exception as e:
        print(f"Error saving metrics CSV: {e}")

    # --- Print Final Tournament Summary (Aggregated) ---
    print("\n--- Tournament Finished ---")
    print("\n--- Final Aggregated Results ---")
    
    # Calculate total scores and points from the DataFrame
    # This assumes column names like 'Agent1_DQN_score', 'Agent2_Baseline_points_won'
    final_summary = {}
    for agent_name in agent_name_map.values():
        total_score = metrics_df[f'{agent_name}_score'].sum()
        total_points = metrics_df[f'{agent_name}_points_won'].sum()
        avg_score_per_episode = metrics_df[f'{agent_name}_score'].mean()
        final_summary[agent_name] = {
            'total_score': total_score,
            'total_points': total_points,
            'avg_score_per_episode': avg_score_per_episode
        }
        print(f"{agent_name}:")
        print(f"  Total Score (sum over all games) = {total_score}")
        print(f"  Total 'Points' Won (sum over all games) = {total_points}")
        print(f"  Average Score per Episode = {avg_score_per_episode:.2f}")

    # Determine overall winner by total points won
    try:
        # Create a Counter for total points from the DataFrame for easy sorting
        overall_points_counter = Counter()
        for agent_name in agent_name_map.values():
            overall_points_counter[agent_name] = metrics_df[f'{agent_name}_points_won'].sum()

        if overall_points_counter:
            winner_by_points = overall_points_counter.most_common(1)[0]
            print(f"\nOverall Winner (most total points): {winner_by_points[0]} with {winner_by_points[1]} points!")
        else:
            print("\nNo 'points' were scored by any agent across all games.")
    except Exception as e: # Catch broad exception for robust summary
        print(f"\nCould not determine winner by points due to: {e}")


    # Save DQN models (as before)
    for agent_instance in agent_instances:
        if hasattr(agent_instance, 'save_model'):
            model_save_path = Path(model_dir) / f"{agent_instance.name}_model.pth"
            agent_instance.save_model(model_save_path)