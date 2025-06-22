# (In src/analysis/plot_results.py)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def load_experiment_data(base_results_dir, experiment_id):
    """Loads metrics CSV for a given experiment ID."""
    results_file = Path(base_results_dir) / experiment_id / "episode_metrics.csv"
    if results_file.exists():
        return pd.read_csv(results_file)
    else:
        print(f"Warning: Metrics file not found for experiment {experiment_id} at {results_file}")
        return None

def plot_learning_curves(all_experiments_data, agent_name_pattern, metric_column_suffix='_score', smoothing_window=10, output_dir="."):
    """
    Plots learning curves (e.g., score over episodes) for a specific agent type across all experiments.
    
    Args:
        all_experiments_data (dict): Dict where keys are exp_id and values are pandas DataFrames of metrics.
        agent_name_pattern (str): A pattern to identify the agent (e.g., "_DQN" for DQN agents).
                                  The plot will pick the first agent matching this.
        metric_column_suffix (str): The suffix for the metric column (e.g., '_score', '_points_won').
        smoothing_window (int): Window size for rolling mean.
        output_dir (str/Path): Directory to save the plot.
    """
    plt.figure(figsize=(12, 7))
    
    for exp_id, df in all_experiments_data.items():
        if df is None:
            continue
            
        # Find the specific agent's column
        # This assumes agent names are consistent (e.g., 'Agent1_DQN', 'Agent2_Baseline')
        agent_column = None
        for col in df.columns:
            if agent_name_pattern in col and col.endswith(metric_column_suffix):
                agent_column = col
                break
        
        if agent_column:
            metric_values = df[agent_column]
            if smoothing_window > 1:
                smoothed_metric = metric_values.rolling(window=smoothing_window, min_periods=1, center=True).mean()
            else:
                smoothed_metric = metric_values
            plt.plot(df['episode_id'], smoothed_metric, label=f'{exp_id} ({agent_column.split(metric_column_suffix)[0]})')
        else:
            print(f"Warning: No agent column matching '{agent_name_pattern}{metric_column_suffix}' found in experiment {exp_id}")

    plt.xlabel("Episode")
    plt.ylabel(f"Smoothed {metric_column_suffix.replace('_', ' ').strip().title()}")
    plt.title(f"Learning Curves for Agent type '{agent_name_pattern}' ({metric_column_suffix.replace('_', ' ').strip()})")
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.) # Adjust legend position
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    
    plot_filename = Path(output_dir) / f"learning_curves_{agent_name_pattern.replace('_','')}{metric_column_suffix}.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.show()


def plot_comparison_bar_chart(all_experiments_data, metric_column_suffix='_score', output_dir="."):
    """
    Plots a bar chart comparing the average final metric (e.g. average score) 
    for all agents across different experiments.
    """
    avg_metrics = []
    
    for exp_id, df in all_experiments_data.items():
        if df is None:
            continue
        
        # Identify all agent metric columns for the given suffix
        agent_metric_cols = [col for col in df.columns if col.endswith(metric_column_suffix)]

        for agent_col in agent_metric_cols:
            agent_name = agent_col.replace(metric_column_suffix, "")
            # Consider average of last N episodes or overall average
            # Here, we take the overall average for simplicity
            avg_value = df[agent_col].mean()
            avg_metrics.append({'experiment_id': exp_id, 'agent_name': agent_name, 'average_metric': avg_value})

    if not avg_metrics:
        print("No data to plot for bar chart.")
        return

    summary_df = pd.DataFrame(avg_metrics)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='experiment_id', y='average_metric', hue='agent_name', data=summary_df, dodge=True)
    
    plt.xlabel("Experiment ID")
    plt.ylabel(f"Average {metric_column_suffix.replace('_', ' ').strip().title()} per Episode")
    plt.title(f"Agent Performance Comparison ({metric_column_suffix.replace('_', ' ').strip()})")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Agent", loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.grid(axis='y')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plot_filename = Path(output_dir) / f"performance_comparison_bar{metric_column_suffix}.png"
    plt.savefig(plot_filename)
    print(f"Bar chart saved to {plot_filename}")
    plt.show()


def main_analysis(base_results_dir, experiment_ids_to_analyze=None):
    """
    Main function to load data from all experiments and generate plots.
    
    Args:
        base_results_dir (str/Path): The base directory where experiment results are stored.
        experiment_ids_to_analyze (list, optional): Specific experiment IDs to analyze. 
                                                     If None, tries to find all subdirs.
    """
    base_results_dir = Path(base_results_dir)
    plots_output_dir = base_results_dir / "analysis_plots"
    os.makedirs(plots_output_dir, exist_ok=True)

    if experiment_ids_to_analyze is None:
        # Automatically find experiment directories
        experiment_ids_to_analyze = [d.name for d in base_results_dir.iterdir() if d.is_dir() and d.name != "analysis_plots"]
        if not experiment_ids_to_analyze:
            print(f"No experiment subdirectories found in {base_results_dir}")
            return
            
    print(f"Analyzing experiments: {experiment_ids_to_analyze}")

    all_data = {}
    for exp_id in experiment_ids_to_analyze:
        df = load_experiment_data(base_results_dir, exp_id)
        if df is not None:
            all_data[exp_id] = df
            
    if not all_data:
        print("No data loaded. Exiting analysis.")
        return

    # --- Generate Plots ---
    # Plot learning curves for the primary DQN agent (assuming its name contains '_DQN')
    plot_learning_curves(all_data, agent_name_pattern='_DQN', metric_column_suffix='_score', smoothing_window=10, output_dir=plots_output_dir)
    plot_learning_curves(all_data, agent_name_pattern='_DQN', metric_column_suffix='_points_won', smoothing_window=10, output_dir=plots_output_dir)

    # Plot learning curves for a baseline agent (e.g., 'Agent2_Baseline')
    # This shows how consistent the baseline is.
    plot_learning_curves(all_data, agent_name_pattern='Baseline', metric_column_suffix='_score', smoothing_window=5, output_dir=plots_output_dir)

    # Plot comparison bar charts
    plot_comparison_bar_chart(all_data, metric_column_suffix='_score', output_dir=plots_output_dir)
    plot_comparison_bar_chart(all_data, metric_column_suffix='_points_won', output_dir=plots_output_dir)
    
    print(f"All analysis plots saved in: {plots_output_dir}")

if __name__ == '__main__':
    # This allows running the analysis script directly
    # You'd need to ensure paths are correct or pass them as arguments
    # Example:
    # current_project_root = Path(__file__).resolve().parent.parent.parent # autonomous_3
    # default_results_base = current_project_root / "results_experiments"
    
    # For simplicity, if you run this script, ensure your CWD is `autonomous_3`
    # or modify pathing.
    from src import config as main_config # If CWD is autonomous_3
    
    print(f"Running analysis from plot_results.py using results from: {main_config.RESULTS_DIR_BASE}")
    main_analysis(main_config.RESULTS_DIR_BASE)