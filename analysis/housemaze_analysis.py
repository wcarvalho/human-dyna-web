from typing import NamedTuple, List

from functools import partial
from flax import struct
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import os.path
from scipy import stats
import seaborn as sns

from housemaze import renderer
from housemaze.human_dyna import utils
from housemaze.human_dyna import multitask_env
from math import sqrt, ceil
from statsmodels.stats.power import TTestPower, TTestIndPower


from analysis.housemaze_model_data import get_model_data
from analysis.housemaze_user_data import get_human_data
from analysis.housemaze_user_data import get_valid_files
from nicewebrl.dataframe import DataFrame

DEFAULT_TITLE_SIZE = 14
DEFAULT_LABEL_SIZE = 12
DEFAULT_LEGEND_SIZE = 10

image_dict = utils.load_image_dict()

default_colors = {
    "reddish purple": (204/255, 121/255, 167/255),
    "yellow": (240/255, 228/255, 66/255),
    "orange": (230/255, 159/255, 0.0), 
    "vermillion": (213/255, 94/255, 0.0),
    "sky blue": (86/255, 180/255, 233/255),
    "bluish green": (0.0, 158/255, 115/255),
    "blue": (0.0, 114/255, 178/255),
    "black": "#2f2f2e",
    'dark gray': "#666666",
    'light gray': "#999999",
    'purple': '#CC79A7',
    'nice purple': '#9B80E6',
    "pretty blue": "#679FE5",
    "google blue": "#186CED",
    "google orange": "#FFB700",
    'white': "#FFFFFF",
}

model_colors = {
    #'human_success': '#0072B2',
    'human': default_colors['orange'],
    #'human_terminate': '#D55E00',
    'usfa': default_colors['light gray'],
    'qlearning': default_colors['dark gray'],
    'dynaq_shared': default_colors["vermillion"],
    'bfs': default_colors['pretty blue'],
    'dfs': default_colors["sky blue"]
}

model_names = {
    'human': 'Human',
    'human_terminate': 'Human (finished)',
    'human_success': 'Human (Succeeded)',
    'qlearning': 'Q-learning',
    'usfa': 'Successor features',
    'dynaq_shared': 'Multi-task preplay',
    'bfs': 'Breadth-first search',
    'dfs': 'Depth-first search',
}

model_order = [
    'human',
    'human_success',
    'human_terminate',
    'usfa',
    'dynaq_shared',
    'qlearning',
    'bfs',
    'dfs']

maze_name = {
    'big_m2_maze2': "Start Manipulation",
    'big_m2_maze2_offpath': "Start Manipulation: Off-path",
    'big_m2_maze2_onpath': "Start Manipulation: On-path",
    'big_m3_maze1': "Path Manipulation",
    'big_m3_maze1_eval': "Path Manipulation: Evaluation",
    'big_m4_maze_long': "Plan Manipulation (long)",
    'big_m4_maze_long_eval_same': "Plan Manipulation (long): Same location",
    'big_m4_maze_long_eval_diff': "Plan Manipulation (long): New location",
    'big_m4_maze_short': "Plan Manipulation (short)",
    'big_m4_maze_short_eval_same': "Plan Manipulation (short): Same location",
    'big_m4_maze_short_eval_diff': "Plan Manipulation (short): New location",
}


class EpisodeData(NamedTuple):
    actions: jax.Array
    timesteps: struct.PyTreeNode  # housemaze.human_dyna.multitask_env.TimeStep
    positions: jax.Array = None
    reaction_times: jax.Array = None
    transitions: struct.PyTreeNode = None

def success(e: EpisodeData):
    """Returns 1.0 if episode received reward > 0.5 at any timestep, 0.0 otherwise."""
    rewards = e.timesteps.reward
    # return rewards
    assert rewards.ndim == 1, 'this is only defined over vector, e.g. 1 episode'
    success = rewards > .5
    return success.any().astype(np.float32)

def get_in_episode(timestep):
  # get mask for within episode
  non_terminal = timestep.discount
  is_last = timestep.last()
  term_cumsum = jnp.cumsum(is_last, -1)
  in_episode = (term_cumsum + non_terminal) < 2
  return in_episode


def filter_train_by_min_success(df: DataFrame, min_successes: int = 16):
    """Filter out users who did not achieve enough successes during training.

    Args:
        df (DataFrame): DataFrame containing episodes for a single user
        min_successes (int, optional): Minimum number of successful episodes required. Defaults to 16.

    Returns:
        bool: True if user should be removed (did not meet min_successes), False otherwise
        
    Notes:
        - Applies success() function to each episode to count total successes
        - Prints debug info about removed users including:
            - Success rate
            - Number of successes vs minimum required
            - Total number of episodes
    """
    successes = df.apply(success)
    remove = True
    if len(successes) == 0:
        return remove

    user = df['user_id'].unique().to_list()[0]
    nsuccess = int(sum(successes))
    remove = nsuccess < min_successes
    if remove:
        print(
            f"removed: user {user} rate: {np.mean(successes)} = {nsuccess}/{min_successes}/{len(successes)}")
    return remove


def filter_outliers(
        df: DataFrame,
        filter_columns: List[str],
        method: str = 'iqr',
        verbose: bool = False,
        threshold: float = 1.5) -> DataFrame:
  """Filter outliers from specified columns using various methods.
  
  Args:
      df (DataFrame): Input DataFrame
      filter_columns (list): List of column names to check for outliers
      method (str): Method to use for outlier detection ('iqr', 'zscore', or 'percentile')
      threshold (float): Threshold for outlier detection:
          - For IQR: Number of IQRs to use (default: 1.5)
          - For zscore: Number of standard deviations (default: 3)
          - For percentile: Percentile range from 0-100 (default: 1, meaning 1st-99th percentile)
  
  Returns:
      DataFrame: Filtered DataFrame with outliers removed
  """
  original_size = len(df)
  
  # Create mask (all True initially)
  mask = np.ones(len(df), dtype=bool)
  bounds = {}

  # Check each column
  for col in filter_columns:
    values = np.array(df[col])
    
    if method == 'iqr':
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
    elif method == 'zscore':
        mean = np.mean(values)
        std = np.std(values)
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        
    elif method == 'percentile':
        lower_bound = np.percentile(values, threshold)
        upper_bound = np.percentile(values, 100 - threshold)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    bounds[col] = {
        'lower': lower_bound,
        'upper': upper_bound
    }
    
    # Update mask with column constraints
    col_mask = (values >= lower_bound) & (values <= upper_bound)
    mask &= col_mask

  # Apply the mask to filter outliers
  filtered_df = df.filter(pl.Series(mask))

  # Print how many were filtered
  n_filtered = original_size - len(filtered_df)
  if n_filtered > 0 and verbose:
    removed_df = df.filter(pl.Series(~mask))
    users_removed = set()
    print(f"Filtered {n_filtered} outliers from {original_size} rows using {method} method")
    for row in removed_df.iter_rows(named=True):
      user = str(row['user_id'])
      users_removed.add(user)
      for col in filter_columns:
          val = row[col]
          bound = bounds[col]
          if val < bound['lower'] or val > bound['upper']:
              print(
                  f"\t{user}. {col}:{val} \t(bounds: {bound['lower']} to {bound['upper']})")
    print(f"Users removed: {users_removed}")

  return filtered_df


def housemaze_render_fn(state: multitask_env.EnvState):
    return renderer.create_image_from_grid(
        state.grid,
        state.agent_pos,
        state.agent_dir,
        image_dict)


def render_path(episode_data, from_model=True, ax=None):
    # get actions that are in episode
    timesteps = episode_data.timesteps
    actions = episode_data.actions
    if from_model:
      in_episode = get_in_episode(timesteps)
      actions = actions[in_episode][:-1]
      positions = jax.tree_map(
          lambda x: x[in_episode][:-1], timesteps.state.agent_pos)
    else:
       positions = timesteps.state.agent_pos[:-1]
    # positions in episode

    state_0 = jax.tree_map(lambda x: x[0], timesteps.state)

    # doesn't matter
    maze_height, maze_width, _ = timesteps.state.grid[0].shape

    if ax is None:
      fig, ax = plt.subplots(1, figsize=(5, 5))
    img = housemaze_render_fn(state_0)

    renderer.place_arrows_on_image(
        img, positions, actions, maze_height, maze_width, arrow_scale=5, ax=ax)


def render_paths(episode_list, render_both: bool=False, colors=None, from_model=True, ax=None):
    """Renders multiple paths on the same maze image.
    
    Args:
        episode_list: List of episode data to render
        colors: List of colors for each path. If None, uses default color cycle
        from_model: Whether the data is from a model or human
        ax: Matplotlib axis to plot on. If None, creates new figure
    """
    if not episode_list:
        raise ValueError("Episode list cannot be empty")

    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Get base image from first episode
    state_0 = jax.tree_map(lambda x: x[0], episode_list[0].timesteps.state)
    maze_height, maze_width, _ = episode_list[0].timesteps.state.grid[0].shape

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5, 5))
    
    render_both = render_both and len(episode_list) > 1
    if render_both:
       state_1 = jax.tree_map(lambda x: x[0], episode_list[1].timesteps.state)
       positions = [state_0.agent_pos, state_1.agent_pos]
    else:
       positions = [state_0.agent_pos]
    img = renderer.create_image_from_grid(
        state_0.grid,
        positions,
        state_0.agent_dir,
        image_dict)

    # Plot each path
    for i, episode_data in enumerate(episode_list):
        timesteps = episode_data.timesteps
        actions = episode_data.actions

        if from_model:
            in_episode = get_in_episode(timesteps)
            actions = actions[in_episode][:-1]
            positions = jax.tree_map(
                lambda x: x[in_episode][:-1], timesteps.state.agent_pos)
        else:
            positions = timesteps.state.agent_pos[:-1]

        color = colors[i % len(colors)]
        renderer.place_arrows_on_image(
            img, positions, actions, maze_height, maze_width,
            arrow_scale=10, ax=ax, arrow_color=color)


def render_optimal_paths(episode_list, colors=None, from_model=True, ax=None, **kwargs):
    """Renders multiple optimal paths on the same maze image.
    
    Args:
        episode_list: List of episode data to render
        colors: List of colors for each path. If None, uses default color cycle
        from_model: Whether the data is from a model or human
        ax: Matplotlib axis to plot on. If None, creates new figure
    """
    if not episode_list:
        raise ValueError("Episode list cannot be empty")

    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Get base image from first episode
    state_0 = jax.tree_map(lambda x: x[0], episode_list[0].timesteps.state)
    maze_height, maze_width, _ = episode_list[0].timesteps.state.grid[0].shape

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5, 5))

    starting_positions = []
    for episode_data in episode_list:
        state = jax.tree_map(lambda x: x[0], episode_data.timesteps.state)
        starting_positions.append(state.agent_pos)

    img = renderer.create_image_from_grid(
        state_0.grid,
        starting_positions,
        state_0.agent_dir,
        image_dict)

    # Plot each path
    for i, episode_data in enumerate(episode_list):
        grid = episode_data.timesteps.state.grid[0]  # only have 1 map
        agent_pos = episode_data.timesteps.state.agent_pos[0]
        agent_pos = tuple([int(i) for i in agent_pos])
        goal = episode_data.timesteps.state.task_object[0]
        positions = utils.find_optimal_path(grid, agent_pos, goal)
        actions = utils.actions_from_path(positions)

        color = colors[i % len(colors)]
        renderer.place_arrows_on_image(
            img, positions, actions, maze_height, maze_width,
            arrow_scale=10, ax=ax, arrow_color=color)


def bar_plot_error(human_data, model_stats, ax=None, ylim=None, legend=True, xlabels=False):
    """Plot bar chart comparing human and model performance with error bars.
    
    Args:
        human_data: Dict containing human_means and human_se
        model_stats: DataFrame with columns 'algo', 'mean', 'se'
        ax: Optional matplotlib axis
        ylim: Optional y-axis limits tuple
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # Combine human and model data
    all_data = {
        'human': human_data['means']
    }
    # Add model data
    yerr = [human_data['se']]  # Start with human SE
    algos = model_stats['algo'].unique().to_list()
    for algo in model_order:
        if not algo in algos:
           continue
        row = model_stats.filter(algo=algo)
        all_data[algo] = row['mean'].to_numpy()[0]
        yerr.append(row['se'].to_numpy()[0])

    # Plot bars
    x_pos = np.arange(len(all_data))
    ordered_keys = [k for k in model_order if k in all_data]
    bars = ax.bar(x_pos,
           [all_data[k] for k in ordered_keys],
           yerr=yerr,
           capsize=5,
           color=[model_colors.get(k, '#333333') for k in ordered_keys])

    # Update x-tick labels to match new ordering
    if xlabels:
      ax.set_xticks(x_pos)
      ax.set_xticklabels([model_names[k] for k in ordered_keys],
                        rotation=45, ha='right')
    else:
      ax.set_xticks([])

    # Add individual dots for human data with jitter
    if 'raw' in human_data:
        x_jitter = np.random.normal(0, 0.15, size=len(human_data['raw']))
        ax.scatter([0 + j for j in x_jitter], human_data['raw'],
                  color='black', alpha=0.5, zorder=3)

    # Add chance level line
    ax.axhline(y=50, color='r', linestyle='--',
               alpha=0.5, label='Chance level')

    if ylim is not None:
        ax.set_ylim(ylim)

    if legend:
        legend_elements = bars
        legend_labels = [model_names[k] for k in ordered_keys]
        ax.legend(legend_elements, legend_labels)

    ax.grid(True, linestyle='--', alpha=0.7)

    return fig, ax


def plot_bar_rt_comparison(
      dfs,
      rt_column,
      ax=None,
      title=None,
      ylabel=None,
      xlabels=None,
      colors=None,
      stats_file=None,
      ):
    """Plot comparison of reaction times between multiple conditions.
    
    Args:
        episode_sets: List of DataFrames, each containing episodes for one condition
        rt_column: Name of reaction time column to analyze (e.g. 'first_rt', 'avg_rt')
        ax: Optional matplotlib axis to plot on
        display_stats: Whether to print statistical test results
        title: Custom title
        ylabel: Custom y-axis label
        xlabels: List of labels for x-axis ticks
        colors: List of colors for bars
    
    Returns:
        matplotlib axis
    """
    if colors is None:
        colors = [default_colors["vermillion"], default_colors["bluish green"]]
        # Extend colors if needed
        while len(colors) < len(dfs):
            colors.extend(colors)
    
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    # Calculate log RTs and group by user for each set
    log_col = f'log_{rt_column}'
    all_data = []
    for episodes in dfs:
        stats = (episodes
                .with_columns((1000*pl.col(rt_column)).log().alias(log_col)))
        all_data.append(stats[log_col].to_numpy())

    # Calculate means and standard errors
    means = [np.mean(data) for data in all_data]
    sems = [np.std(data) / np.sqrt(len(data)) for data in all_data]

    # Create bar plot
    x_pos = np.arange(len(dfs))
    bars = ax.bar(x_pos, means, yerr=sems, capsize=5, color=colors[:len(dfs)])

    # Add individual points with jitter
    for i, data in enumerate(all_data):
        x_jitter = np.random.normal(i, 0.04, size=len(data))
        ax.scatter(x_jitter, data, alpha=0.3, color='black', s=20)

    # Customize plot
    ax.set_xticks(x_pos)
    if xlabels:
        ax.set_xticklabels(xlabels, ha='center')
    ax.set_ylabel(ylabel or f'Log {rt_column}', fontsize=DEFAULT_LABEL_SIZE)
    ax.set_title(title or f'{rt_column} Comparison', fontsize=DEFAULT_TITLE_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=DEFAULT_LABEL_SIZE)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    all_data_combined = np.concatenate(all_data)
    y_min, y_max = np.percentile(all_data_combined, [1, 99])
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    power_results = power_analysis_between_groups(
        all_data[0], all_data[1], stats_file=stats_file)
    return ax


def plot_mean_rt_comparison(
    dfs,
    rt_column,
    ax=None,
    display_stats=True,
    title=None,
    ylabel=None,
    xlabels=None,
    colors=None,
):
    """Plot comparison of reaction times between multiple conditions.
    
    Args:
        episode_sets: List of DataFrames, each containing episodes for one condition
        rt_column: Name of reaction time column to analyze (e.g. 'first_rt', 'avg_rt')
        ax: Optional matplotlib axis to plot on
        display_stats: Whether to print statistical test results
        title: Custom title
        ylabel: Custom y-axis label
        xlabels: List of labels for x-axis ticks
        colors: Optional list of colors for bars
    
    Returns:
        matplotlib axis
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    # Set default colors if none provided
    if colors is None:
        colors = [default_colors["vermillion"], default_colors["bluish green"]]
        # Extend colors if needed
        while len(colors) < len(dfs):
            colors.extend(colors)

    # Calculate log RTs and group by user for each set
    log_col = f'log_{rt_column}'
    all_data = []
    for episodes in dfs:
        stats = (episodes
                .with_columns((1000*pl.col(rt_column)).log().alias(log_col)))
        all_data.append(stats[log_col].to_numpy())

    # Calculate means and standard errors
    means = [np.mean(data) for data in all_data]
    sems = [np.std(data) / np.sqrt(len(data)) for data in all_data]

    # Create horizontal lines for means
    x_pos = np.linspace(0.25, 0.75, len(dfs))

    # Plot means as horizontal lines with standard error ranges
    for i in range(len(dfs)):
        ax.axhline(y=means[i], color=colors[i],
                  linestyle='-', linewidth=2,
                  label=f'Mean: {means[i]:.2f}')
        ax.axhspan(means[i] - sems[i], means[i] + sems[i],
                  alpha=0.2, color=colors[i],
                  label=f'SE: {sems[i]:.2f}')

    # Add individual points with jitter
    for i, data in enumerate(all_data):
        x_jitter = np.random.normal(x_pos[i], 0.02, size=len(data))
        ax.scatter(x_jitter, data, alpha=0.3, color='black', s=20)

    # Customize plot
    ax.set_xticks(x_pos)
    if xlabels:
        ax.set_xticklabels(xlabels, ha='center')
    ax.set_ylabel(ylabel or f'Log {rt_column}', fontsize=DEFAULT_LABEL_SIZE)
    ax.set_title(title or f'{rt_column} Comparison', fontsize=DEFAULT_TITLE_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=DEFAULT_LABEL_SIZE)
    ax.grid(True, linestyle='--', alpha=0.7)
    #ax.legend(fontsize=DEFAULT_LEGEND_SIZE)

    # Calculate y-axis limits based on all data
    all_data_combined = np.concatenate(all_data)
    y_min, y_max = np.percentile(all_data_combined, [1, 99])
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    ax.set_xlim(0, 1)

    return ax


def plot_bar_rt_comparison_columns(
    df,
    rt_columns,
    ax=None,
    title=None,
    ylabel=None,
    xlabels=None,
    colors=None,
):
    """Plot comparison of reaction times between multiple columns in a DataFrame.
    
    Args:
        df: DataFrame containing reaction time data
        rt_columns: List of column names to analyze (e.g. ['first_rt', 'avg_rt'])
        ax: Optional matplotlib axis to plot on
        title: Custom title
        ylabel: Custom y-axis label
        xlabels: List of labels for x-axis ticks (defaults to column names)
        colors: List of colors for bars
    
    Returns:
        matplotlib axis
    """

    if colors is None:
        colors = [i for i in default_colors.values()]
        # Extend colors if needed
        while len(colors) < len(rt_columns):
            colors.extend(colors)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    # Calculate log RTs and group by user for each column
    all_data = []
    for col in rt_columns:
        log_col = f'log_{col}'
        stats = (df
                 .with_columns((1000*pl.col(col)).log().alias(log_col))
                 .group_by('user_id')
                 .agg(pl.col(log_col).mean()))
        all_data.append(stats[log_col].to_numpy())

    # Calculate means and standard errors
    means = [np.mean(data) for data in all_data]
    sems = [np.std(data) / np.sqrt(len(data)) for data in all_data]

    # Create bar plot
    x_pos = np.arange(len(rt_columns))
    bars = ax.bar(x_pos, means, yerr=sems, capsize=5,
                  color=colors[:len(rt_columns)])

    # Add individual points with jitter
    for i, data in enumerate(all_data):
        x_jitter = np.random.normal(i, 0.04, size=len(data))
        ax.scatter(x_jitter, data, alpha=0.3, color='black', s=20)

    # Customize plot
    ax.set_xticks(x_pos)
    if xlabels:
        ax.set_xticklabels(xlabels, ha='center')
    else:
        ax.set_xticklabels(rt_columns, ha='center')
    ax.set_ylabel(ylabel or 'Log Reaction Time', fontsize=DEFAULT_LABEL_SIZE)
    ax.set_title(title or 'Reaction Time Comparison', fontsize=DEFAULT_TITLE_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=DEFAULT_LABEL_SIZE)
    ax.grid(True, linestyle='--', alpha=0.7)

    return ax


def plot_rt_condition_differences(
    df: DataFrame,
    cond1_settings: dict,
    cond2_settings: dict,
    rt_columns: List[str],
    filter_columns: List[str] = None,
    ax=None,
    title=None,
    ylabel="Log RT Difference (Cond2 - Cond1)",
    xlabels=None,
    colors=None,
    remove_outliers: bool = True,
    outlier_threshold: float = 1.5,
    stats_file=None,
):
    """Plot reaction time differences between two conditions for multiple RT measures.
    
    Args:
        df: DataFrame containing the data
        cond1_settings: Filter settings for condition 1 
        cond2_settings: Filter settings for condition 2
        rt_columns: List of RT column names to analyze
        filter_columns: List of columns to use for outlier filtering
        ax: Optional matplotlib axis to plot on
        title: Custom title
        ylabel: Custom y-axis label
        xlabels: List of labels for x-axis ticks (defaults to rt_columns)
        colors: List of colors for bars
        remove_outliers: Whether to remove outliers using IQR method
        outlier_threshold: Number of IQRs to use for outlier detection (default: 1.5)
    
    Returns:
        matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    if colors is None:
        colors = default_colors.values()

    # Get data for each condition
    cond1_df = df.filter(**cond1_settings)
    cond2_df = df.filter(**cond2_settings)
    
    # Get common users between conditions
    users = set(cond1_df['user_id'].unique()) & set(cond2_df['user_id'].unique())
    
    # Dictionary to store valid differences for each RT column
    all_diffs = {}
    
    for rt_col in set(rt_columns + (filter_columns or [])):
        valid_diffs = []
        valid_users = []

        for user in users:
            user_cond1 = cond1_df.filter(user_id=user)
            user_cond2 = cond2_df.filter(user_id=user)
            # For each maze
            mazes1 = sorted(user_cond1['maze'].unique())
            mazes2 = sorted(user_cond2['maze'].unique())
            
            for maze1, maze2 in zip(mazes1, mazes2):
                maze_cond1 = user_cond1.filter(maze=maze1)
                maze_cond2 = user_cond2.filter(maze=maze2)
                
                if len(maze_cond1) == 0 or len(maze_cond2) == 0:
                    continue
                    
                # Calculate log RTs for this maze/condition pair
                rt1 = np.log(maze_cond1[rt_col].to_numpy())
                rt2 = np.log(maze_cond2[rt_col].to_numpy())
                
                # Calculate mean for each condition
                mean1 = np.mean(rt1)
                mean2 = np.mean(rt2)
                diff = mean2 - mean1
                
                valid_diffs.append(diff)
                valid_users.append(user)
        
        all_diffs[rt_col] = (np.array(valid_diffs), valid_users)
    
    # Apply outlier detection if requested
    if remove_outliers:
        shared_mask = np.ones(len(valid_users), dtype=bool)
        
        # Only use specified filter columns for outlier detection
        for rt_col in filter_columns or []:
            diffs, users = all_diffs[rt_col]
            q1 = np.percentile(diffs, 25)
            q3 = np.percentile(diffs, 75)
            iqr = q3 - q1
            lower_bound = q1 - outlier_threshold * iqr
            upper_bound = q3 + outlier_threshold * iqr
            
            col_mask = (diffs >= lower_bound) & (diffs <= upper_bound)
            shared_mask &= col_mask

            # Print outlier information
            if not col_mask.all():
                print(f"\nOutliers identified for {rt_col}:")
                print(f"Bounds: {lower_bound:.3f} to {upper_bound:.3f}")
                for i, (diff, user) in enumerate(zip(diffs, users)):
                    if not col_mask[i]:
                        print(f"User {user}, Maze diff: {diff:.3f}")

        if not shared_mask.all():
            print(f"\nTotal measurements removed: {(~shared_mask).sum()}")
            
        # Apply mask to all RT columns
        for rt_col in rt_columns:
            diffs, users = all_diffs[rt_col]
            all_diffs[rt_col] = (diffs[shared_mask], np.array(users)[shared_mask])

    # Calculate statistics and create plot
    means = []
    sems = []
    all_plot_diffs = []
    
    for rt_col in rt_columns:
        diffs, _ = all_diffs[rt_col]
        analyze_rt_differences(
            diffs, rt_col, stats_file=stats_file)
        all_plot_diffs.append(diffs)
        means.append(np.mean(diffs))
        sems.append(np.std(diffs) / np.sqrt(len(diffs)))

    # Create/get axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    # Create bar plot
    x_pos = np.arange(len(rt_columns))
    bars = ax.bar(x_pos, means, yerr=sems, capsize=5, color=colors)

    # Add individual points with jitter
    for i, diffs in enumerate(all_plot_diffs):
        x_jitter = np.random.normal(i, 0.1, size=len(diffs))
        ax.scatter(x_jitter, diffs, alpha=0.3, color='black', s=20)

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Customize plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xlabels or rt_columns, ha='center')
    ax.set_ylabel(ylabel, fontsize=DEFAULT_LABEL_SIZE)
    if title:
        ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=DEFAULT_LABEL_SIZE)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Set y-axis limits based on all data points
    all_data = np.concatenate(all_plot_diffs)
    y_min, y_max = np.percentile(all_data, [1, 99])
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    return fig, ax

######################################
# Power Analysis function
######################################

def analyze_proportion_test(data, mu=0.5, alpha=0.05, plot=False, stats_file=None):
    """Analyze proportion data with appropriate statistical tests and effect size.
    
    Args:
        data: numpy array of proportions (0-100 scale)
        mu: null hypothesis value (default: 0.5 * 100 = 50)
        alpha: significance level for tests (default: 0.05)
        plot: whether to show diagnostic plots (default: False)
        stats_file: optional file handle to write stats output
        
    Returns:
        dict containing test results and effect size
    """
    # Convert to 0-1 scale for analysis
    n = len(data)

    # Test normality using Shapiro-Wilk test
    _, normality_p = stats.shapiro(data)
    is_normal = normality_p > alpha

    # Calculate mean and standard error
    mean = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)

    # Perform appropriate statistical test
    if is_normal:
        # One-sided t-test
        t_stat, p_value = stats.ttest_1samp(data, mu)
        # Convert to one-sided p-value if t-statistic is in predicted direction
        p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2

        # Calculate Cohen's d effect size
        d = (mean - mu) / np.std(data, ddof=1)
        effect_size = {'name': "Cohen's d", 'value': d}

        test_name = "One-sample t-test"
        test_stat = t_stat

    else:
        # One-sided Wilcoxon signed-rank test
        w_stat, p_value = stats.wilcoxon(data - mu,
                                         alternative='greater')

        # Calculate r effect size (correlation coefficient) for Wilcoxon test
        # r = Z / sqrt(N) where Z is the standardized test statistic
        # Convert p-value to Z score using inverse normal CDF
        z = stats.norm.ppf(1 - p_value)  # One-sided p-value
        r = z / np.sqrt(n)  # Standardize by sample size
        effect_size = {'name': 'r', 'value': r}

        test_name = "Wilcoxon signed-rank test"
        test_stat = w_stat

    if plot:
        # Create diagnostic plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Histogram with density plot
        sns.histplot(data, kde=True, ax=ax1)
        ax1.axvline(mu, color='r', linestyle='--',
                    label=f'Null (μ={mu})')
        ax1.set_title('Distribution of Proportions')
        ax1.set_xlabel('Proportion')
        ax1.legend()

        # Q-Q plot
        stats.probplot(data, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot')

        plt.tight_layout()
        plt.show()

    # Prepare results
    results = {
        'n': n,
        'mean': mean,
        'se': se,
        'normality': {
            'is_normal': is_normal,
            'p_value': normality_p
        },
        'test': {
            'name': test_name,
            'statistic': test_stat,
            'p_value': p_value
        },
        'effect_size': effect_size
    }

    # Print summary
    summary = f"\nAnalysis Results (N={n}):\n"
    summary += f"Mean: {results['mean']:.2f}% (SE: {results['se']:.2f}%)\n"
    summary += f"\nNormality Test (Shapiro-Wilk):\n"
    summary += f"p = {normality_p:.3f} ({'Normal' if is_normal else 'Non-normal'} distribution)\n"
    summary += f"\n{test_name}:\n"
    summary += f"statistic = {test_stat:.3f}\n"
    summary += f"p = {p_value:.3f}\n"
    summary += f"\nEffect Size ({effect_size['name']}):\n"
    summary += f"{effect_size['value']:.3f}\n"
    

    # Calculate required sample size for different power levels
    power_levels = [0.8, 0.9, 0.95]
    summary += "\nRequired Sample Sizes:\n"

    if is_normal:
        # For t-test
        effect = effect_size['value']  # Cohen's d
        for power in power_levels:
            analysis = TTestPower()
            n_required = analysis.solve_power(
                effect_size=effect,
                alpha=0.05,
                power=power,
                alternative='larger'  # one-sided test
            )
            summary += f"Power {power*100:g}%: N ≥ {ceil(n_required)} participants\n"
    else:
        # For Wilcoxon test (based on asymptotic relative efficiency)
        # Wilcoxon test is ~95% as efficient as t-test
        effect = effect_size['value']  # r score
        # Convert r to d using formula: d = 2r/sqrt(1-r^2)
        d = 2 * effect / sqrt(1 - effect**2) if abs(effect) < 1 else float('inf')
        for power in power_levels:
            analysis = TTestPower()
            n_required = analysis.solve_power(
                effect_size=d,
                alpha=0.05,
                power=power,
                alternative='larger'  # one-sided test
            )
            # Adjust for Wilcoxon efficiency
            n_required = ceil(n_required / 0.95)
            summary += f"Power {power*100:g}%: N ≥ {n_required} participants\n"

    if stats_file:
        stats_file.write(summary)

    return results


def power_analysis_between_groups(data1, data2, alpha=0.05, power=0.8, stats_file=None):
    """Perform power analysis for between-groups comparison and calculate required sample size.
    
    Args:
        data1: Array of measurements from group 1
        data2: Array of measurements from group 2
        alpha: Significance level (default: 0.05)
        power: Desired statistical power (default: 0.8)
        stats_file: Optional file handle to write stats output
    
    Returns:
        dict containing:
            - normality: Results of normality tests
            - effect_size: Effect size (Cohen's d or r)
            - n_required: Required sample size per group
            - actual_power: Actual power achieved with current sample size
            - test_results: Results of statistical test on current data
    """
    # Test for normality in both groups
    _, p1 = stats.shapiro(data1)
    _, p2 = stats.shapiro(data2)
    is_normal = (p1 > alpha) and (p2 > alpha)

    n1, n2 = len(data1), len(data2)
    mean1, mean2 = np.mean(data1), np.mean(data2)

    if is_normal:
        # Use t-test and Cohen's d for normal data
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)

        # Pooled standard deviation
        pooled_sd = np.sqrt(
            ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # Cohen's d
        d = (mean1 - mean2) / pooled_sd
        effect_size = {'name': "Cohen's d", 'value': d}

        # Perform power analysis
        analysis = TTestIndPower()

        # Calculate required sample size
        n_required = analysis.solve_power(
            effect_size=abs(d),
            alpha=alpha,
            power=power,
            ratio=1.0,
            alternative='larger'
        )

        # Calculate actual power with current sample size
        actual_power = analysis.power(
            effect_size=abs(d),
            nobs1=min(n1, n2),  # Changed from nobs to nobs1
            alpha=alpha,
            ratio=1.0,
            alternative='larger'
        ) * 0.95  # Adjust for efficiency

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(data1, data2, alternative='greater')
        test_name = "Independent t-test"
        test_stat = t_stat

    else:
        # Use Mann-Whitney U test and r effect size for non-normal data
        u_stat, p_value = stats.mannwhitneyu(
            data1, data2, alternative='greater')
        test_name = "Mann-Whitney U test"
        test_stat = u_stat

        # Calculate r effect size for Mann-Whitney U test
        # r = Z / sqrt(N)
        z = stats.norm.ppf(1 - p_value)  # Convert p-value to Z score
        r = z / np.sqrt(n1 + n2)
        effect_size = {'name': 'r', 'value': r}

        # For non-normal data, convert r to approximate d for power analysis
        # d = 2r/sqrt(1-r^2)
        d = 2 * r / np.sqrt(1 - r**2) if abs(r) < 1 else float('inf')

        # Use t-test power analysis as approximation
        analysis = TTestIndPower()
        n_required = ceil(analysis.solve_power(
            effect_size=abs(d),
            alpha=alpha,
            power=power,
            ratio=1.0,
            alternative='larger'
        ) / 0.95)  # Adjust for ~95% efficiency of Mann-Whitney vs t-test

        actual_power = analysis.power(
            effect_size=abs(d),
            nobs1=min(n1, n2),  # Changed from nobs to nobs1
            alpha=alpha,
            ratio=1.0,
            alternative='larger'
        ) * 0.95  # Adjust for efficiency

    results = {
        'normality': {
            'is_normal': is_normal,
            'p_values': (p1, p2)
        },
        'effect_size': effect_size,
        'n_required': ceil(n_required),
        'actual_power': actual_power,
        'test_results': {
            'name': test_name,
            'statistic': test_stat,
            'p_value': p_value,
            'n1': n1,
            'n2': n2
        }
    }

    # Write results to stats file if provided
    if stats_file:
        stats_file.write("\nPower Analysis Results:\n")
        stats_file.write(f"Normality test p-values: {results['normality']['p_values']}\n")
        stats_file.write(f"Using {results['test_results']['name']} ({'normal' if results['normality']['is_normal'] else 'non-normal'} distribution)\n")
        stats_file.write(f"\nEffect size ({results['effect_size']['name']}): {results['effect_size']['value']:.3f}\n")
        stats_file.write(f"Current sample sizes: n1={results['test_results']['n1']}, n2={results['test_results']['n2']}\n")
        stats_file.write(f"Required sample size per group (80% power): {results['n_required']}\n")
        stats_file.write(f"Actual power with current sample size: {results['actual_power']:.3f}\n")
        stats_file.write("\nCurrent Data Analysis:\n")
        stats_file.write(f"Test statistic: {results['test_results']['statistic']:.3f}\n")
        stats_file.write(f"p-value: {results['test_results']['p_value']:.3f}\n")

    return results


def analyze_rt_differences(differences, rt_col, alpha=0.05, stats_file=None):
    """Analyze RT differences between conditions with appropriate statistical tests.
    
    Args:
        differences: numpy array of RT differences (cond2 - cond1)
        rt_col: Name of RT column being analyzed
        alpha: significance level for tests (default: 0.05)
        stats_file: optional file handle to write stats output
        
    Returns:
        dict containing test results and effect size
    """
    n = len(differences)

    # Test normality using Shapiro-Wilk test
    _, normality_p = stats.shapiro(differences)
    is_normal = normality_p > alpha

    # Calculate mean and standard error
    mean = np.mean(differences)
    se = np.std(differences, ddof=1) / np.sqrt(n)

    if stats_file:
        stats_file.write(f"\nRT Analysis for {rt_col}:\n")
        stats_file.write("-" * (len(rt_col) + 15) + "\n")
        stats_file.write(f"N = {n} participants\n")
        stats_file.write(f"Mean difference = {mean:.3f} (SE: {se:.3f})\n\n")
        stats_file.write("Normality Test (Shapiro-Wilk):\n")
        stats_file.write(
            f"p = {normality_p:.3f} ({'Normal' if is_normal else 'Non-normal'} distribution)\n\n")

    if is_normal:
        # One-sided paired t-test (testing if condition 1 < condition 2)
        t_stat, p_value = stats.ttest_rel(
            differences, np.zeros_like(differences))
        # Convert to one-sided p-value if t-statistic is in predicted direction
        p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2

        # Calculate Cohen's d effect size for paired differences
        d = mean / np.std(differences, ddof=1)
        effect_size = {'name': "Cohen's d", 'value': d}

        test_name = "Paired t-test"
        test_stat = t_stat

    else:
        # One-sided Wilcoxon signed-rank test
        w_stat, p_value = stats.wilcoxon(differences, alternative='greater')

        # Calculate r effect size for Wilcoxon test
        z = stats.zscore(differences)
        r = np.abs(np.mean(z)) / np.sqrt(n)
        effect_size = {'name': 'r', 'value': r}

        test_name = "Wilcoxon signed-rank test"
        test_stat = w_stat

    if stats_file:
        stats_file.write(f"{test_name}:\n")
        stats_file.write(f"statistic = {test_stat:.3f}\n")
        stats_file.write(f"p = {p_value:.3f}\n\n")
        stats_file.write(f"Effect Size ({effect_size['name']}):\n")
        stats_file.write(f"{effect_size['value']:.3f}\n")
        stats_file.write("\n" + "="*50 + "\n")

    return {
        'n': n,
        'mean': mean,
        'se': se,
        'normality': {'is_normal': is_normal, 'p_value': normality_p},
        'test': {'name': test_name, 'statistic': test_stat, 'p_value': p_value},
        'effect_size': effect_size
    }

######################################
# Model Analysis
######################################

def episode_q_values(e, reduce: bool = False):
    actions = e.actions
    preds = e.transitions.extras['preds']
    q_values = preds.q_vals  # [T, A]
    actions = e.actions  # [T]

    if reduce:
      q_values = jnp.take_along_axis(q_values, actions[:, None], axis=-1)  # [T, 1]
      q_values = jnp.squeeze(q_values, axis=-1)  # [T]

    in_episode = get_in_episode(e.timesteps)
    q_values = q_values[in_episode]
    # [T', ... ]
    return q_values

def episode_sf_value(e, idx=None):
    actions = e.actions
    preds = e.transitions.extras['preds']
    sf_values = preds.sf  # [T, N, A, W]
    actions = e.actions  # [T]

    sf_values = jnp.take_along_axis(
        sf_values, actions[:, None, None, None], axis=-2)

    sf_values = jnp.squeeze(sf_values, axis=-2)  # [T, N, W]

    in_episode = get_in_episode(e.timesteps)
    sf_values = sf_values[in_episode]
    # [T', ... ]
    if idx is not None:
        sf_values = sf_values[:, idx]
    return sf_values

def plot_sf_values(e, idxs=None, line_mask=None, line_names=None, figsize=None, colors=None, styles=None, plot_q_values=True):
    """Plot successor feature values as lines in multiple panels.
    
    Args:
        e: Episode data
        idxs: List of indices for SF values to plot in separate panels. If None, plots all indices
        line_mask: Optional boolean mask of length N to filter which lines to plot
        line_names: Optional list of names for each line
        figsize: Optional figure size tuple (width, height)
        colors: Optional list of colors for each line pair
        styles: Optional list of linestyles for first/second half
        plot_q_values: Boolean to determine whether to plot Q-values (default: True)
    
    Returns:
        fig: matplotlib figure object
        axs: array of matplotlib axis objects
    """
    # Get all indices if none specified
    all_sf_values = episode_sf_value(e)  # Get full SF values to determine shape
    if idxs is None:
        idxs = list(range(all_sf_values.shape[1]))  # Use all available indices
        
    # Calculate figure size based on number of panels
    if figsize is None:
        figsize = (7 * len(idxs), 5)
        
    fig, axs = plt.subplots(1, len(idxs), figsize=figsize)
    if len(idxs) == 1:
        axs = [axs]  # Make iterable for single panel case

    line_mask = line_mask or [
        True, True, False, False,
        True, True, False, False
    ]

    line_names = line_names or [
        'main', 'off-task', 'main2', 'off-task2',
        'near main', 'near off-task', 'near-main2', 'near-off-task2'
    ]
    # Get first half of line names and take every even index (0, 2)
    first_half = line_names[:len(line_names)//2]  # ['main', 'off-task', 'main2', 'off-task2']
    policy_names = first_half[::2]  # ['main', 'main2']
    
    colors = colors or ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    styles = styles or ['-', '--']

    task_w = e.timesteps.observation.task_w
    in_episode = get_in_episode(e.timesteps)
    task_w = task_w[in_episode]
    max_value = -1000
    for panel_idx, idx in enumerate(idxs):
        sf_values = all_sf_values[:, idx]
        q_value = (sf_values*task_w).sum(-1)
        max_value = max(max_value, q_value.max(), sf_values.max())
        ax = axs[panel_idx]
        
        time_steps = np.arange(sf_values.shape[0])
        n_total = sf_values.shape[1]
        n_half = n_total // 2

        for i in range(sf_values.shape[1]):
            if line_mask is not None and not line_mask[i]:
                continue

            color_idx = i % n_half
            style_idx = i // n_half

            # Only show legend in first panel
            label = line_names[i] if line_names and i < len(line_names) and panel_idx == 0 else None
            ax.plot(time_steps, sf_values[:, i],
                    label=label,
                    color=colors[color_idx],
                    linestyle=styles[style_idx])

        # Add Q-value plot if plot_q_values is True
        if plot_q_values:
            ax.plot(time_steps, q_value, label='Q-value' if panel_idx == 0 else None, color='k', linestyle='-')

        if len(idxs) > 1:
          ax.set_title(f'Successor Feature Predictions (task={policy_names[idx]})', fontsize=DEFAULT_TITLE_SIZE)
        else:
           ax.set_title(f'Successor Feature Predictions', fontsize=DEFAULT_TITLE_SIZE)
        ax.set_xlabel('Time Step', fontsize=DEFAULT_LABEL_SIZE)
        ax.set_ylabel('Value', fontsize=DEFAULT_LABEL_SIZE)
        ax.set_xlim(0, sf_values.shape[0] - 1)
        ax.set_ylim(0, 1.1*max_value)

    # Only show legend in first panel
    if line_names is not None:
        axs[0].legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    return fig, axs

################################################################################
# Exp 1
################################################################################
def experiment_1_results(
      user_df: DataFrame,
      model_df: DataFrame,
      save_dir: str,
      display_figs: bool = False,
      save_figs: bool = True,
      ):
  """_summary_

  1. Filter out users with less than 16 successes during training

    Args:
      user_df (DataFrame): _description_
      model_df (DataFrame): _description_
  """
  save_dir = os.path.join(save_dir, 'exp1')
  os.makedirs(save_dir, exist_ok=True)
  
  # Open stats file
  stats_file = open(os.path.join(save_dir, 'stats.txt'), 'w')
  stats_file.write("Experiment 1 Statistical Analysis\n\n")

  ##################
  # Get relevant simulations
  ##################
  mdf = model_df.filter(maze='big_m3_maze1', eval=True)

  ##################
  # get all episodes for users who achieved at least 16 successes during training
  ##################
  exp1_eval_df = user_df.filter_by_group(
      input_episode_filter=filter_train_by_min_success,
      input_settings=dict(eval=False),
      output_settings=dict(manipulation=3),
      group_key='user_id',
  ).filter(eval=True)

  ##################
  # filter outliers based on episode path length and max reaction time
  ##################
  # TODO. QUESTION: should I separately filter out participants that reused the training path vs. though that took a new path?
  # Check if reuse column is string type
  exp1_eval_df = filter_outliers(
      exp1_eval_df,
      filter_columns=['path_length', 'avg_rt'],
  )
  ##################
  # Create example paths
  ##################
  # Convert reuse column from string to boolean
  if exp1_eval_df.schema['reuse'] == pl.String:
      exp1_eval_df = exp1_eval_df.with_columns(
          pl.col('reuse') == 'true'
      )
  elif exp1_eval_df.schema['reuse'] == pl.Boolean:
      pass
  else:
      raise ValueError("Reuse column is type: ", exp1_eval_df.schema['reuse'])
  
  old_path_cond = exp1_eval_df.filter(maze='big_m3_maze1_(F,F)', reuse=True)
  new_path_cond = exp1_eval_df.filter(maze='big_m3_maze1_(F,F)', reuse=False)

  fig, ax = plt.subplots(figsize=(6, 6))
  render_paths(
      episode_list=[old_path_cond.episodes[0], new_path_cond.episodes[0]],
      colors=[model_colors['dynaq_shared'], model_colors['dfs']],
      ax=ax)
  if save_figs:
      fig.savefig(
         os.path.join(save_dir, 'exp1_1_example_paths.pdf'),
                  bbox_inches='tight')
  if display_figs:
    plt.show()


  ##################
  # Create success rate and path reuse plots
  ##################
  # compute the mean by user
  # Get mean success rate per user
  human_successes = exp1_eval_df.group_by('user_id').agg(
      pl.col('success').mean()).select('success').to_numpy().flatten()
  
  # put in (0, 100) range
  human_successes = 100*human_successes
  

  # Calculate mean and standard error across users
  human_mean = np.mean(human_successes)
  human_se = np.std(human_successes) / np.sqrt(len(human_successes))

  model_stats = (
     mdf.group_by('algo')
                 .agg(
      mean=pl.col('success').mean() * 100,
      se=(pl.col('success').std() /
          pl.col('success').count().sqrt()) * 100
  ))

  fig, ax = plt.subplots(figsize=(6, 3))
  bar_plot_error(
      human_data=dict(
          #raw=human_successes,
          means=human_mean,
          se=human_se),
      model_stats=model_stats,
      ax=ax,
      legend=False,
  )
  ax.set_title('Exp 1 Generalization Success Rate', fontsize=DEFAULT_TITLE_SIZE)
  ax.set_ylabel('Success Rate (%)', fontsize=DEFAULT_LABEL_SIZE)
  if save_figs:
      fig.savefig(
          os.path.join(save_dir, 'exp1_2_success_rate.pdf'),
          bbox_inches='tight')

  if display_figs:
    plt.show()

  ##################
  # Create path re-use analysis plots
  ##################
  human_reuse = exp1_eval_df.group_by('user_id').agg(
      pl.col('reuse').mean()).select('reuse').to_numpy().flatten()
  stats_file.write("Path Reuse Analysis\n")
  stats_file.write("--------------------\n")
  reuse_results = analyze_proportion_test(human_reuse, mu=0.5, alpha=0.05,
                          plot=False, stats_file=stats_file)
  # put in (0, 100) range
  human_reuse = 100*human_reuse

  # Calculate mean and standard error across users
  human_mean = np.mean(human_reuse)
  human_se = np.std(human_reuse) / np.sqrt(len(human_reuse))

  model_stats = (
      mdf.group_by('algo')
      .agg(
          mean=pl.col('reuse').mean() * 100,
          se=(pl.col('reuse').std() /
              pl.col('reuse').count().sqrt()) * 100
      ))

  fig, ax = plt.subplots(figsize=(6, 3))
  bar_plot_error(
      human_data=dict(
          raw=human_reuse,
          means=human_mean,
          se=human_se),
      model_stats=model_stats,
      ax=ax,
  )
  ax.set_title('Exp 1 Path Reuse', fontsize=DEFAULT_TITLE_SIZE)
  ax.set_ylabel('Path Reuse (%)', fontsize=DEFAULT_LABEL_SIZE)
  if save_figs:
      fig.savefig(
          os.path.join(save_dir, 'exp1_3_path_reuse.pdf'),
          bbox_inches='tight')

  if display_figs:
    plt.show()

  ######################
  # Plot reaction times when using new path vs. partial reuse
  ######################
  # two plots, one for first RT and one for avg RT
  # 1. first compute log RT values
  # 2. then get the average per user
  # 3. then for each value, plot the mean and standard error across users
  # 4. finally, add the per-use mean values
  reuse_episodes = exp1_eval_df.filter(reuse=True)
  no_reuse_episodes = exp1_eval_df.filter(reuse=False)

  stats_file.write("\n\nReaction Time Analysis\n")
  stats_file.write("--------------------\n")
  # First RT bar comparison
  fig, ax = plt.subplots(figsize=(3, 3))
  plot_bar_rt_comparison(
      [no_reuse_episodes, reuse_episodes],
      'first_rt',
      title='First Reaction Time',
      ylabel='log milliseconds',
      xlabels=['New Path', 'Partial Reuse'],
      colors=[default_colors['nice purple'], default_colors['bluish green']],
      stats_file=stats_file,
      ax=ax)
  if save_figs:
    fig.savefig(
        os.path.join(save_dir, 'exp1_4_bar_first_rt.pdf'),
        bbox_inches='tight')
  if display_figs:
    plt.show()

  # Average RT bar comparison  
  fig, ax = plt.subplots(figsize=(3, 3))
  plot_bar_rt_comparison(
      [no_reuse_episodes, reuse_episodes],
      'avg_rt',
      title='Average Reaction Time',
      ylabel='log avg(milliseconds/step)',
      xlabels=['New Path', 'Partial Reuse'],
      colors=[default_colors['nice purple'], default_colors['bluish green']],
      ax=ax)
  if save_figs:
    fig.savefig(
        os.path.join(save_dir, 'exp1_4_bar_avg_rt.pdf'),
        bbox_inches='tight')
  if display_figs:
    plt.show()

  # First RT mean comparison
  fig, ax = plt.subplots(figsize=(3, 3))
  plot_mean_rt_comparison(
      [no_reuse_episodes, reuse_episodes],
      'first_rt',
      title='First Reaction Time',
      ylabel='log milliseconds',
      xlabels=['New Path', 'Partial Reuse'],
      colors=[default_colors['nice purple'], default_colors['bluish green']],
      ax=ax)
  if save_figs:
    fig.savefig(
        os.path.join(save_dir, 'exp1_4_mean_first_rt.pdf'),
        bbox_inches='tight')
  if display_figs:
    plt.show()

  # Average RT mean comparison
  fig, ax = plt.subplots(figsize=(3, 3))
  plot_mean_rt_comparison(
      [no_reuse_episodes, reuse_episodes],
      'avg_rt',
      title='Average Reaction Time',
      ylabel='log avg(milliseconds/step)',
      xlabels=['New Path', 'Partial Reuse'],
      colors=[default_colors['nice purple'], default_colors['bluish green']],
      ax=ax)
  if save_figs:
    fig.savefig(
        os.path.join(save_dir, 'exp1_4_mean_avg_rt.pdf'),
        bbox_inches='tight')
  if display_figs:
    plt.show()

  ######################
  # When do partial reuse, plot first, max post, avg post
  ######################
  fig, ax = plt.subplots(1, 1, figsize=(6, 3))
  colors = ["bluish green", "reddish purple", "yellow", "orange"]
  plot_bar_rt_comparison_columns(
      reuse_episodes,
      ['first_rt', 'max_init_post_rt', 'max_final_rt', 'avg_post_rt'],
      title='Reaction time measurements over episode',
      ylabel='log milliseconds',
      xlabels=['First', 'Max 1st ½', 'Max 2nd ½', 'Avg Post'],
      colors=[default_colors[c] for c in colors],
      ax=ax)
  if save_figs:
    fig.savefig(
        os.path.join(save_dir, 'exp1_5_bar_reuse_rt_comparison.pdf'),
        bbox_inches='tight')
  if display_figs:
    plt.show()
  

  ######################
  # SF Model
  ######################
  sf_episodes = model_df.filter(maze='big_m3_maze1', eval=False, algo='usfa')
  fig, ax = plot_sf_values(
     sf_episodes.episodes[0],
     plot_q_values=False,
     figsize=(5,4),
     idxs=[0])
  if save_figs:
    fig.savefig(
        os.path.join(save_dir, 'exp1_6_sf_predictions.pdf'),
        bbox_inches='tight')
  if display_figs:
    plt.show()

  # Close stats file at the end
  stats_file.close()
  with open(os.path.join(save_dir, 'stats.txt'), 'r') as f:
    print(f.read())


def experiment_2_results(
    user_df: DataFrame,
    model_df: DataFrame,
    save_dir: str,
    display_figs: bool = False,
    save_figs: bool = True,
):
  """_summary_

  1. Filter out users with less than 16 successes during training

    Args:
      user_df (DataFrame): _description_
      model_df (DataFrame): _description_
  """
  save_dir = os.path.join(save_dir, 'exp2')
  os.makedirs(save_dir, exist_ok=True)

  # Open stats file
  stats_file = open(os.path.join(save_dir, 'stats.txt'), 'w')
  stats_file.write("Experiment 2 Statistical Analysis\n")
  stats_file.write("===============================\n\n")

  ##################
  # Get relevant simulations
  ##################
  mdf = model_df.filter(maze='big_m1_maze3_shortcut', eval=True)

  ##################
  # get all episodes for users who achieved at least 16 successes during training
  ##################
  exp2_eval_df = user_df.filter_by_group(
      input_episode_filter=filter_train_by_min_success,
      input_settings=dict(eval=False),
      output_settings=dict(manipulation=1),
      group_key='user_id',
  ).filter(eval=True)

  ##################
  # filter outliers based on episode path length and max reaction time
  ##################
  # TODO. QUESTION: should I separately filter out participants that reused the training path vs. though that took a new path?
  # Check if reuse column is string type
  exp2_eval_df = filter_outliers(
      exp2_eval_df,
      filter_columns=['path_length', 'avg_rt'],
  )
  ##################
  # Create example paths
  ##################
  # Convert reuse column from string to boolean
  if exp2_eval_df.schema['reuse'] == pl.String:
      exp2_eval_df = exp2_eval_df.with_columns(
          pl.col('reuse') == 'true'
      )
  elif exp2_eval_df.schema['reuse'] == pl.Boolean:
      pass
  else:
      raise ValueError("Reuse column is type: ", exp2_eval_df.schema['reuse'])

  old_path_cond = user_df.filter(maze='big_m1_maze3_(F,F)', room=0).sort('path_length', descending=True)
  new_path_cond = exp2_eval_df.filter(maze='big_m1_maze3_shortcut_(F,F)', reuse=False)

  fig, ax = plt.subplots(figsize=(6, 6))
  render_paths(
      episode_list=[new_path_cond.episodes[0], old_path_cond.episodes[0]],
      colors=[model_colors['dfs'], model_colors['dynaq_shared']],
      ax=ax)
  if save_figs:
      fig.savefig(
          os.path.join(save_dir, 'exp2_1_example_paths.pdf'),
          bbox_inches='tight')
  if display_figs:
    plt.show()

  ##################
  # Create success rate and path reuse plots
  ##################
  # compute the mean by user
  # Get mean success rate per user
  human_successes = exp2_eval_df.group_by('user_id').agg(
      pl.col('success').mean()).select('success').to_numpy().flatten()
  # put in (0, 100) range
  human_successes = 100*human_successes

  # Calculate mean and standard error across users
  human_mean = np.mean(human_successes)
  human_se = np.std(human_successes) / np.sqrt(len(human_successes))

  model_stats = (
      mdf.group_by('algo')
      .agg(
          mean=pl.col('success').mean() * 100,
          se=(pl.col('success').std() /
              pl.col('success').count().sqrt()) * 100
      )
  )

  fig, ax = plt.subplots(figsize=(6, 3))
  bar_plot_error(
      human_data=dict(
          # raw=human_successes,
          means=human_mean,
          se=human_se),
      model_stats=model_stats,
      ax=ax,
      legend=False,
  )
  ax.set_title('Exp 2 Generalization Success Rate', fontsize=DEFAULT_TITLE_SIZE)
  ax.set_ylabel('Success Rate (%)', fontsize=DEFAULT_LABEL_SIZE)
  if save_figs:
      fig.savefig(
          os.path.join(save_dir, 'exp2_2_success_rate.pdf'),
          bbox_inches='tight')

  if display_figs:
    plt.show()

  ##################
  # Create path re-use analysis plots
  ##################
  human_reuse = exp2_eval_df.group_by('user_id').agg(
      pl.col('reuse').mean()).select('reuse').to_numpy().flatten()
  # put in (0, 100) range
  human_reuse = 100*human_reuse

  stats_file.write("\nPath Reuse Analysis\n")
  stats_file.write("------------------\n")
  analyze_proportion_test(human_reuse, mu=0.5, alpha=0.05, plot=False, stats_file=stats_file)

  # Calculate mean and standard error across users
  human_mean = np.mean(human_reuse)
  human_se = np.std(human_reuse) / np.sqrt(len(human_reuse))

  model_stats = (
      mdf.group_by('algo')
      .agg(
          mean=pl.col('reuse').mean() * 100,
          se=(pl.col('reuse').std() /
              pl.col('reuse').count().sqrt()) * 100
      )
  )

  fig, ax = plt.subplots(figsize=(6, 3))
  bar_plot_error(
      human_data=dict(
          raw=human_reuse,
          means=human_mean,
          se=human_se),
      model_stats=model_stats,
      ax=ax,
  )
  ax.set_title('Exp 2 Partial Path Reuse', fontsize=DEFAULT_TITLE_SIZE)
  ax.set_ylabel('Path Reuse (%)', fontsize=DEFAULT_LABEL_SIZE)
  if save_figs:
      fig.savefig(
          os.path.join(save_dir, 'exp2_3_path_reuse.pdf'),
          bbox_inches='tight')

  if display_figs:
    plt.show()

  # Close stats file at the end
  stats_file.close()
  with open(os.path.join(save_dir, 'stats.txt'), 'r') as f:
    print(f.read())


def experiment_3_results(
    user_df: DataFrame,
    model_df: DataFrame,
    save_dir: str,
    filter_columns: List[str] = None,
    display_figs: bool = False,
    save_figs: bool = True,
):
  """_summary_

  1. Filter out users with less than 16 successes during training

    Args:
      user_df (DataFrame): _description_
      model_df (DataFrame): _description_
      save_dir (str): Directory to save figures
      filter_columns (List[str], optional): Columns to use for outlier filtering in RT analysis.
          Defaults to ['avg_rt'].
      display_figs (bool, optional): Whether to display figures. Defaults to False.
      save_figs (bool, optional): Whether to save figures. Defaults to True.
  """
  save_dir = os.path.join(save_dir, 'exp3')
  os.makedirs(save_dir, exist_ok=True)
  # Default to ['avg_rt'] if no filter columns specified
  filter_columns = filter_columns or ['avg_rt']

  stats_file = open(os.path.join(save_dir, 'stats.txt'), 'w')
  stats_file.write("Experiment 3 Statistical Analysis\n")
  stats_file.write("===============================\n\n")

  ##################
  # get all episodes for users who achieved at least 16 successes during training
  ##################
  exp3_eval_df = user_df.filter_by_group(
      input_episode_filter=filter_train_by_min_success,
      input_settings=dict(eval=False),
      output_settings=dict(manipulation=2),
      group_key='user_id',
  ).filter(eval=True)

  ##################
  # filter outliers based on episode path length and max reaction time
  ##################
  # TODO. QUESTION: should I separately filter out participants that reused the training path vs. though that took a new path?
  # Check if reuse column is string type
  exp3_eval_df = filter_outliers(
      exp3_eval_df,
      filter_columns=['path_length'],
  )
  ##################
  # Create example paths
  ##################
  # Convert reuse column from string to boolean
  if exp3_eval_df.schema['reuse'] == pl.String:
      exp3_eval_df = exp3_eval_df.with_columns(
          pl.col('reuse') == 'true'
      )
  elif exp3_eval_df.schema['reuse'] == pl.Boolean:
      pass
  else:
      raise ValueError("Reuse column is type: ", exp3_eval_df.schema['reuse'])

  old_path_cond = exp3_eval_df.filter(
      maze='big_m2_maze2_onpath_(F,F)')
  new_path_cond = exp3_eval_df.filter(
      maze='big_m2_maze2_offpath_(F,F)')

  fig, ax = plt.subplots(figsize=(6, 6))
  render_paths(
      episode_list=[old_path_cond.episodes[0], new_path_cond.episodes[0]],
      render_both=True,
      colors=[model_colors['dynaq_shared'], model_colors['dfs']],
      ax=ax)
  if save_figs:
      fig.savefig(
          os.path.join(save_dir, 'exp3_1_example_paths.pdf'),
          bbox_inches='tight')
  if display_figs:
    plt.show()

  # Create filter string for filename
  filter_str = ','.join(filter_columns)

  fig, ax = plot_rt_condition_differences(
      df=exp3_eval_df,
      cond1_settings=dict(manipulation=2, eval=True, condition=1),
      cond2_settings=dict(manipulation=2, eval=True, condition=2),
      rt_columns=['first_rt', 'avg_rt'],
      filter_columns=filter_columns,
      colors=[default_colors['google blue'], default_colors["google orange"]],
      title="Exp 3 Reaction Time Difference",
      ylabel="log milliseconds",
      xlabels=['First', 'Average'],
      stats_file=stats_file,
      ax=ax
  )
  if save_figs:
      fig.savefig(
          os.path.join(save_dir, f'exp3_2_rt_diff_filter_{filter_str}.pdf'),
          bbox_inches='tight')
  if display_figs:
    plt.show()
  stats_file.close()
  with open(os.path.join(save_dir, 'stats.txt'), 'r') as f:
    print(f.read())

def experiment_4_results(
    user_df: DataFrame,
    model_df: DataFrame,
    save_dir: str,
    filter_columns: List[str] = None,
    display_figs: bool = False,
    save_figs: bool = True,
):
  """Analyze results from experiment 4.

  Args:
      user_df (DataFrame): DataFrame containing user data
      model_df (DataFrame): DataFrame containing model data 
      save_dir (str): Directory to save figures
      filter_columns (List[str], optional): Columns to use for outlier filtering in RT analysis. 
          Defaults to ['avg_rt'].
      display_figs (bool, optional): Whether to display figures. Defaults to False.
      save_figs (bool, optional): Whether to save figures. Defaults to True.
  """
  save_dir = os.path.join(save_dir, 'exp4')
  os.makedirs(save_dir, exist_ok=True)
  # Default to ['avg_rt'] if no filter columns specified
  filter_columns = filter_columns or ['avg_rt']

  def get_eval_df(setting: str, version: str):
    if version == 'regular':
        same_cond = dict(
            name=f'big_m4_maze_{setting}')
    elif version == 'blind':
        same_cond = dict(
            name=f'big_m4_maze_{setting}_blind')

    ##################
    # get all episodes for users who achieved at least 16 successes during training
    ##################
    exp4_eval_df = user_df.filter_by_group(
        input_episode_filter=partial(filter_train_by_min_success, min_successes=8*4),
        input_settings=dict(eval=False, **same_cond),
        output_settings=dict(manipulation=4),
        group_key='user_id',
    )

    ##################
    # filter outliers based on episode path length and max reaction time
    ##################
    # TODO. QUESTION: should I separately filter out participants that reused the training path vs. though that took a new path?
    # Check if reuse column is string type
    exp4_eval_df = filter_outliers(
        exp4_eval_df,
        filter_columns=['path_length'],
    )
    return exp4_eval_df
  ##################
  # Create example paths
  ##################
  def plot_example(setting: str='short', version: str = 'regular'):
    assert setting in ['short', 'long']
    assert version in ['blind', 'regular']
    exp4_eval_df = get_eval_df(setting, version)
    if version == 'regular':
        cond0 = exp4_eval_df.filter(maze=f'big_m4_maze_{setting}_(F,F)').sort('path_length', descending=True)
        #user = cond0['user_id'].unique()[0]
        #cond1 = exp4_eval_df.filter(maze=f'big_m4_maze_{setting}_eval_same_(F,F)', user_id=user)
        #cond2 = exp4_eval_df.filter(maze=f'big_m4_maze_{setting}_eval_diff_(F,F)', user_id=user)
    elif version == 'blind':
        cond0 = exp4_eval_df.filter(maze=f'big_m4_maze_{setting}_blind_(F,F)').sort('path_length', descending=True)
        #user = cond0['user_id'].unique()[0]
        #cond1 = exp4_eval_df.filter(
        #    maze=f'big_m4_maze_{setting}_eval_same_blind_(F,F)', user_id=user)
        #cond2 = exp4_eval_df.filter(
        #    maze=f'big_m4_maze_{setting}_eval_diff_(F,F)', user_id=user)

    fig, ax = plt.subplots(figsize=(6, 6))
    render_paths(
        episode_list=[cond0.episodes[0]],
        #render_both=True,
        colors=[default_colors['white']],
        ax=ax)
    if save_figs:
        fig.savefig(
            os.path.join(save_dir, f'exp4_1_example_paths_{setting}_{version}.pdf'),
            bbox_inches='tight')
    if display_figs:
      plt.show()

  #plot_example(setting='short', version='regular')
  #plot_example(setting='short', version='blind')
  #plot_example(setting='long', version='regular')

  def plot_rt_diff(setting: str = 'short', version: str = 'regular', ax=None):
    assert setting in ['short', 'long']
    assert version in ['blind', 'regular']
    exp4_eval_df = get_eval_df(setting, version)
    if version == 'regular':
        tell_reuse: int = 1
        same_cond = dict(name=f'big_m4_maze_{setting}_eval_same', tell_reuse=tell_reuse)
    elif version == 'blind':
        tell_reuse: int = 0
        same_cond = dict(
            name=f'big_m4_maze_{setting}_eval_same_blind', tell_reuse=tell_reuse)

    # LARGER
    diff_cond = dict(name=f'big_m4_maze_{setting}_eval_diff', tell_reuse=tell_reuse)
    label = dict(
       short='Near',
       long='Far'
    )[setting]
    v = dict(
       regular='Known',
       blind='Unknown'
    )[version]

    # Create filter string for filename
    filter_str = ','.join(filter_columns)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    
    plot_rt_condition_differences(
        df=exp4_eval_df,
        cond1_settings=same_cond,
        cond2_settings=diff_cond,
        rt_columns=['first_rt', 'avg_rt'],
        filter_columns=filter_columns,
        colors=[default_colors['google blue'], default_colors["google orange"]],
        title=f"Exp 4 RT Diff ({label} x {v})",
        ylabel="log milliseconds",
        xlabels=['First', 'Average'],
        ax=ax
    )
    
    # Save individual figure if one was created
    if fig is not None and save_figs:
        fig.savefig(
            os.path.join(save_dir, f'exp4_2_rt_diff_{setting}_{version}_filter_{filter_str}.pdf'),
            bbox_inches='tight')
        plt.close(fig)
        #if display_figs:
        #    plt.show()

    return ax

  # Create individual plots
  plot_rt_diff(setting='short', version='regular')
  plot_rt_diff(setting='short', version='blind')
  plot_rt_diff(setting='long', version='regular')
  plot_rt_diff(setting='long', version='blind')

  # Create combined figure
  fig, axs = plt.subplots(1, 4, figsize=(20, 4))
  #fig.suptitle('Experiment 4: Reaction Time Differences', fontsize=DEFAULT_TITLE_SIZE)
  
  # Plot all conditions in the combined figure
  plot_rt_diff(setting='short', version='regular', ax=axs[0])
  plot_rt_diff(setting='short', version='blind', ax=axs[1]) 
  plot_rt_diff(setting='long', version='regular', ax=axs[2])
  plot_rt_diff(setting='long', version='blind', ax=axs[3])
  
  # Adjust layout
  plt.tight_layout()
  
  # Save combined figure
  if save_figs:
      filter_str = ','.join(filter_columns)
      fig.savefig(
          os.path.join(save_dir, f'exp4_2_rt_diff_combined_filter_{filter_str}.pdf'),
          bbox_inches='tight')
  if display_figs:
      plt.show()


if __name__ == "__main__":
  data_dir = '/Users/wilka/git/research/results/human_dyna/'

  ################
  # Load model data
  ################
  model_df = get_model_data(
      qlearning_path=f'{data_dir}/model_data/ql/save_data/ql-big-2/tota=40000000,exp=exp2/seed=*',
      sf_path=f'{data_dir}/model_data/usfa/save_data/usfa-big-10-search/sf_h=1024,num_=2,tota=40000000,exp=exp2/seed=*',
      dyna_path=f'{data_dir}/model_data/dynaq_shared/save_data/dynaq-big-4/alg=dynaq_shared,agen=256,tota=100000000,exp=exp2/seed=*',
      search_path=f'{data_dir}/search_algos',
      overwrite_episodes=False,
      overwrite_df=False,
      cache_dir=f'{data_dir}/model_data/cache',
  )

  ################
  # Load user data
  ################
  searches = {
      'Paths': f'{data_dir}/user_data/*exps*/*v1*paths*.json',
      'Start': f'{data_dir}/user_data/*exps*/*v1*start*.json',
      'Plan (Tell)': f'{data_dir}/user_data/*exps*/*v1*r1-t0-plan*.json',
      "Plan (Don't Tell)": f'{data_dir}/user_data/*exps*/*v1*r0-t0-plan*.json',
      'Shortcut': f'{data_dir}/user_data/*exps*/*v1*shortcut*.json',
  }

  valid_files = get_valid_files(searches, verbose=True, plot=False)
  user_df = get_human_data(
      valid_files,
      overwrite_episode_data=False,
      overwrite_episode_info=False
  )
  ################
  # Exp 1 analysis
  ################
  save_dir = f'{data_dir}/analysis_results/'
  os.makedirs(save_dir, exist_ok=True)

  experiment_1_results(
    user_df,
    model_df,
    save_dir=save_dir,
    save_figs=True)

  experiment_2_results(
      user_df,
      model_df,
      save_dir=save_dir,
      save_figs=True)
