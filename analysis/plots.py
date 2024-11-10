import collections
from functools import partial
from typing import Callable, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import jax
import numpy as np
from scipy import stats
from collections import defaultdict
import pandas as pd
import polars as pl

from housemaze.env import KeyboardActions
from analysis.data_loading import EpisodeData
from analysis import housemaze_utils
from nicewebrl.dataframe import DataFrame

from analysis.housemaze_utils import success, success_or_not_terminate, terminated, went_to_junction

model_colors = {
    'human_success': '#0072B2',
    'human': '#009E73',
    'human_terminate': '#D55E00',
    'qlearning': '#CC79A7',
    'dyna': '#F0E442',
    'bfs': '#56B4E9',
    'dfs': '#E69F00'
}

model_names = {
    'human': 'Human',
    'human_terminate': 'Human (finished)',
    'human_success': 'Human (Succeeded)',
    'qlearning': 'Q-learning',
    'usfa': 'Successor features',
    'dyna': 'Multitask preplay',
    'bfs': 'Breadth-first search',
    'dfs': 'Depth-first search',
}

maze_name = {
    'big_m2_maze2': "Start Manipulation",
    'big_m2_maze2_offpath' : "Start Manipulation: Off-path",
    'big_m2_maze2_onpath' : "Start Manipulation: On-path",
    'big_m3_maze1': "Path Manipulation",
    'big_m3_maze1_eval': "Path Manipulation: Evaluation",
    'big_m4_maze_long': "Plan Manipulation (long)",
    'big_m4_maze_long_eval_same': "Plan Manipulation (long): Same location",
    'big_m4_maze_long_eval_diff' : "Plan Manipulation (long): New location",
    'big_m4_maze_short' : "Plan Manipulation (short)",
    'big_m4_maze_short_eval_same' : "Plan Manipulation (short): Same location",
    'big_m4_maze_short_eval_diff' : "Plan Manipulation (short): New location",
}

# Add these constants at the top of the file
DEFAULT_TITLE_SIZE = 14
DEFAULT_LABEL_SIZE = 12
DEFAULT_LEGEND_SIZE = 10

def total_rt(e: EpisodeData):
    return np.sum(e.reaction_times[:-1])

def avg_rt(e: EpisodeData):
    return np.mean(e.reaction_times[:-1])

def post_first_rt(e: EpisodeData):
    return e.reaction_times[1:-1]

def first_rt(e: EpisodeData):
    return e.reaction_times[0]

def get_ylim_without_outliers(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return max(0, lower_bound), upper_bound

def bar_plot_results(model_dict, ax=None, figsize=(8, 4), error_bars=True, title="", ylabel="", autolabel=True):
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Set style
    sns.set_style("whitegrid")

    # Prepare data for plotting
    models = list(model_dict.keys())
    values = [np.mean(arr) for arr in model_dict.values()]
    errors = [np.std(arr)/np.sqrt(len(arr)) if not model.startswith('human') else 0
              for model, arr in model_dict.items()] if error_bars else None

    # Create the bar plot with consistent colors
    bars = ax.bar([model_names.get(model, model) for model in models], values, 
                  yerr=errors, capsize=5, 
                  color=[model_colors.get(model, '#333333') for model in models])

    # Customize the plot
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Rotate x-axis labels and set alignment
    ax.tick_params(axis='x', rotation=45)
    ax.set_xticklabels(ax.get_xticklabels(), ha='right')

    # Add value labels on top of each bar
    if autolabel:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom')

    # Adjust layout if we created the figure
    if ax is None:
        plt.tight_layout()
        plt.show()
        
    return ax

def success_termination_results(success_dict, termination_dict, title="", ylabel=""):
    # Set up the plot style
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Prepare data for plotting
    models = list(success_dict.keys())
    success_values = [np.mean(arr) for arr in success_dict.values()]
    success_errors = [np.std(arr)/np.sqrt(len(arr))
                      for arr in success_dict.values()]
    termination_values = [np.mean(arr) for arr in termination_dict.values()]
    termination_errors = [np.std(arr)/np.sqrt(len(arr))
                          for arr in termination_dict.values()]

    # Set up bar positions
    x = np.arange(len(models))
    width = 0.35

    # Create the bar plot with consistent colors
    fig, ax = plt.subplots(figsize=(12, 6))
    success_bars = ax.bar(x - width/2, success_values, width, yerr=success_errors, capsize=5,
                          color=[model_colors.get(model, '#333333')
                                 for model in models],
                          label='Success Rate', hatch='//')
    termination_bars = ax.bar(x + width/2, termination_values, width, yerr=termination_errors, capsize=5,
                              color=[model_colors.get(model, '#333333')
                                     for model in models],
                              label='Termination Rate', alpha=0.7)

    # Customize the plot
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Data source", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')

    # Add legend
    ax.legend()

    # Add value labels on top of each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')

    autolabel(success_bars)
    autolabel(termination_bars)

    # Adjust layout and display the plot
    fig.tight_layout()
    plt.show()

def add_reuse_columns(df: DataFrame, overlap_threshold=0.15) -> DataFrame:
    """Add a 'reuse' column to the DataFrame indicating whether each episode reused paths.
    
    Args:
        df (DataFrame): Input DataFrame
        manipulation (int, optional): Manipulation number. Defaults to 3.
        mazes (List[str], optional): List of maze names. Defaults to None.
        overlap_threshold (float, optional): Threshold for path reuse. Defaults to 0.15.
    
    Returns:
        DataFrame: DataFrame with added 'reuse' column
    """

    # Create a dictionary to store reuse values
    reuse_dict = {}
    def update_reuse_dict(manipulation, train_mazes, test_mazes):
        # Get unique users
        users = df.filter(manipulation=manipulation)['user_id'].unique()

        # Process each user's data
        for user in users:
            for train_maze, test_maze in zip(train_mazes, test_mazes):
                # Get train episodes
                train = df.filter(
                    user=user, maze=train_maze, room=0, eval=False,
                    episode_filter=lambda e: not success(e)
                )

                if len(train.episodes) == 0:
                    continue

                # Create map for training episodes
                train_map = housemaze_utils.create_maps(train.episodes).sum(0)

                # Get test episodes
                test = df.filter(user=user, maze=test_maze, eval=True)

                # Process each test episode
                for idx, row in enumerate(test._df.iter_rows(named=True)):
                    global_index = row['global_episode_idx']
                    episode = test.episodes[idx]
                    # Create map for single test episode
                    test_map = housemaze_utils.create_maps([episode]).sum(0)
                    overlap = housemaze_utils.overlap(train_map, test_map)

                    # Store the reuse value
                    episode_id = (user, test_maze, global_index)
                    reuse_dict[episode_id] = overlap.mean() > overlap_threshold

    #-----------------
    # paths manipulation (3)
    #-----------------
    # Define mazes if not provided
    manipulation = 3
    train_mazes = test_mazes = [
        'big_m3_maze1_(F,F)',
        'big_m3_maze1_(F,T)',
        'big_m3_maze1_(T,F)',
        'big_m3_maze1_(T,T)',
    ]
    update_reuse_dict(manipulation, train_mazes, test_mazes)
    #-----------------
    # shortcut manipulation (1)
    #-----------------
    # Define mazes if not provided
    manipulation = 1
    train_mazes  = [
        'big_m1_maze3_(F,F)',
        'big_m1_maze3_(F,T)',
        'big_m1_maze3_(T,F)',
        'big_m1_maze3_(T,T)',
    ]

    test_mazes = [
        'big_m1_maze3_shortcut_(F,F)',
        'big_m1_maze3_shortcut_(F,T)',
        'big_m1_maze3_shortcut_(T,F)',
        'big_m1_maze3_shortcut_(T,T)',
    ]
    update_reuse_dict(manipulation, train_mazes, test_mazes)

    #-----------------
    # add everything
    #-----------------
    # Create a new column with reuse values
    reuse_values = pl.Series([
        reuse_dict.get((row['user_id'], row['maze'], row['global_episode_idx']), None)
        for row in df.iter_rows(named=True)
    ])

    # Add the new column to the DataFrame
    new_df = df.with_columns([
        pl.Series("reuse", reuse_values)
    ])

    return new_df

# Function to remove outliers using IQR method


def coplot_dists(df1, df2, val, settings: dict):
    mdf1 = df1.filter(**settings)
    mdf2 = df2.filter(**settings)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 3))

    # Combine data to calculate shared bins
    all_data = np.concatenate([mdf1[val], mdf2[val]])
    bins = np.histogram_bin_edges(all_data, bins=30)

    # Create overlaid histograms with shared bins
    sns.histplot(
        data=mdf1[val],
        bins=bins,
        color='skyblue',
        alpha=0.5,
        ax=ax,
        label='User DF'
    )

    sns.histplot(
        data=mdf2[val],
        bins=bins,
        color='red',
        alpha=0.5,
        ax=ax,
        label='Filtered DF'
    )

    # Add title and legend
    ax.set_title(f'Distribution of {val}')
    ax.legend()

    plt.show()

def filter_outliers(
        df: DataFrame,
        filter_settings: dict,
        filter_columns: List[str],
        method: str = 'iqr',
        threshold: float = 1.5) -> DataFrame:
    """Filter outliers from reaction time columns using various methods.
    
    Args:
        df (DataFrame): Input DataFrame
        filter_settings (dict): Settings to filter evaluation data
        filter_columns (list): List of reaction time column names to check
        method (str): Method to use for outlier detection ('iqr', 'zscore', or 'percentile')
        threshold (float): Threshold for outlier detection:
            - For IQR: Number of IQRs to use (default: 1.5)
            - For zscore: Number of standard deviations (default: 3)
            - For percentile: Percentile range from 0-100 (default: 1, meaning 1st-99th percentile)
    
    Returns:
        DataFrame: Filtered DataFrame with outliers removed
    """
    original_size = len(df)
    
    # Get evaluation data and indices once
    eval_data = df.filter(**filter_settings, reindex=False)
    eval_indices = eval_data['index'].to_numpy()
    
    # Create numpy array for the mask (all True initially)
    eval_mask = np.ones(len(eval_data), dtype=bool)
    bounds = {}

    # Check each RT column
    bounds = {}
    for col in filter_columns:
        # Get values for calculations
        values = np.array(eval_data[col])
        
        if method == 'iqr':
            # Calculate quartiles and IQR
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            # Calculate bounds
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
        elif method == 'zscore':
            # Calculate mean and standard deviation
            mean = np.mean(values)
            std = np.std(values)
            
            # Calculate bounds
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            
        elif method == 'percentile':
            # Calculate percentile bounds
            lower_bound = np.percentile(values, threshold)
            upper_bound = np.percentile(values, 100 - threshold)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        bounds[col] = {
            'lower': lower_bound,
            'upper': upper_bound
        }
        
        # Update eval_mask with column constraints
        col_mask = (values >= lower_bound) & (values <= upper_bound)
        eval_mask &= col_mask

    # Create full mask (all True)
    full_mask = np.ones(len(df), dtype=bool)
    # Update only the evaluation indices in the full mask
    full_mask[eval_indices] = eval_mask

    # Apply the mask to filter outliers
    #filtered_df = df.filter(full_mask)
    filtered_df = df.filter(pl.Series(full_mask))
    removed_df = eval_data.reindex().filter(pl.Series(~eval_mask))

    # Print how many were filtered
    n_filtered = original_size - len(filtered_df)
    if n_filtered > 0:
        users_removed = set()
        print(f"Filtered {n_filtered} outliers from {len(eval_mask)} rows using {method} method")
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

def filter_rt_outliers(episodes: List[EpisodeData], rt_fn: Callable, n_std: float = 3.0) -> List[EpisodeData]:
    """Filter episodes whose reaction times are outside n standard deviations from the mean.
    
    Args:
        episodes (List[EpisodeData]): List of episodes to filter
        rt_fn (Callable): Function to calculate reaction time for an episode
        n_std (float): Number of standard deviations for threshold (default: 3.0)
    
    Returns:
        List[EpisodeData]: Filtered list of episodes
    """
    # Calculate log reaction times
    rts = np.log([rt_fn(e) for e in episodes])

    # Calculate mean and std
    mean_rt = np.mean(rts)
    std_rt = np.std(rts)
    threshold = n_std * std_rt

    # Create mask for valid data points
    mask = np.abs(rts - mean_rt) <= threshold

    # Apply mask and return filtered episodes
    filtered_episodes = [ep for ep, keep in zip(episodes, mask) if keep]

    # Print how many were filtered
    n_filtered = len(episodes) - len(filtered_episodes)
    if n_filtered > 0:
        print(f"Filtered {n_filtered} outliers from {len(episodes)} episodes")

    return filtered_episodes

###################
# Training (1) reaction times (2) success rate (3) episode count
###################

def plot_train_reaction_times(
        user_df: DataFrame,
        rt_fn: str = 'first',
        stages: Optional[List[str]] = None,
        n_cols: int = 4,
        **kwargs
        ):
    title_size = kwargs.pop('title_size', DEFAULT_TITLE_SIZE)
    label_size = kwargs.pop('label_size', DEFAULT_LABEL_SIZE)
    legend_size = kwargs.pop('legend_size', DEFAULT_LEGEND_SIZE)
    
    """Plot the reaction times for each stage and user.

    Args:
        user_df (pd.DataFrame): A DataFrame containing the user data.
        rt_fn (Callable): Function to calculate reaction time for an episode.
        stages (Optional[List[str]]): List of stages to plot.
    """
    user_df = user_df.filter(eval=False)
    stages = stages or user_df['name'].unique()
    users = user_df['user_id'].unique()

    # Calculate the number of rows and columns
    n_stages = len(stages)
    n_cols = min(n_cols, n_stages)
    n_rows = (n_stages + n_cols - 1) // n_cols

    if n_stages == 1:
        fig, ax = plt.subplots(figsize=(4, 4))
        axs = np.array([[ax]])  # Wrap the single axis in a 2D array
    else:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
        if n_rows == 1:
            axs = axs.reshape(1, -1)  # Ensure axs is always 2D

    for i, name in enumerate(stages):
        row = i // n_cols
        col = i % n_cols
        ax = axs[row, col]
        sub_df = user_df.filter(name=name, room=0)

        all_reaction_times = []

        for user in users:
            episodes = sub_df.filter(user_id=user).episodes

            episodes = [e for e in episodes if success(e)]
            if len(episodes) < 8:
                continue
            
            if rt_fn == 'first':
                rt_fn_ = first_rt
            elif rt_fn == 'speed':
                rt_fn_ = avg_rt
            else:
                raise ValueError(f'Unknown reaction time function: {rt_fn}')

            reaction_times = [rt_fn_(e) for e in episodes]
            x = np.arange(len(reaction_times))
            ax.plot(x, reaction_times, color='gray', alpha=0.3)

            all_reaction_times.append(reaction_times)

        # Calculate and plot the average reaction time
        if all_reaction_times:
            avg_reaction_times = np.mean(all_reaction_times, axis=0)
            ax.plot(x, avg_reaction_times, color='blue', linewidth=3, label='Average')
        
        # Set y-axis limits without outliers
        ymin, ymax = get_ylim_without_outliers(all_reaction_times)
        ax.set_ylim(ymin, ymax)
        
        stage = maze_name.get(name, name)
        ax.set_title(stage, fontsize=title_size)
        ax.set_xlabel('Episode', fontsize=label_size)
        ax.set_ylabel(f'Reaction Time ({rt_fn})', fontsize=label_size)
        ax.legend(loc='upper left', fontsize=legend_size)
        #ax.tick_params(axis='both', which='major', labelsize=label_size)
        ax.grid(True, linestyle='--', alpha=0.7)

    # Remove any unused subplots
    for i in range(n_stages, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axs[row, col])

    plt.tight_layout()
    plt.show()

def plot_train_reaction_times_dual(
        user_df: DataFrame,
        stages: Optional[List[str]] = None,
        **kwargs
):
    title_size = kwargs.pop('title_size', DEFAULT_TITLE_SIZE)
    label_size = kwargs.pop('label_size', DEFAULT_LABEL_SIZE)
    legend_size = kwargs.pop('legend_size', DEFAULT_LEGEND_SIZE)
    
    """Plot the reaction times for each stage and user, with average and speed in separate columns.

    Args:
        user_df (pd.DataFrame): A DataFrame containing the user data.
        stages (Optional[List[str]]): List of stages to plot.
    """
    user_df = user_df.filter(eval=False)
    stages = stages or user_df['name'].unique()
    users = user_df['user_id'].unique()

    # Calculate the number of rows
    n_stages = len(stages)
    n_rows = n_stages

    fig, axs = plt.subplots(n_rows, 2, figsize=(8, 4 * n_rows), squeeze=False)

    for i, name in enumerate(stages):
        sub_df = user_df.filter(name=name, room=0)

        for j, (rt_fn, rt_label) in enumerate([('first', 'First'), ('speed', 'Speed')]):
            ax = axs[i, j]
            all_reaction_times = []

            for user in users:
                episodes = sub_df.filter(user_id=user).episodes

                episodes = [e for e in episodes if success(e)]
                if len(episodes) < 8:
                    continue

                if rt_fn == 'first':
                    rt_fn_ = first_rt
                elif rt_fn == 'speed':
                    rt_fn_ = avg_rt

                reaction_times = [rt_fn_(e) for e in episodes]
                x = np.arange(len(reaction_times))
                ax.plot(x, reaction_times, color='gray', alpha=0.3)

                all_reaction_times.append(reaction_times)

            # Calculate and plot the average reaction time
            if all_reaction_times:
                avg_reaction_times = np.mean(all_reaction_times, axis=0)
                ax.plot(x, avg_reaction_times, color='blue', linewidth=3, label='Average')

            # Set y-axis limits without outliers
            ymin, ymax = get_ylim_without_outliers(all_reaction_times)
            ax.set_ylim(ymin, ymax)

            stage = maze_name.get(name, name)
            ax.set_title(f"{stage}  - {rt_label}", fontsize=title_size)
            ax.set_xlabel('Episode', fontsize=label_size)
            ax.set_ylabel(f'Reaction Time ({rt_label})', fontsize=label_size)
            ax.legend(loc='upper left', fontsize=legend_size)
            #ax.tick_params(axis='both', which='major', labelsize=label_size)
            ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def plot_train_success_rate_histograms(
        user_df: DataFrame,
        stages: Optional[List[str]] = None,
        n_cols: int = 2,
        **kwargs
):
    title_size = kwargs.pop('title_size', DEFAULT_TITLE_SIZE)
    label_size = kwargs.pop('label_size', DEFAULT_LABEL_SIZE)
    legend_size = kwargs.pop('legend_size', DEFAULT_LEGEND_SIZE)
    
    """Plot histograms of success rates for each stage.

    Args:
        user_df (DataFrame): A DataFrame containing the user data.
        stages (Optional[List[str]]): List of stages to plot.
        n_cols (int): Number of columns in the plot grid.
        **kwargs: Additional keyword arguments for plot customization.
    """

    user_df = user_df.filter(eval=False)
    stages = stages or user_df['name'].unique()
    users = user_df['user_id'].unique()

    # Calculate the number of rows and columns
    n_stages = len(stages)
    n_cols = min(n_cols, n_stages)
    n_rows = (n_stages + n_cols - 1) // n_cols

    if n_stages == 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        axs = np.array([[ax]])  # Wrap the single axis in a 2D array
    else:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 6 * n_rows))
        if n_rows == 1:
            axs = axs.reshape(1, -1)  # Ensure axs is always 2D

    max_frequency = 0

    # First pass to determine the maximum frequency
    for i, name in enumerate(stages):
        row, col = divmod(i, n_cols)
        sub_df = user_df.filter(name=name, room=0)

        success_rates = []
        for user in users:
            episodes = sub_df.filter(user_id=user).episodes
            if len(episodes) < 8:
                continue
            success_rate = np.mean([success(e) for e in episodes])
            success_rates.append(success_rate)

        if success_rates:
            n, _, _ = axs[row, col].hist(
                success_rates, bins=20, edgecolor='black')
            max_frequency = max(max_frequency, max(n))

    # Function to plot histogram and add statistics
    def plot_histogram(ax, data, title):
        n, bins, patches = ax.hist(data, bins=20, edgecolor='black')
        ax.set_title(title, fontsize=title_size)
        ax.set_xlabel('Success Rate', fontsize=label_size)
        ax.set_ylabel('Frequency', fontsize=label_size)
        ax.tick_params(axis='both', which='major', labelsize=10)

        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        mean_rate = np.mean(data)
        median_rate = np.median(data)
        ax.axvline(mean_rate, color='red', linestyle='dashed',
                   linewidth=2, label=f'Mean: {mean_rate:.2f}')
        ax.axvline(median_rate, color='green', linestyle='dashed',
                   linewidth=2, label=f'Median: {median_rate:.2f}')
        ax.legend(fontsize=legend_size)

        return mean_rate, median_rate

    # Second pass to plot with consistent y-axis and add statistics
    for i, name in enumerate(stages):
        row, col = divmod(i, n_cols)
        ax = axs[row, col]
        sub_df = user_df.filter(name=name, room=0)

        success_rates = []
        for user in users:
            episodes = sub_df.filter(user_id=user).episodes
            if len(episodes) < 8:
                continue
            success_rate = np.mean([success(e) for e in episodes])
            success_rates.append(success_rate)

        if success_rates:
            ax.clear()  # Clear the previous plot
            stage = maze_name.get(name, name)
            plot_histogram(ax, success_rates, stage)
            # Set y-axis limit with 10% padding
            ax.set_ylim(0, max_frequency * 1.1)

    # Remove any unused subplots
    for i in range(n_stages, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        fig.delaxes(axs[row, col])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust to make room for suptitle
    plt.show()

def plot_episode_counts(
        user_df: DataFrame,
        stages: Optional[List[str]] = None,
        n_cols: int = 2,
        **kwargs
):
    title_size = kwargs.pop('title_size', DEFAULT_TITLE_SIZE)
    label_size = kwargs.pop('label_size', DEFAULT_LABEL_SIZE)
    legend_size = kwargs.pop('legend_size', DEFAULT_LEGEND_SIZE)
    
    """Plot histograms of episode counts for each stage.

    Args:
        user_df (DataFrame): A DataFrame containing the user data.
        stages (Optional[List[str]]): List of stages to plot.
        n_cols (int): Number of columns in the plot grid.
        **kwargs: Additional keyword arguments for plot customization.
    """
    user_df = user_df.filter(eval=False)
    stages = stages or user_df['name'].unique()
    users = user_df['user_id'].unique()

    # Calculate the number of rows and columns
    n_stages = len(stages)
    n_cols = min(n_cols, n_stages)
    n_rows = (n_stages + n_cols - 1) // n_cols

    if n_stages == 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        axs = np.array([[ax]])  # Wrap the single axis in a 2D array
    else:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 6 * n_rows))
        if n_rows == 1:
            axs = axs.reshape(1, -1)  # Ensure axs is always 2D

    max_frequency = 0

    # Function to plot histogram and add statistics
    def plot_histogram(ax, data, title):
        n, bins, patches = ax.hist(data, bins=range(min(data), max(data) + 2, 1),
                                   edgecolor='black', align='left')
        ax.set_title(title, fontsize=title_size)
        ax.set_xlabel('Number of Episodes', fontsize=label_size)
        ax.set_ylabel('Frequency', fontsize=label_size)
        ax.tick_params(axis='both', which='major', labelsize=label_size)

        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        mean_count = np.mean(data)
        median_count = np.median(data)
        ax.axvline(mean_count, color='red', linestyle='dashed',
                   linewidth=2, label=f'Mean: {mean_count:.2f}')
        ax.axvline(median_count, color='green', linestyle='dashed',
                   linewidth=2, label=f'Median: {median_count:.2f}')
        ax.legend(fontsize=legend_size)

        return max(n)

    for i, name in enumerate(stages):
        row, col = divmod(i, n_cols)
        ax = axs[row, col]
        sub_df = user_df.filter(name=name, room=0)

        episode_counts = []
        for user in users:
            episodes = sub_df.filter(user_id=user).episodes
            episode_counts.append(len(episodes))

        if episode_counts:
            stage = maze_name.get(name, name)
            max_freq = plot_histogram(ax, episode_counts, stage)
            max_frequency = max(max_frequency, max_freq)

    # Set consistent y-axis limits
    for ax in axs.flatten():
        if ax.get_title():  # Check if the subplot is used
            ax.set_ylim(0, max_frequency * 1.1)

    # Remove any unused subplots
    for i in range(n_stages, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        fig.delaxes(axs[row, col])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust to make room for suptitle
    plt.show()

def plot_episode_length_seconds(user_df: DataFrame, settings=None, **kwargs):
    title_size = kwargs.pop('title_size', DEFAULT_TITLE_SIZE)
    label_size = kwargs.pop('label_size', DEFAULT_LABEL_SIZE)
    legend_size = kwargs.pop('legend_size', DEFAULT_LEGEND_SIZE)

    stages = kwargs.pop('stages', None)

    settings = settings or dict(eval=False)
    user_df = user_df.filter(**settings)
    stages = stages or user_df['name'].unique()
    user_ids = user_df['user_id'].unique()

    # Dictionary to store episode lengths for each user and stage
    user_episode_lengths = defaultdict(lambda: defaultdict(list))

    for stage in stages:
        for user_id in user_ids:
            # Filter the DataFrame for the current user and stage
            user_data = user_df.filter(user_id=user_id, name=stage, room=0)
            for episode in user_data.episodes:
                # Calculate episode length as the sum of reaction times
                episode_length = total_rt(episode)
                user_episode_lengths[stage][user_id].append(episode_length)

    # Calculate overall episode lengths across all stages
    overall_episode_lengths = [
        length for s in stages for user_lengths in user_episode_lengths[s].values() for length in user_lengths]

    # Create a grid of subplots
    n_stages = len(stages)
    n_cols = min(2, n_stages)
    n_rows = (n_stages + 1) // 2  # +1 for the overall plot
    if n_stages == 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        axs = np.array([[ax]])  # Wrap the single axis in a 2D array
    else:
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(8 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axs = axs.reshape(1, -1)  # Ensure axs is always 2D


    # Function to plot histogram and add statistics
    def plot_histogram(ax, data, title):
        n, bins, patches = ax.hist(data, bins=30, edgecolor='black')
        ax.set_title(title, fontsize=title_size)
        ax.set_xlabel('Episode Length (seconds)', fontsize=label_size)
        ax.set_ylabel('Frequency', fontsize=label_size)
        ax.tick_params(axis='both', which='major', labelsize=label_size)
        ax.grid(True, alpha=0.3)

        mean_length = np.mean(data)
        median_length = np.median(data)
        ax.axvline(mean_length, color='red', linestyle='dashed',
                   linewidth=3, label=f'Mean: {mean_length:.2f}s')
        ax.axvline(median_length, color='green', linestyle='dashed',
                   linewidth=3, label=f'Median: {median_length:.2f}s')
        ax.legend(fontsize=legend_size)

        return mean_length, median_length, max(n)

    # First pass to determine the maximum frequency
    max_frequency = 0
    for i, stage in enumerate(stages):
        data = [length for user_lengths in user_episode_lengths[stage].values()
                for length in user_lengths]
        _, _, freq = plot_histogram(
            axs[i//n_cols, i % n_cols], data, maze_name.get(stage, stage))
        max_frequency = max(max_frequency, freq)

    # Include overall data in max frequency calculation
    _, _, freq = plot_histogram(
        axs[-1, -1], overall_episode_lengths, 'Overall')
    max_frequency = max(max_frequency, freq)

    # Second pass to plot with consistent y-axis and add text
    for i, stage in enumerate(stages):
        data = [length for user_lengths in user_episode_lengths[stage].values()
                for length in user_lengths]
        row, col = divmod(i, n_cols)
        axs[row, col].clear()  # Clear the previous plot
        plot_histogram(axs[row, col], data, maze_name.get(stage, stage))
        # Set y-axis limit with 10% padding
        axs[row, col].set_ylim(0, max_frequency * 1.1)

    ## Plot overall histogram with consistent y-axis
    #axs[-1, -1].clear()  # Clear the previous plot
    #plot_histogram(axs[-1, -1], overall_episode_lengths, 'Overall')
    ## Set y-axis limit with 10% padding
    #axs[-1, -1].set_ylim(0, max_frequency * 1.1)

    # Remove any unused subplots
    for i in range(n_stages + 1, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        fig.delaxes(axs[row, col])

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust to make room for suptitle
    plt.show()


#########################################################
# Manipulation-specific plots
#########################################################
def plot_compare_rt(
    rts1: np.ndarray,
    rts2: np.ndarray,
    label1: str = 'RT1',
    label2: str = 'RT2',
    title: str = 'Reaction Time Comparison',
    ylabel: str = 'log seconds',
    ylim: Optional[tuple] = None,
    ax=None,
):
    """Compare two sets of reaction times with visualization.
    
    Args:
        rts1 (np.ndarray): First set of reaction times
        rts2 (np.ndarray): Second set of reaction times
        label1 (str): Label for first set
        label2 (str): Label for second set
        title (str): Plot title
        ylabel (str): Y-axis label
        ylim (tuple, optional): Y-axis limits (min, max)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # Calculate log reaction times
    log_rts1 = np.log(rts1)
    log_rts2 = np.log(rts2)

    # Calculate means and standard errors
    mean1, mean2 = np.mean(log_rts1), np.mean(log_rts2)
    std1 = np.std(log_rts1) / np.sqrt(len(log_rts1))
    std2 = np.std(log_rts2) / np.sqrt(len(log_rts2))

    # Set y-axis limits
    if ylim is None:
        all_rts = np.concatenate([log_rts1, log_rts2])
        y_min, y_max = np.percentile(all_rts, [1, 99])
        y_range = y_max - y_min
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
    else:
        y_min, y_max = ylim

    # Plot means as horizontal lines
    ax.axhline(y=mean1, color='blue',
               linestyle='-', linewidth=2, label=f'{label1} Mean: {mean1:.2f}')
    ax.axhspan(mean1 - std1, mean1 + std1,
               alpha=0.2, color='blue', label=f'{label1} SE: {std1:.2f}')

    # Add standard error ranges
    ax.axhline(y=mean2, color='green',
               linestyle='-', linewidth=2, label=f'{label2} Mean: {mean2:.2f}')
    ax.axhspan(mean2 - std2, mean2 + std2,
               alpha=0.2, color='green', label=f'{label2} SE: {std2:.2f}')

    # Add strip plot for individual data points
    x_jitter1 = np.random.normal(0.25, 0.02, size=len(log_rts1))
    x_jitter2 = np.random.normal(0.75, 0.02, size=len(log_rts2))
    ax.scatter(x_jitter1, log_rts1, color='black', alpha=0.5)
    ax.scatter(x_jitter2, log_rts2, color='black', alpha=0.5)

    # Customize plot
    ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)
    ax.set_ylabel(ylabel, fontsize=DEFAULT_LABEL_SIZE)
    ax.set_xticks([0.25, 0.75])
    ax.set_xticklabels([label1, label2], ha='center')
    ax.tick_params(axis='both', which='major', labelsize=DEFAULT_LABEL_SIZE)
    ax.set_ylim(y_min, y_max)
    ax.legend(fontsize=DEFAULT_LEGEND_SIZE, loc='lower center')

    # Add statistical test
    t_stat, p_value = stats.ttest_ind(log_rts1, log_rts2)
    ax.text(0.5, 0.98, f't-stat: {t_stat:.3f}\np-value: {p_value:.3f}',
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='center',
            bbox=dict(facecolor='white', alpha=0.8))

def reaction_times_dual(
        episodes1: List[EpisodeData],
        episodes2: List[EpisodeData],
        label1='group1',
        label2='group2',
        rt_types=['first', 'speed'],
        ylim=None):
    rt_functions = {
        'speed': avg_rt,
        'post_first': lambda e: np.mean(post_first_rt(e)),
        'first': first_rt,
        'total': total_rt
    }

    fig, axs = plt.subplots(1, len(rt_types), figsize=(5 * len(rt_types), 6))
    
    # Ensure axes is always a list, even for a single subplot
    if len(rt_types) == 1:
        axs = [axs]

    def plot_rt_comparison(ax, rt_type, rt_fn, i=0):
        # Calculate log reaction times for both groups
        rts1 = np.log([rt_fn(e)*1000 for e in filter_rt_outliers(episodes1, rt_fn)])
        rts2 = np.log([rt_fn(e)*1000 for e in filter_rt_outliers(episodes2, rt_fn)])
        
        # Calculate means and standard errors
        mean1, mean2 = np.mean(rts1), np.mean(rts2)
        std1 = np.std(rts1) / np.sqrt(len(rts1))
        std2 = np.std(rts2) / np.sqrt(len(rts2))

        # Set y-axis limits
        if ylim is None:
            all_rts = np.concatenate([rts1, rts2])
            y_min, y_max = np.percentile(all_rts, [1, 99])
            y_range = y_max - y_min
            y_min -= 0.1 * y_range
            y_max += 0.1 * y_range
        else:
            if len(ylim) == len(rt_types):
                y_min, y_max = ylim[i]
            else:
                y_min, y_max = ylim

        # Create box plot with individual points
        #box_data = [rts1, rts2]
        labels = [label1, label2]
        
        # Plot means as horizontal lines
        ax.axhline(y=mean1, color='blue', 
                   linestyle='-', linewidth=2, label=f'{label1} Mean: {mean1:.2f}')
        ax.axhspan(mean1 - std1, mean1 + std1,
                   alpha=0.2, color='blue', label=f'{label1} SE: {std1:.2f}')

        # Add standard error ranges
        ax.axhline(y=mean2, color='green', 
                   linestyle='-', linewidth=2, label=f'{label2} Mean: {mean2:.2f}')
        ax.axhspan(mean2 - std2, mean2 + std2,
                   alpha=0.2, color='green', label=f'{label2} SE: {std2:.2f}')
        
        # Add strip plot for individual data points
        x_jitter1 = np.random.normal(0.25, 0.02, size=len(rts1))
        x_jitter2 = np.random.normal(0.75, 0.02, size=len(rts2))
        ax.scatter(x_jitter1, rts1, color='black', alpha=0.5)
        ax.scatter(x_jitter2, rts2, color='black', alpha=0.5)

        if rt_type == 'first':
            title = "First Reaction Time"
            ylabel = "log seconds"
        elif rt_type == 'speed':
            title = "Average Reaction Time"
            ylabel = "log avg(seconds/step)"
        elif rt_type == 'total':
            title = "Total Reaction Time"
            ylabel = "log total(seconds)"
        elif rt_type == 'post_first':
            title = "Average Post-First Reaction Time"
            ylabel = "log avg(seconds/step)"
        
        ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)
        ax.set_ylabel(ylabel, fontsize=DEFAULT_LABEL_SIZE)
        ax.set_xticks([0.25, 0.75])
        ax.set_xticklabels(labels, ha='right')
        ax.tick_params(axis='both', which='major', labelsize=DEFAULT_LABEL_SIZE)
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=DEFAULT_LEGEND_SIZE)

    # Plot for each RT type
    for i, rt_type in enumerate(rt_types):
        plot_rt_comparison(axs[i], rt_type, rt_functions[rt_type], i)

    plt.tight_layout()
    plt.show()

def reaction_times_difference(
        cond1: DataFrame,
        cond2: DataFrame,
        rt_types=['first', 'avg', 'total'],
        filter_outliers=False,
        axs=None,
        ylim=None):

    # Create empty lists to store data
    data_rows = []
    rt_functions = {'first': first_rt, 'avg': avg_rt, 'total': total_rt}
    users = set(cond1['user_id'].unique()) & set(cond2['user_id'].unique())

    # Collect raw reaction times for DataFrame
    for user in users:
        cond1_user = cond1.filter(user_id=user)
        cond2_user = cond2.filter(user_id=user)

        if len(cond1_user.episodes) == 0 or len(cond2_user.episodes) == 0:
            continue
        
        if len(cond1_user.episodes) != len(cond2_user.episodes):
            print(f"Skipping {user}: {len(cond1_user.episodes)} != {len(cond2_user.episodes)}")
            continue

        nepisodes = len(cond1_user.episodes)
        
        # Calculate raw RTs for each episode
        for i in range(nepisodes):
            row = {'user_id': user, 'episode': i}
            
            # Add RTs for both conditions and both RT types
            for rt_type, rt_fn in rt_functions.items():
                row[f'{rt_type}_rt_cond1'] = np.asarray(rt_fn(cond1_user.episodes[i]))
                row[f'{rt_type}_rt_cond2'] = np.asarray(rt_fn(cond2_user.episodes[i]))

            data_rows.append(row)

    # Create DataFrame with raw RTs
    df = pd.DataFrame(data_rows)

    # Function to remove outliers using IQR method
    def remove_outliers_iqr(df, columns, paired=True):
        """Remove outliers from paired measurements using IQR method.
        
        Args:
            df: DataFrame containing the data
            columns: List of column pairs to check for outliers
            paired: If True, removes outliers considering pairs of measurements
        """
        original_size = len(df)
        mask = pd.Series(True, index=df.index)
        
        if paired:
            # For paired data, consider differences between conditions
            for col1, col2 in columns:
                differences = df[col2] - df[col1]
                Q1 = differences.quantile(0.25)
                Q3 = differences.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                mask &= (differences >= lower_bound) & (differences <= upper_bound)
        else:
            # For unpaired data, consider each column separately
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)
        
        df_filtered = df[mask]
        filtered_size = len(df_filtered)
        print(f"Filtered {original_size - filtered_size} outliers")
        return df_filtered

    # Remove outliers from both conditions
    if filter_outliers:
        column_pairs = [
            ('first_rt_cond1', 'first_rt_cond2'),
            #('avg_rt_cond1', 'avg_rt_cond2'),
            ('total_rt_cond1', 'total_rt_cond2')
        ]
        df = remove_outliers_iqr(df, column_pairs, paired=True)

    # Calculate log means per user
    def log_mean(x):
        return np.mean(np.log(np.array(x.tolist())))

    user_means = df.groupby('user_id').agg({
        'first_rt_cond1': log_mean,
        'first_rt_cond2': log_mean,
        'avg_rt_cond1': log_mean,
        'avg_rt_cond2': log_mean,
        'total_rt_cond1': log_mean,
        'total_rt_cond2': log_mean
    }).reset_index()

    # Calculate differences
    for rt_type in rt_types:
        user_means[f'{rt_type}_diff'] = user_means[f'{rt_type}_rt_cond2'] - user_means[f'{rt_type}_rt_cond1']

    # Remove any rows with inf or nan
    user_means = user_means[~user_means.isin([np.inf, -np.inf]).any(axis=1)]
    user_means = user_means.dropna()

    n_plots = len(rt_types)
    if axs is None:
        fig, axs = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))

    # If there's only one plot, axs will be a single axis, not an array
    if n_plots == 1:
        axs = [axs]

    def plot_rt_difference(ax, data, rt_type, i=0):
        diff_data = data[f'{rt_type}_diff'].values
        
        # Calculate mean and standard deviation
        mean = np.mean(diff_data)
        std = np.std(diff_data)/np.sqrt(len(diff_data))

        # Set y-axis limits
        if ylim is None:
            y_min, y_max = np.percentile(diff_data, [1, 99])
            y_range = y_max - y_min
            y_min -= 0.1 * y_range  # Add 10% padding
            y_max += 0.1 * y_range
        else:
            if len(ylim) == len(rt_types):
                y_min, y_max = ylim[i]
            else:
                y_min, y_max = ylim

        # Plot mean as a horizontal line
        ax.axhline(y=mean, color='blue', linestyle='-', linewidth=2, label=f'Mean: {mean:.2f}')
        
        # Add standard deviation range
        ax.axhspan(mean - std, mean + std, alpha=0.2,
                   color='blue', label=f'SE: {std:.2f}')
        
        # Add strip plot for individual data points
        sns.stripplot(data=diff_data, ax=ax, color='black', alpha=0.5, jitter=True)

        if rt_type == 'first':
            title = "First Reaction Time Difference"
            ylabel = "log seconds"
        elif rt_type == 'avg':
            title = "Average Reaction Time Difference"
            ylabel = "log avg(seconds/step) "
        elif rt_type == 'total':
            title = "Total Reaction Time Difference"
            ylabel = "log total(seconds)"
        
        ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)
        ax.set_ylabel(ylabel, fontsize=DEFAULT_LABEL_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=DEFAULT_LABEL_SIZE)

        # Set y-axis limits
        ax.set_ylim(y_min, y_max)

        # Add a horizontal line at y=0
        ax.axhline(y=0, color='r', linestyle='--')
        
        # Add legend
        ax.legend(fontsize=DEFAULT_LEGEND_SIZE)

    # Plot for each RT type using the DataFrame
    for i, rt_type in enumerate(rt_types):
        plot_rt_difference(axs[i], user_means, rt_type, i)

    plt.tight_layout()
    plt.show()

def initial_action_distribution(cond1, cond2, key2model, model_colors, action_indices):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    def plot_action_distribution(ax, episodes, title):
        action_counts = np.zeros(len(action_indices))
        for episode in episodes:
            action0 = episode.actions[0]
            if action0 < 0:
                continue
            action_counts[action_indices.index(int(action0))] += 1

        action_proportions = action_counts / len(episodes)

        x = np.arange(len(action_indices))

        bars = ax.bar(x, action_proportions)

        ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)
        ax.set_ylim(0, 1)
        ax.set_xticks(x)

        # Customize x-axis labels and add model names
        x_labels = []
        for index, action in enumerate(action_indices):
            if action.name in key2model:
                model = key2model[action.name]
                x_labels.append(action.name)
                # x_labels.append(f"{action.name}\n{model}")
                bars[index].set_color(model_colors.get(model, '#333333'))
            else:
                x_labels.append(action.name)

        ax.set_xticklabels(x_labels, rotation=45, ha='right',
                           fontsize=DEFAULT_LABEL_SIZE)
        ax.set_ylabel('Proportion', fontsize=DEFAULT_LABEL_SIZE)
        ax.tick_params(axis='both', which='major',
                       labelsize=DEFAULT_LABEL_SIZE)

        # Add value labels on top of each bar
        for rect in bars:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=DEFAULT_LEGEND_SIZE)

    plot_action_distribution(ax1, cond1.episodes, "Same target location")
    plot_action_distribution(ax2, cond2.episodes, "New target location")

    plt.tight_layout()
    plt.show()

def group_filter_fn(df: DataFrame, min_successes: int = 16):
    successes = df.apply(success)
    remove = True
    if len(successes) == 0:
        return 0, remove
    nsuccess = int(sum(successes))
    remove = nsuccess < min_successes
    #remove |= np.mean(successes) < 0.5
    if remove:
        user = df['user_id'].unique().to_list()[0]
        print(f"removed: user {user} rate: {np.mean(successes)} = {nsuccess}/{min_successes}/{len(successes)}")
    return remove


def success_termination_results(success_dict, termination_dict, title="", ylabel=""):
    # Set up the plot style
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Prepare data for plotting
    models = list(success_dict.keys())
    success_values = [np.mean(arr) for arr in success_dict.values()]
    success_errors = [np.std(arr)/np.sqrt(len(arr))
                      for arr in success_dict.values()]
    termination_values = [np.mean(arr) for arr in termination_dict.values()]
    termination_errors = [np.std(arr)/np.sqrt(len(arr))
                          for arr in termination_dict.values()]

    # Set up bar positions
    x = np.arange(len(models))
    width = 0.35

    # Create the bar plot with consistent colors
    fig, ax = plt.subplots(figsize=(12, 6))
    success_bars = ax.bar(x - width/2, success_values, width, yerr=success_errors, capsize=5,
                          color=[model_colors.get(model, '#333333')
                                 for model in models],
                          label='Success Rate', hatch='//')
    termination_bars = ax.bar(x + width/2, termination_values, width, yerr=termination_errors, capsize=5,
                              color=[model_colors.get(model, '#333333')
                                     for model in models],
                              label='Termination Rate', alpha=0.7)

    # Customize the plot
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Data source", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')

    # Add legend
    ax.legend()

    # Add value labels on top of each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')

    autolabel(success_bars)
    autolabel(termination_bars)

    # Adjust layout and display the plot
    fig.tight_layout()
    plt.show()



#########################################################
# Paths manipulation (3)
#########################################################

def plot_m3_example(user_df: DataFrame, finished=False):
    if finished:
        output_episode_filter = lambda e: not success(e)
    else:
        output_episode_filter = lambda e: not success_or_not_terminate(e)

    subset = user_df.filter_by_group(
        input_episode_filter=group_filter_fn,
        output_episode_filter=output_episode_filter,
        input_settings=dict(eval=False),
        output_settings=dict(manipulation=3, maze='big_m3_maze1_(F,F)'),
    )
    old_path_cond = subset.filter(reuse=True)
    new_path_cond = subset.filter(reuse=False)


    # Create a figure with 3 subplots for render_path
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    housemaze_utils.render_path(old_path_cond.episodes[0], ax=axs[0])
    axs[0].set_title('Using prior path')

    housemaze_utils.render_path(new_path_cond.episodes[0], ax=axs[1])
    axs[1].set_title('Using new path')

    plt.tight_layout()
    plt.show()

def m3_reaction_times(user_df: DataFrame, **kwargs):
    manipulation = 3

    subset = user_df.filter_by_group(
        input_episode_filter=group_filter_fn,
        #output_episode_filter=lambda e: not success_or_not_terminate(e),
        output_episode_filter=lambda e: not success_or_not_terminate(e),
        input_settings=dict(eval=False),
        output_settings=dict(manipulation=manipulation, eval=True),
    )

    # Split episodes based on reuse column
    reuse_episodes = subset.filter(reuse=True).episodes
    no_reuse_episodes = subset.filter(reuse=False).episodes

    # Plot reaction times
    reaction_times_dual(
        no_reuse_episodes,
        reuse_episodes,
        label1='Used New Path',
        label2='Partially reused path',
        **kwargs
    )

def reaction_times_across_conditions_m3(
        episodes1: List[EpisodeData],
        episodes2: List[EpisodeData],
        episodes3: List[EpisodeData], label1='group1', label2='group2', label3='group3', ylim=None):
    rt_types = ['speed', 'first']
    rt_functions = [avg_rt, first_rt]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for i, (ax, rt_fn, rt_type) in enumerate(zip(axes, rt_functions, rt_types)):
        group1_rts = np.array([rt_fn(e) for e in episodes1])
        group2_rts = np.array([rt_fn(e) for e in episodes2])
        group3_rts = np.array([rt_fn(e) for e in episodes3])

        # Create box plot with individual points
        box_data = [group2_rts, group3_rts]
        labels = [label2, label3]

        sns.boxplot(data=box_data, ax=ax, width=0.5,
                    palette=['red', 'green'])
        sns.stripplot(data=box_data, ax=ax, color='black',
                      alpha=0.5, jitter=True)

        ax.set_xticklabels(labels, rotation=45, ha='right',
                           fontsize=DEFAULT_LABEL_SIZE)
        ax.set_ylabel('Reaction Time', fontsize=DEFAULT_LABEL_SIZE)
        ax.set_title(f'{rt_type.capitalize()} Reaction Time',
                     fontsize=DEFAULT_TITLE_SIZE)
        ax.tick_params(axis='both', which='major',
                       labelsize=DEFAULT_LABEL_SIZE)
        if ylim is not None:
            ax.set_ylim(*ylim[i])

    plt.tight_layout()
    plt.show()


def create_success_termination_results_m3(user_df: DataFrame, model_df: DataFrame, **kwargs):
    model_setting = dict(maze_name='big_m3_maze1', eval=True)

    def post_fn(x):
        return np.array(x)[0]

    #def output_transform(l: List):
    #    return np.concatenate(l)

    #def get_human_data(fn):
    #    return user_df.apply_by_group(
    #        fn=fn,
    #        input_episode_filter=group_filter_fn,
    #        input_settings=dict(manipulation=manipulation, eval=False),
    #        output_settings=dict(manipulation=manipulation, eval=True),
    #        output_transform=output_transform
    #    )
    #succeeded = get_human_data(success_fn)

    manipulation = 3
    m3_df = user_df.filter(manipulation=manipulation, eval=True)
    succeeded = m3_df.group_by('user_id').agg(
        pl.col('success').mean()).select('success').to_numpy().flatten()


    
    success_fn = lambda e: 100*success(e)
    def model_fn(e): return jax.vmap(success_fn)(e)

    # Success rate data
    data = {
        'human': 100*succeeded,
        #'human_terminate': finished,
        'qlearning': model_df.apply(fn=model_fn, output_transform=post_fn, algo="qlearning", **model_setting),
        'usfa': model_df.apply(fn=model_fn, output_transform=post_fn, algo="usfa", **model_setting),
        'dyna': model_df.apply(fn=model_fn, output_transform=post_fn, algo="dynaq_shared", **model_setting),
        'bfs': model_df.apply(fn=model_fn, output_transform=post_fn, algo='bfs', **model_setting),
        'dfs': model_df.apply(fn=model_fn, output_transform=post_fn, algo='dfs', **model_setting),
    }

    bar_plot_results(
        data,
        # data_termination,
        title='Success Rate',
        #ylabel='Pr'
        error_bars=False,
        autolabel=False,
        **kwargs
    )

def _reuse_bar_plot_results(
    user_df: DataFrame,
    model_df: DataFrame,
    manipulation: int,
    junction: tuple,
    maze_name: str,
    ax=None,
    ylim=None
) -> tuple:
    """Helper function to create bar plot results for path reuse analysis.
    
    Args:
        user_df (DataFrame): User data
        model_df (DataFrame): Model data
        manipulation (int): Manipulation number
        junction (tuple): Junction coordinates to check
        maze_name (str): Name of the maze
        ax (Optional[Axes]): Matplotlib axes to plot on. If None, creates new figure/axes
        ylim (Optional[tuple]): Y-axis limits
        
    Returns:
        tuple: (figure, axes) - The matplotlib figure and axes objects
    """
    # Calculate statistics
    user_df = user_df.filter(manipulation=manipulation, eval=True)
    human_means = user_df.group_by('user_id').agg(
        pl.col('reuse').mean()).select('reuse').to_numpy().flatten()*100
    human_mean = human_means.mean()
    human_std = human_means.std()
    human_se = human_std / np.sqrt(len(human_means))

    # Perform one-sample t-test against chance (0.5)
    t_stat, p_value = stats.ttest_1samp(human_means, 0.5)

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    sns.set_style("whitegrid")

    model_setting = dict(maze_name=maze_name, eval=True)
    fn = partial(went_to_junction, junction=junction)
    def model_fn(e): return 100*jax.vmap(fn)(e)
    def post_fn(x): return np.array(x)[0]

    # Create bar plot data
    data = {
        'human': human_means,
        'qlearning': model_df.apply(fn=model_fn, output_transform=post_fn, algo="qlearning", **model_setting),
        'usfa': model_df.apply(fn=model_fn, output_transform=post_fn, algo="usfa", **model_setting),
        'dyna': model_df.apply(fn=model_fn, output_transform=post_fn, algo="dynaq_shared", **model_setting),
        'bfs': model_df.apply(fn=model_fn, output_transform=post_fn, algo='bfs', **model_setting),
        'dfs': model_df.apply(fn=model_fn, output_transform=post_fn, algo='dfs', **model_setting),
    }
    #data = jax.tree_map(lambda x: 100*x, data)

    # Plot bars
    ax.bar([model_names.get(model, model) for model in data.keys()],
                      [np.mean(arr) for arr in data.values()],
                      yerr=[np.std(arr)/np.sqrt(len(arr)) for model, arr in data.items()],
                      capsize=5,
                      color=[model_colors.get(model, '#333333') for model in data.keys()])

    # Add individual dots for human data with jitter
    x_pos = 0  # Position of human bar
    x_jitter = np.random.normal(0, 0.15, size=len(human_means))
    ax.scatter([x_pos + j for j in x_jitter], human_means,
               color='black', alpha=0.5, zorder=3)

    # Add chance level line
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Chance level')

    # Add statistics annotations
    stats_text = (f"Human mean: {human_mean:.3f} ± {human_se:.3f}\n"
                 f"t-stat: {t_stat:.3f}\n"
                 f"p-value: {p_value:.3f}")
    ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8))

    # Add p-value annotation
    ax.text(x_pos, ax.get_ylim()[1], f'p = {p_value:.3f}',
            horizontalalignment='center', verticalalignment='bottom')

    ax.set_title('Path Reuse Analysis', fontsize=16)
    ax.set_ylabel('Proportion of Path Reuse', fontsize=12)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([model_names.get(model, model) for model in data.keys()],
                       rotation=45, ha='right')

    if ylim is not None:
        ax.set_ylim(ylim)

    return fig, ax

def create_bar_plot_results_m1(user_df: DataFrame, model_df: DataFrame, ax=None, ylim=None):
    """Create bar plot results for manipulation 1."""
    return _reuse_bar_plot_results(
        user_df=user_df,
        model_df=model_df,
        manipulation=1,
        junction=(2, 14),
        maze_name='big_m1_maze3_shortcut',
        ax=ax,
        ylim=ylim
    )

def create_bar_plot_results_m3(user_df: DataFrame, model_df: DataFrame, ax=None, ylim=None):
    """Create bar plot results for manipulation 3."""
    return _reuse_bar_plot_results(
        user_df=user_df,
        model_df=model_df,
        manipulation=3,
        junction=(14, 25),
        maze_name='big_m3_maze1',
        ax=ax,
        ylim=ylim
    )

#########################################################
# Shortcut manipulation (3)
#########################################################


def plot_m1_example(user_df: DataFrame):

    old_path_cond = user_df.filter(manipulation=1, reuse=True, eval=True)
    new_path_cond = user_df.filter(manipulation=1, reuse=False, eval=True)

    # Create a figure with 3 subplots for render_path
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    housemaze_utils.render_path(old_path_cond.episodes[0], ax=axs[0])
    axs[0].set_title('Using prior path')

    housemaze_utils.render_path(new_path_cond.episodes[0], ax=axs[1])
    axs[1].set_title('Using new path')

    plt.tight_layout()
    plt.show()


def create_success_termination_results_m1(user_df: DataFrame, model_df: DataFrame, **kwargs):
    manipulation = 1
    model_setting = dict(maze_name='big_m1_maze3_shortcut', eval=True)

    def post_fn(x):
        return np.array(x)[0]
    def success_fn(e): return 100*success(e)
    def model_fn(e): return jax.vmap(success_fn)(e)

    m1_df = user_df.filter(manipulation=manipulation, eval=True)
    human_success = m1_df.group_by('user_id').agg(
        pl.col('success').mean()).select('success').to_numpy().flatten()

    # Success rate data
    data = {
        'human': human_success*100,
        # 'human_terminate': finished,
        'qlearning': model_df.apply(fn=model_fn, output_transform=post_fn, algo="qlearning", **model_setting),
        'usfa': model_df.apply(fn=model_fn, output_transform=post_fn, algo="usfa", **model_setting),
        'dyna': model_df.apply(fn=model_fn, output_transform=post_fn, algo="dynaq_shared", **model_setting),
        'bfs': model_df.apply(fn=model_fn, output_transform=post_fn, algo='bfs', **model_setting),
        'dfs': model_df.apply(fn=model_fn, output_transform=post_fn, algo='dfs', **model_setting),
    }

    bar_plot_results(
        data,
        # data_termination,
        title='Success Rate',
        # ylabel='Pr'
        error_bars=True,
        autolabel=False,
        **kwargs
    )


def create_bar_plot_results_m1(user_df: DataFrame, model_df: DataFrame, ax=None, ylim=None):
    """Create bar plot results for manipulation 1."""
    return _reuse_bar_plot_results(
        user_df=user_df,
        model_df=model_df,
        manipulation=1,
        junction=(2, 14),
        maze_name='big_m1_maze3_shortcut',
        ax=ax,
        ylim=ylim
    )

def create_bar_plot_results_m3(user_df: DataFrame, model_df: DataFrame, ax=None, ylim=None):
    """Create bar plot results for manipulation 3."""
    return _reuse_bar_plot_results(
        user_df=user_df,
        model_df=model_df,
        manipulation=3,
        junction=(14, 25),
        maze_name='big_m3_maze1',
        ax=ax,
        ylim=ylim
    )

def m1_reaction_times(user_df: DataFrame, **kwargs):
    manipulation = 1

    subset = user_df.filter_by_group(
        input_episode_filter=group_filter_fn,
        #output_episode_filter=lambda e: not success_or_not_terminate(e),
        output_episode_filter=lambda e: not success_or_not_terminate(e),
        input_settings=dict(eval=False),
        output_settings=dict(manipulation=manipulation, eval=True),
    )

    # Split episodes based on reuse column
    reuse_episodes = subset.filter(reuse=True).episodes
    no_reuse_episodes = subset.filter(reuse=False).episodes

    # Plot reaction times
    reaction_times_dual(
        no_reuse_episodes,
        reuse_episodes,
        label1='Used New Path',
        label2='Partially reused path',
        **kwargs
    )

#########################################################
# Starting point manipulation (2)
#########################################################

def plot_m2_example(user_df: DataFrame):
    subset = user_df.filter_by_group(
        input_episode_filter=group_filter_fn,
        output_episode_filter=lambda e: not success_or_not_terminate(e),
        input_settings=dict(eval=False),
        output_settings=dict(manipulation=2),
    )
    # SHOULD BE SMALLER
    cond1 = subset.filter(
        maze='big_m2_maze2_onpath_(F,F)',
        manipulation=2, eval=True, condition=1)  # On-path
    # SHOULD BE LARGER
    cond2 = subset.filter(
        maze='big_m2_maze2_offpath_(F,F)',
        manipulation=2, eval=True, condition=2)  # Off-path

    # Create a figure with 3 subplots for render_path
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    housemaze_utils.render_path(cond1.episodes[0], ax=axs[0])
    axs[0].set_title('Starting location on-path')

    housemaze_utils.render_path(cond2.episodes[0], ax=axs[1])
    axs[1].set_title('Starting location off-path')

    plt.tight_layout()
    plt.show()

def m2_reaction_time_difference(user_df: DataFrame, rt_types=['first', 'avg'], **kwargs):
    subset = user_df.filter_by_group(
        input_episode_filter=group_filter_fn,
        output_episode_filter=lambda e: not success_or_not_terminate(e),
        input_settings=dict(eval=False),
        output_settings=dict(manipulation=2),
    )
    # SHOULD BE SMALLER
    cond1 = subset.filter(manipulation=2, eval=True, condition=1)  # On-path
    # SHOULD BE LARGER
    cond2 = subset.filter(manipulation=2, eval=True, condition=2)  # Off-path

    reaction_times_difference(cond1, cond2, rt_types=rt_types, **kwargs)

#########################################################
# Planning manipulation (4)
#########################################################

def plot_m4_example(user_df: DataFrame, setting: str='short', version: str = 'regular'):
    assert setting in ['short', 'long']
    assert version in ['blind', 'regular']
    subset = user_df.filter_by_group(
        input_episode_filter=partial(group_filter_fn, min_successes=8),
        output_episode_filter=lambda e: not success_or_not_terminate(e),
        input_settings=dict(eval=False),
        output_settings=dict(manipulation=4),
    )
    if version == 'regular':
        cond0 = subset.filter(maze=f'big_m4_maze_{setting}_(F,F)')
        cond1 = subset.filter(maze=f'big_m4_maze_{setting}_eval_same_(F,F)')
        cond2 = subset.filter(maze=f'big_m4_maze_{setting}_eval_diff_(F,F)')
    elif version == 'blind':
        cond0 = subset.filter(maze=f'big_m4_maze_{setting}_blind_(F,F)')
        cond1 = subset.filter(maze=f'big_m4_maze_{setting}_eval_same_blind_(F,F)')
        cond2 = subset.filter(maze=f'big_m4_maze_{setting}_eval_diff_(F,F)')

    # Create a figure with 3 subplots for render_path
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Render paths on separate subplots
    housemaze_utils.render_path(cond0.episodes[1], ax=axs[0])
    axs[0].set_title("Training")

    housemaze_utils.render_path(cond1.episodes[1], ax=axs[1])
    axs[1].set_title('Same target location')

    housemaze_utils.render_path(cond2.episodes[0], ax=axs[2])
    axs[2].set_title('New target location')

    plt.tight_layout()
    plt.show()

def m4_initial_action_distribution(user_df: DataFrame, setting: str='short'):
    assert setting in ['short', 'long']
    subset = user_df.filter_by_group(
        input_settings=dict(eval=False, name=f'big_m4_maze_{setting}'),
        input_episode_filter=partial(group_filter_fn, min_successes=8),
        output_settings=dict(manipulation=4),
        output_episode_filter=lambda e: not success_or_not_terminate(e),
    )
    cond1 = subset.filter(name=f'big_m4_maze_{setting}_eval_same')
    cond2 = subset.filter(name=f'big_m4_maze_{setting}_eval_diff')

    if setting == 'short':
        key2model = {
            'right': 'Multitask preplay',
            'up': 'Model-free',
            'left': 'Planning',
        }
        action_indices = [
            KeyboardActions.left,
            KeyboardActions.right,
            KeyboardActions.up,
            KeyboardActions.down,
            ]
    else:
        key2model = {
            'left': 'Multitask preplay',
            'down': 'Model-free',
            'right': 'Planning',
        }
        action_indices = [
            KeyboardActions.left,
            KeyboardActions.right,
            KeyboardActions.down,
            KeyboardActions.up,
            ]

    model_colors = {
        'Model-free': '#CC79A7',
        'Multitask preplay': '#F0E442',
        'Planning': '#E69F00'
    }

    initial_action_distribution(
        cond1, cond2, key2model, model_colors, action_indices=action_indices)

def m4_reaction_times(user_df: DataFrame, setting='short', rt_type='speed', ylim=None):
    assert setting in ['short', 'long']
    assert rt_type in ['speed', 'first']

    def create_plots(output_filter_fn, title_suffix):
        subset = user_df.filter_by_group(
            input_episode_filter=partial(group_filter_fn, min_successes=8),
            output_episode_filter=output_filter_fn,
            input_settings=dict(
                name=f'big_m4_maze_{setting}',
            ),
            output_settings=dict(manipulation=4),
        )

        cond1 = subset.filter(name=f'big_m4_maze_{setting}_eval_diff')
        cond2 = subset.filter(name=f'big_m4_maze_{setting}_eval_same')

        if setting == 'short':
            key2model = {
                'right': 'Multitask preplay',
                'up': 'Model-free',
                'left': 'Planning',
            }
            action_indices = [
                KeyboardActions.left,
                KeyboardActions.right,
                KeyboardActions.up,
                KeyboardActions.down
            ]
        else:
            key2model = {
                'left': 'Multitask preplay',
                'down': 'Model-free',
                'right': 'Planning',
            }
            action_indices = [
                KeyboardActions.left,
                KeyboardActions.right,
                KeyboardActions.down,
                KeyboardActions.up,
                              ]

        model_colors = {
            'Model-free': '#CC79A7',
            'Multitask preplay': '#F0E442',
            'Planning': '#E69F00'
        }

        # Create a figure with 2 subplots
        fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
        fig.suptitle(
            f"{setting.capitalize()} Maze - {title_suffix} - {rt_type.capitalize()} Reaction Time", fontsize=DEFAULT_TITLE_SIZE)

        # Function to plot reaction times by initial action
        def plot_rt_by_action(ax, episodes, title):
            action_rts = {action: [] for action in action_indices}
            
            for episode in episodes:
                initial_action = int(episode.actions[0])
                if initial_action < 0: continue
                if rt_type == 'speed':
                    rt = avg_rt(episode)
                else:  # 'first'
                    rt = first_rt(episode)
                action_rts[initial_action].append(rt)

            data = [np.asarray(action_rts[action]) for action in action_indices]
            labels = [action.name for action in action_indices]

            # Create boxplot with custom colors
            box_colors = [model_colors.get(key2model.get(label.lower(), ''), '#333333') for label in labels]
            sns.boxplot(data=data, ax=ax, width=0.5, palette=box_colors)
            sns.stripplot(data=data, ax=ax, color='black',
                          alpha=0.5, jitter=True)

            ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)
            #ax.set_xlabel('Initial Action', fontsize=DEFAULT_LABEL_SIZE)
            ax.set_ylabel(f'{rt_type.capitalize()} Reaction Time (s)', fontsize=DEFAULT_LABEL_SIZE)
            ax.tick_params(axis='both', which='major', labelsize=DEFAULT_LABEL_SIZE)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            if ylim is not None:
                ax.set_ylim(*ylim)

            # Add model names below the action names
            for i, label in enumerate(labels):
                if label.lower() in key2model:
                    ax.text(i, ax.get_ylim()[0], key2model[label.lower()],
                            ha='center', va='top', rotation=45,
                            fontsize=DEFAULT_LABEL_SIZE-2, color=box_colors[i])

        plot_rt_by_action(axs[1], cond1.episodes, 'New target location')
        plot_rt_by_action(axs[0], cond2.episodes, 'Same target location')

        plt.tight_layout()
        plt.show()

    # Create plots for all episodes
    create_plots(None, "All Episodes")

def m4_action_reaction_times(
        user_df: DataFrame, setting='short', rt_type='speed'):
    assert setting in ['short', 'long']
    assert rt_type in ['speed', 'first']

    subset = user_df.filter_by_group(
        input_episode_filter=partial(group_filter_fn, min_successes=8),
        input_settings=dict(
            name=f'big_m4_maze_{setting}',
        ),
        output_settings=dict(manipulation=4),
    )


    if setting == 'short':
        action_cond1 = KeyboardActions.left
        action_cond2 = KeyboardActions.right
    else:
        action_cond1 = KeyboardActions.right
        action_cond2 = KeyboardActions.left

    cond1 = subset.filter(name=f'big_m4_maze_{setting}_eval_diff')
    cond2 = subset.filter(name=f'big_m4_maze_{setting}_eval_same')

    # Function to get reaction times for a specific action
    def get_rts_for_action(episodes, action):
        rts = []
        for episode in episodes:
            if int(episode.actions[0]) == action:
                if rt_type == 'speed':
                    rt = avg_rt(episode)
                else:  # 'first'
                    rt = first_rt(episode)
                rts.append(rt)
        return np.array(rts)

    # Get reaction times for planning actions in both conditions
    rts_cond1 = get_rts_for_action(cond1.episodes, action_cond1)
    rts_cond2 = get_rts_for_action(cond2.episodes, action_cond2)

    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot boxplots
    box_data = [rts_cond1, rts_cond2]
    labels = ['New target location', 'Same target location']
    sns.boxplot(data=box_data, ax=ax, width=0.5,
                palette=['#E69F00', '#E69F00'])
    sns.stripplot(data=box_data, ax=ax, color='black', alpha=0.5, jitter=True)

    # Customize the plot
    ax.set_title(f"{setting.capitalize()}",
                 fontsize=DEFAULT_TITLE_SIZE)
    ax.set_xlabel('Condition', fontsize=DEFAULT_LABEL_SIZE)
    ax.set_ylabel(f'{rt_type.capitalize()} Reaction Time (s)',
                  fontsize=DEFAULT_LABEL_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=DEFAULT_LABEL_SIZE)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Add statistical test
    _, p_value = stats.ttest_ind(rts_cond1, rts_cond2)
    ax.text(0.5, 0.95, f'p-value: {p_value:.4f}',
            transform=ax.transAxes, ha='center', va='top', fontsize=DEFAULT_LABEL_SIZE)

    # Add mean values on top of each box
    means = [np.mean(rts_cond1), np.mean(rts_cond2)]
    for i, mean in enumerate(means):
        ax.text(i, ax.get_ylim()[1], f'Mean: {mean:.2f}',
                ha='center', va='bottom', fontsize=DEFAULT_LABEL_SIZE)

    plt.tight_layout()
    plt.show()

def m4_reaction_time_difference(
        user_df: DataFrame,
        setting='short',
        rt_types=['first', 'avg'],
        extra_settings=None,
        version: str = 'regular',
        **kwargs):
    assert setting in ['short', 'long']
    if version == 'regular':
        input_settings = dict(eval=False, name=f'big_m4_maze_{setting}')
    elif version == 'blind':
        input_settings = dict(eval=False, name=f'big_m4_maze_{setting}_blind')
    else:
        raise ValueError(f"Unknown version: {version}")

    subset = user_df.filter_by_group(
        input_episode_filter=partial(group_filter_fn, min_successes=8),
        output_episode_filter=lambda e: not success_or_not_terminate(e),
        input_settings=input_settings,
        output_settings=dict(manipulation=4),
    )

    extra_settings = extra_settings or {}
    # SMALLER
    if version == 'regular':
        same_cond = subset.filter(name=f'big_m4_maze_{setting}_eval_same', **extra_settings)
    elif version == 'blind':
        same_cond = subset.filter(
            name=f'big_m4_maze_{setting}_eval_same_blind', **extra_settings)

    # LARGER
    diff_cond = subset.filter(name=f'big_m4_maze_{setting}_eval_diff', **extra_settings)

    return reaction_times_difference(
        same_cond, diff_cond,
        rt_types=rt_types, **kwargs)

def m4_action_reaction_time_difference(
        user_df: DataFrame,
        setting='short',
        rt_types=['first', 'avg'],
        **kwargs):
    assert setting in ['short', 'long']

    subset = user_df.filter_by_group(
        input_episode_filter=partial(group_filter_fn, min_successes=8),
        input_settings=dict(
            name=f'big_m4_maze_{setting}',
        ),
        output_settings=dict(manipulation=4),
    )

    if setting == 'short':
        same_cond_action = KeyboardActions.right
        diff_cond_action = KeyboardActions.left
    else:
        same_cond_action = KeyboardActions.right
        diff_cond_action = KeyboardActions.left


    same_cond = subset.filter(
        name=f'big_m4_maze_{setting}_eval_same',
        episode_filter=lambda e: e.actions[0] != same_cond_action
    )
    diff_cond = subset.filter(
        name=f'big_m4_maze_{setting}_eval_diff',
        episode_filter=lambda e: e.actions[0] != diff_cond_action
    )

    reaction_times_difference(
        cond1=same_cond,
        cond2=diff_cond,
        #label1=f'New target location\nchose {diff_cond_action.name}',
        #label2=f'Same target location\nchose {same_cond_action.name}',
        rt_types=rt_types,
        **kwargs)

    #reaction_times_difference(
    #    cond1=diff_cond.episodes,
    #    cond2=same_cond.episodes,
    #    label1=f'New target location\nchose {diff_cond_action.name}',
    #    label2=f'Same target location\nchose {same_cond_action.name}',
    #    rt_types=rt_types,
    #    **kwargs)

def plot_m3_episode_length_histogram(user_df: DataFrame, **kwargs):
    title_size = kwargs.pop('title_size', DEFAULT_TITLE_SIZE)
    label_size = kwargs.pop('label_size', DEFAULT_LABEL_SIZE)
    legend_size = kwargs.pop('legend_size', DEFAULT_LEGEND_SIZE)

    # Filter the data as specified
    subset = user_df.filter_by_group(
        input_episode_filter=partial(group_filter_fn, min_successes=8),
        input_settings=dict(eval=False, manipulation=3),
        output_settings=dict(eval=True, manipulation=3),
    )
    #subset = user_df.filter(eval=True, manipulation=3)

    # Calculate episode lengths
    episode_lengths = []
    for episode in subset.episodes:
        episode_length = total_rt(episode)
        episode_lengths.append(episode_length)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    n, bins, patches = ax.hist(episode_lengths, bins=30, edgecolor='black')

    # Customize the plot
    ax.set_title("Distribution of Episode Lengths (Manipulation 3)",
                 fontsize=title_size)
    ax.set_xlabel('Episode Length (seconds)', fontsize=label_size)
    ax.set_ylabel('Frequency', fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=label_size)
    ax.grid(True, alpha=0.3)

    # Add mean and median lines
    mean_length = np.mean(episode_lengths)
    median_length = np.median(episode_lengths)
    ax.axvline(mean_length, color='red', linestyle='dashed',
               linewidth=2, label=f'Mean: {mean_length:.2f}s')
    ax.axvline(median_length, color='green', linestyle='dashed',
               linewidth=2, label=f'Median: {median_length:.2f}s')

    ax.legend(fontsize=legend_size)

    plt.tight_layout()
    plt.show()

def m4_condition_reaction_times(
        user_df: DataFrame,
        rt_types=['first', 'avg'],
        filter_by_action: bool = False,
        axs=None,
        ylim=None,
        **kwargs):

    def get_conditions(setting):
        subset = user_df.filter_by_group(
            input_episode_filter=partial(group_filter_fn, min_successes=8),
            input_settings=dict(
                name=f'big_m4_maze_{setting}',
            ),
            output_settings=dict(manipulation=4),
        )
        if setting == 'short':
            same_cond_action = KeyboardActions.right
            diff_cond_action = KeyboardActions.left
        else:
            same_cond_action = KeyboardActions.right
            diff_cond_action = KeyboardActions.left

        if filter_by_action:
            def same_episode_filter(e): return e.actions[0] != same_cond_action
            def diff_episode_filter(e): return e.actions[0] != diff_cond_action
        else:
            same_episode_filter = None
            diff_episode_filter = None

        same_cond = subset.filter(
            name=f'big_m4_maze_{setting}_eval_same',
            episode_filter=same_episode_filter
        )
        diff_cond = subset.filter(
            name=f'big_m4_maze_{setting}_eval_diff',
            episode_filter=diff_episode_filter
        )
        return same_cond, diff_cond

    short_same_cond, short_diff_cond = get_conditions('short')
    long_same_cond, long_diff_cond = get_conditions('long')

    rt_data = {rt_type: {'short': [], 'long': []} for rt_type in rt_types}
    users = (set(short_same_cond['user_id'].unique()) &
             set(short_diff_cond['user_id'].unique()) &
             set(long_same_cond['user_id'].unique()) &
             set(long_diff_cond['user_id'].unique()))

    rt_functions = {'first': first_rt, 'avg': avg_rt, 'total': total_rt}

    def good_number(x):
        good = not np.isnan(x)
        good &= not np.isinf(x)
        return good

    for user in users:
        short_same_user = short_same_cond.filter(user_id=user)
        short_diff_user = short_diff_cond.filter(user_id=user)
        long_same_user = long_same_cond.filter(user_id=user)
        long_diff_user = long_diff_cond.filter(user_id=user)

        if len(short_same_user.episodes) > 0 and len(short_diff_user.episodes) > 0 and len(long_same_user.episodes) > 0 and len(long_diff_user.episodes) > 0:
            for rt_type in rt_types:
                assert len(short_same_user.episodes) == len(short_diff_user.episodes) == 1
                assert len(long_same_user.episodes) == len(long_diff_user.episodes) == 1
                rt_fn = rt_functions[rt_type]
                rt_short_same = rt_fn(short_same_user.episodes[0])
                rt_short_diff = rt_fn(short_diff_user.episodes[0])
                rt_long_same = rt_fn(long_same_user.episodes[0])
                rt_long_diff = rt_fn(long_diff_user.episodes[0])

                if all(good_number(rt) for rt in [rt_short_same, rt_short_diff, rt_long_same, rt_long_diff]):
                    rt_data[rt_type]['short'].append(rt_short_diff - rt_short_same)
                    rt_data[rt_type]['long'].append(rt_long_diff - rt_long_same)
                else:
                    print(f"Skipping {user}, short same: {rt_short_same:.2f}, short diff: {rt_short_diff:.2f}, long same: {rt_long_same:.2f}, long diff: {rt_long_diff:.2f}")

    # Create separate plots for each RT type
    n_plots = len(rt_types)
    if axs is None:
        fig, axs = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))

    # If there's only one plot, axs will be a single axis, not an array
    if n_plots == 1:
        axs = [axs]

    def plot_rt_comparison(ax, data, title, i=0):
        short_data, long_data = data['short'], data['long']
        
        # Prepare data for box plots
        box_data = [short_data, long_data]
        
        # Create box plots
        bp = ax.boxplot(box_data, positions=[1, 2], widths=0.6, patch_artist=True)
        
        # Customize box colors
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Add scatter plots for individual data points
        for i, dataset in enumerate(box_data):
            y = dataset
            x = np.random.normal(i + 1, 0.04, size=len(y))
            ax.plot(x, y, 'r.', alpha=0.2)
        
        # Add lines connecting same participant data
        for i in range(len(short_data)):
            ax.plot([1, 2], [short_data[i], long_data[i]], 'k-', alpha=0.1)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Short', 'Long'])
        ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)
        
        if rt_type == 'first':
            ylabel = "First Reaction Time Difference (seconds)"
        elif rt_type == 'avg':
            ylabel = "Average Reaction Time Difference (steps/second)"
        ax.set_ylabel(ylabel, fontsize=DEFAULT_LABEL_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=DEFAULT_LABEL_SIZE)

        # Set y-axis limits if provided
        if ylim is not None:
            if len(ylim) == len(rt_types):
                ax.set_ylim(*ylim[i])
            else:
                ax.set_ylim(*ylim)

        # Add a horizontal line at y=0
        ax.axhline(y=0, color='r', linestyle='--')

        # Add legend
        ax.legend([bp["boxes"][0], bp["boxes"][1]], ['Short', 'Long'], 
                  loc='upper right', fontsize=DEFAULT_LEGEND_SIZE)

    for i, (ax, rt_type) in enumerate(zip(axs, rt_types)):
        plot_rt_comparison(ax, rt_data[rt_type], f"{rt_type.capitalize()} Reaction Time Difference", i)

    plt.tight_layout()
    plt.show()
