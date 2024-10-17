from functools import partial
from typing import Callable, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats
from collections import defaultdict

from analysis.data_loading import EpisodeData
from analysis import housemaze
from nicewebrl.dataframe import DataFrame

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


def success(e: EpisodeData):
    rewards = e.timesteps.reward
    # return rewards
    assert rewards.ndim == 1, 'this is only defined over vector, e.g. 1 episode'
    success = rewards > .5
    return success.any().astype(np.float32)

def success_or_not_terminate(e: EpisodeData):
    terminated = e.timesteps.last().any()
    succeeded = success(e) > 0
    keep = not terminated or succeeded
    return keep


def went_to_junction(episode_data, junction=(0, 11)):
    # positions = episode_data.positions
    # if positions is None:
    positions = episode_data.timesteps.state.agent_pos
    match = jnp.array(junction) == positions
    match = (match).sum(-1) == 2  # both x and y matches
    return match.any().astype(jnp.float32)  # if any matched


def total_rt(e: EpisodeData):
    return sum(e.reaction_times[:-1]/1000.)

def avg_rt(e: EpisodeData):
    return np.mean(e.reaction_times[:-1])/1000.

def first_rt(e: EpisodeData):
    return e.reaction_times[0]/1000.

def get_ylim_without_outliers(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return max(0, lower_bound), upper_bound


def bar_plot_results(model_dict, figsize=(8, 4), error_bars=False, title="", ylabel=""):
    # Set up the plot style
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")

    # Prepare data for plotting
    models = list(model_dict.keys())
    values = [np.mean(arr) for arr in model_dict.values()]
    errors = [np.std(arr)/np.sqrt(len(arr))
              for arr in model_dict.values()] if error_bars else None

    # Create the bar plot with consistent colors
    bars = plt.bar([model_names.get(model, model) for model in models], values, yerr=errors,
                   capsize=5, color=[model_colors.get(model, '#333333') for model in models])

    # Customize the plot
    plt.title(title, fontsize=16)
    plt.xlabel("Data source", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha='right')

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

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

###################
# Manipulation-specific plots
###################

def plot_reaction_times_across_conditions(episodes1: List[EpisodeData], episodes2: List[EpisodeData], label1='group1', label2='group2'):
    rt_types = ['speed', 'first']
    rt_functions = [avg_rt, first_rt]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, rt_fn, rt_type in zip(axes, rt_functions, rt_types):
        group1_rts = np.array([rt_fn(e) for e in episodes1])
        group2_rts = np.array([rt_fn(e) for e in episodes2])

        # Create box plot with individual points
        box_data = [group1_rts, group2_rts]
        labels = [label1, label2]
        
        sns.boxplot(data=box_data, ax=ax, width=0.5, palette=['green', 'red'])
        sns.stripplot(data=box_data, ax=ax, color='black', alpha=0.5, jitter=True)

        ax.set_xticklabels(labels)
        ax.set_ylabel('Reaction Time')
        ax.set_title(f'{rt_type.capitalize()} Reaction Time')

    plt.tight_layout()
    plt.show()


def split_filter_fn(df: DataFrame, min_successes: int = 16):
    successes = df.apply(success)
    remove = True
    if len(successes) == 0:
        return 0, remove
    nsuccess = sum(successes)
    remove = nsuccess < min_successes
    return nsuccess, remove


def m2_reaction_times(user_df: DataFrame):
    subset = user_df.subset(
        filter_fn=split_filter_fn,
        output_filter_fn=lambda e: not success_or_not_terminate(e),
        filter_settings=dict(eval=False),
        output_settings=dict(manipulation=2),
    )
    cond1 = subset.filter(manipulation=2, eval=True, condition=1)
    cond2 = subset.filter(manipulation=2, eval=True, condition=2)

    # Create a figure with 3 subplots for render_path
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Render paths on separate subplots
    housemaze.render_path(subset.episodes[0], ax=axs[0])
    axs[0].set_title("Training")
    
    housemaze.render_path(cond1.episodes[0], ax=axs[1])
    axs[1].set_title("On-path")
    
    housemaze.render_path(cond2.episodes[0], ax=axs[2])
    axs[2].set_title("Off-path")

    plt.tight_layout()
    plt.show()

    # Plot reaction times
    plot_reaction_times_across_conditions(
        cond2.episodes,
        cond1.episodes,
        label2='On-path',
        label1='Off-path'
    )


def m2_reaction_time_difference(user_df: DataFrame):
    subset = user_df.subset(
        filter_fn=split_filter_fn,
        output_filter_fn=lambda e: not success_or_not_terminate(e),
        filter_settings=dict(eval=False),
        output_settings=dict(manipulation=2),
    )
    cond1 = subset.filter(manipulation=2, eval=True, condition=1)  # On-path
    cond2 = subset.filter(manipulation=2, eval=True, condition=2)  # Off-path

    # Compute RT difference for each user for both RT types
    rt_differences = {'first': [], 'avg': []}
    users = set(cond1['user_id'].unique()) & set(cond2['user_id'].unique())

    for user in users:
        cond1_user = cond1.filter(user_id=user)
        cond2_user = cond2.filter(user_id=user)

        if len(cond1_user.episodes) > 0 and len(cond2_user.episodes) > 0:
            for rt_type, rt_fn in [('first', first_rt), ('avg', avg_rt)]:
                rt_cond1 = np.mean([rt_fn(e) for e in cond1_user.episodes])
                rt_cond2 = np.mean([rt_fn(e) for e in cond2_user.episodes])
                rt_differences[rt_type].append(
                    rt_cond2 - rt_cond1)  # Off-path minus On-path

    # Plot the distribution of RT differences
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, rt_type in zip([ax1, ax2], ['first', 'avg']):
        sns.histplot(rt_differences[rt_type], kde=True, ax=ax)
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_title(
            f"{rt_type.capitalize()} RT Difference (Start Manipulation)")
        ax.set_xlabel("RT Difference (Off-path - On-path)")
        ax.set_ylabel("Count")

        # Calculate and display statistics
        positive_count = sum(diff > 0 for diff in rt_differences[rt_type])
        total_count = len(rt_differences[rt_type])
        positive_percentage = (positive_count / total_count) * 100

        ax.text(0.05, 0.95, f"Total users: {total_count}\n"
                f"Users with positive difference: {positive_count} ({positive_percentage:.2f}%)",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

def m4_reaction_times(user_df: DataFrame, setting='short'):
    subset = user_df.subset(
        filter_fn=partial(split_filter_fn, min_successes=8),
        output_filter_fn=lambda e: not success_or_not_terminate(e),
        #output_filter_fn=lambda e: not success(e),
        filter_settings=dict(
            maze='big_m4_maze_short' if setting == 'short' else 'big_m4_maze_long',
        ),
        output_settings=dict(manipulation=4),
    )
    if setting == 'short':
        cond0 = subset.filter(maze='big_m4_maze_short')
        cond1 = subset.filter(maze='big_m4_maze_short_eval_diff')
        cond2 = subset.filter(maze='big_m4_maze_short_eval_same')
    elif setting == 'long':
        cond0 = subset.filter(maze='big_m4_maze_long')
        cond1 = subset.filter(maze='big_m4_maze_long_eval_diff')
        cond2 = subset.filter(maze='big_m4_maze_long_eval_same')

    # Create a figure with 3 subplots for render_path
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Render paths on separate subplots
    housemaze.render_path(cond0.episodes[0], ax=axs[0])
    axs[0].set_title("Training")

    housemaze.render_path(cond1.episodes[0], ax=axs[1])
    axs[1].set_title('New target location')

    housemaze.render_path(cond2.episodes[0], ax=axs[2])
    axs[2].set_title('Same target location')

    plt.tight_layout()
    plt.show()

    # Plot reaction times
    plot_reaction_times_across_conditions(
        cond1.episodes,
        cond2.episodes,
        label1='New target location',
        label2='Same target location'
    )


def m4_reaction_time_difference(user_df: DataFrame, setting='short'):
    subset = user_df.subset(
        filter_fn=partial(split_filter_fn, min_successes=8),
        output_filter_fn=lambda e: not success_or_not_terminate(e),
        #output_filter_fn=lambda e: not success(e),
        filter_settings=dict(
            maze='big_m4_maze_short' if setting == 'short' else 'big_m4_maze_long',
        ),
        output_settings=dict(manipulation=4),
    )
    if setting == 'short':
        cond1 = subset.filter(maze='big_m4_maze_short_eval_diff')
        cond2 = subset.filter(maze='big_m4_maze_short_eval_same')
    elif setting == 'long':
        cond1 = subset.filter(maze='big_m4_maze_long_eval_diff')
        cond2 = subset.filter(maze='big_m4_maze_long_eval_same')
    else:
        raise ValueError("Setting must be 'short' or 'long'")

    # Compute RT difference for each user for both RT types
    rt_differences = {'first': [], 'avg': []}
    users = set(cond1['user_id'].unique()) & set(cond2['user_id'].unique())

    for user in users:
        cond1_user = cond1.filter(user_id=user)
        cond2_user = cond2.filter(user_id=user)

        if len(cond1_user.episodes) > 0 and len(cond2_user.episodes) > 0:
            for rt_type, rt_fn in [('first', first_rt), ('avg', avg_rt)]:
                rt_cond1 = np.mean([rt_fn(e) for e in cond1_user.episodes])
                rt_cond2 = np.mean([rt_fn(e) for e in cond2_user.episodes])
                rt_differences[rt_type].append((rt_cond1 - rt_cond2))

    # Plot the distribution of RT differences
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, rt_type in zip([ax1, ax2], ['first', 'avg']):
        sns.histplot(rt_differences[rt_type], kde=True, ax=ax)
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_title(
            f"{rt_type.capitalize()} ({setting.capitalize()} setting)")
        ax.set_xlabel("Time(New target location) - Time(Old target location)")
        ax.set_ylabel("Count")

        # Calculate and display statistics
        positive_count = sum(diff > 0 for diff in rt_differences[rt_type])
        total_count = len(rt_differences[rt_type])
        positive_percentage = (positive_count / total_count) * 100

        ax.text(0.05, 0.95, f"Total users: {total_count}\n"
                f"Users with positive difference: {positive_count} ({positive_percentage:.2f}%)",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


def m3_reaction_times(user_df: DataFrame, episode_selection='latest'):
    manipulation = 3

    subset = user_df.subset(
        filter_fn=split_filter_fn,
        output_filter_fn=lambda e: not success_or_not_terminate(e),
        filter_settings=dict(eval=False),
        output_settings=dict(manipulation=manipulation),
    )

    # Get the selected training episode (longest or latest)
    training_episodes = subset.filter(eval=False, room=0)
    selected_training_episodes = []
    for user_id in training_episodes['user_id'].unique():
        user_episodes = training_episodes.filter(user_id=user_id).episodes
        if user_episodes:
            if episode_selection == 'longest':
                selected_episode = max(user_episodes, key=lambda e: len(e.timesteps.reward))
            elif episode_selection == 'latest':
                selected_episode = user_episodes[-1]  # Assuming episodes are in chronological order
            else:
                raise ValueError("episode_selection must be 'longest' or 'latest'")
            selected_training_episodes.append(selected_episode)

    eval_episodes = subset.filter(eval=True).episodes
    old_path = partial(went_to_junction, junction=(14, 25))
    new_path = partial(went_to_junction, junction=(14, 0))
    old = [e for e in eval_episodes if old_path(e)]
    new = [e for e in eval_episodes if new_path(e)]

    # Create a figure with 3 subplots for render_path
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Render paths on separate subplots
    housemaze.render_path(selected_training_episodes[0], ax=axs[0])
    axs[0].set_title(f"{episode_selection.capitalize()} Training Episode")

    housemaze.render_path(new[0], ax=axs[1])
    axs[1].set_title("Used new path")

    housemaze.render_path(old[0], ax=axs[2])
    axs[2].set_title("Used old path")

    plt.tight_layout()
    plt.show()

    # Plot reaction times
    plot_reaction_times_across_conditions_m3(
        selected_training_episodes,
        new,
        old,
        label1=f'{episode_selection.capitalize()} Training',
        label2='Used new path',
        label3='Used old path'
    )


def plot_reaction_times_across_conditions_m3(episodes1: List[EpisodeData], episodes2: List[EpisodeData], episodes3: List[EpisodeData], label1='group1', label2='group2', label3='group3'):
    rt_types = ['speed', 'first']
    rt_functions = [avg_rt, first_rt]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, rt_fn, rt_type in zip(axes, rt_functions, rt_types):
        group1_rts = np.array([rt_fn(e) for e in episodes1])
        group2_rts = np.array([rt_fn(e) for e in episodes2])
        group3_rts = np.array([rt_fn(e) for e in episodes3])

        # Create box plot with individual points
        box_data = [group1_rts, group2_rts, group3_rts]
        labels = [label1, label2, label3]

        sns.boxplot(data=box_data, ax=ax, width=0.5,
                    palette=['blue', 'green', 'red'])
        sns.stripplot(data=box_data, ax=ax, color='black',
                      alpha=0.5, jitter=True)

        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Reaction Time')
        ax.set_title(f'{rt_type.capitalize()} Reaction Time')

    plt.tight_layout()
    plt.show()


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


def create_success_termination_results_m3(user_df, model_df):
    manipulation = 3
    model_setting = dict(maze_name='big_m3_maze1', eval=True)

    def post_fn(x):
        return np.array(x)[0]

    def get_human_data(fn):
        return user_df.split_apply(
            fn=fn,
            filter_fn=split_filter_fn,
            filter_settings=dict(manipulation=manipulation, eval=False),
            output_settings=dict(manipulation=manipulation, eval=True),
        )
    fn = success
    def model_fn(e): return jax.vmap(fn)(e)

    # Success rate data
    data = {
        'human_success': get_human_data(fn),
        'human_terminate': get_human_data(lambda e: e.timesteps.last().any()),
        'qlearning': model_df.apply(fn=model_fn, post_fn=post_fn, algo="qlearning", **model_setting),
        'dyna': model_df.apply(fn=model_fn, post_fn=post_fn, algo="dynaq_shared", **model_setting),
        'bfs': model_df.apply(fn=model_fn, post_fn=post_fn, algo='bfs', **model_setting),
        'dfs': model_df.apply(fn=model_fn, post_fn=post_fn, algo='dfs', **model_setting),
    }

    bar_plot_results(
        data,
        #data_termination,
        title='Success Rate',
        ylabel='Rate'
    )


def create_bar_plot_results_m3(user_df: DataFrame, model_df: DataFrame):
    manipulation = 3
    model_setting = dict(maze_name='big_m3_maze1', eval=True)
    fn = partial(went_to_junction, junction=(14, 25))
    #fn = partial(went_to_junction, junction=(17, 17))
    def model_fn(e): return jax.vmap(fn)(e)
    def post_fn(x):
        return np.array(x)[0]

    # Human data with first filter
    human_data_1 = user_df.split_apply(
        fn=fn,
        filter_fn=split_filter_fn,
        filter_settings=dict(manipulation=manipulation, eval=False),
        output_filter_fn=lambda e: not success_or_not_terminate(e),
        output_settings=dict(manipulation=manipulation, eval=True),
    )

    # Human data with second filter
    human_data_2 = user_df.split_apply(
        fn=fn,
        filter_fn=split_filter_fn,
        filter_settings=dict(manipulation=manipulation, eval=False),
        output_filter_fn=lambda e: not success(e),
        output_settings=dict(manipulation=manipulation, eval=True),
    )

    data = {
        'human': human_data_1,
        'human_success': human_data_2,
        'qlearning': model_df.apply(fn=model_fn, post_fn=post_fn, algo="qlearning", **model_setting),
        'usfa': model_df.apply(fn=model_fn, post_fn=post_fn, algo="usfa", **model_setting),
        'dyna': model_df.apply(fn=model_fn, post_fn=post_fn, algo="dynaq_shared", **model_setting),
        'bfs': model_df.apply(fn=model_fn, post_fn=post_fn, algo='bfs', **model_setting),
        'dfs': model_df.apply(fn=model_fn, post_fn=post_fn, algo='dfs', **model_setting),
    }
    figsize=(10, 6)
    title='Partially reused training path when shorter path exists'
    ylabel='Proportion'
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")

    error_bars=True 
    model_dict = data
    # Prepare data for plotting
    models = list(model_dict.keys())
    values = [np.mean(arr) for arr in model_dict.values()]
    errors = [np.std(arr)/np.sqrt(len(arr))
              for arr in model_dict.values()] if error_bars else None

    # Create the bar plot with consistent colors
    bars = plt.bar([model_names.get(model, model) for model in models], values, yerr=errors,
                   capsize=5, color=[model_colors.get(model, '#333333') for model in models])

    # Customize the plot
    plt.title(title, fontsize=16)
    plt.xlabel("Data source", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha='right')

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
