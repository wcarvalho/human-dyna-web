from typing import NamedTuple, List, Tuple

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

# from analysis.housemaze_analysis_garbarge import plot_rt_condition_differences
from housemaze.human_dyna import utils
from math import sqrt, ceil
from statsmodels.stats.power import TTestPower
import pandas as pd
import statsmodels.formula.api as smf
from multiprocessing import Pool
from glob import glob

from analysis.housemaze_model_data import get_model_data
from analysis.housemaze_user_data import get_human_data
from analysis.housemaze_user_data import get_valid_files
from nicewebrl.dataframe import DataFrame
import matplotlib.patches as mpatches

DEFAULT_TITLE_SIZE = 14
DEFAULT_LABEL_SIZE = 12
DEFAULT_LEGEND_SIZE = 10

image_dict = utils.load_image_dict()

default_colors = {
  "reddish purple": (204 / 255, 121 / 255, 167 / 255),
  "yellow": (240 / 255, 228 / 255, 66 / 255),
  "orange": (230 / 255, 159 / 255, 0.0),
  "vermillion": (213 / 255, 94 / 255, 0.0),
  "sky blue": (86 / 255, 180 / 255, 233 / 255),
  "bluish green": (0.0, 158 / 255, 115 / 255),
  "blue": (0.0, 114 / 255, 178 / 255),
  "black": "#2f2f2e",
  "dark gray": "#666666",
  "light gray": "#999999",
  "purple": "#CC79A7",
  "nice purple": "#9B80E6",
  "pretty blue": "#679FE5",
  "google blue": "#186CED",
  "google orange": "#FFB700",
  "white": "#FFFFFF",
}

model_colors = {
  #'human_success': '#0072B2',
  "human": default_colors["orange"],
  #'human_terminate': '#D55E00',
  "usfa": default_colors["light gray"],
  "qlearning": default_colors["dark gray"],
  "dynaq_shared": default_colors["vermillion"],
  "bfs": default_colors["pretty blue"],
  "dfs": default_colors["sky blue"],
}

model_names = {
  "human": "Human",
  "human_terminate": "Human (finished)",
  "human_success": "Human (Succeeded)",
  "qlearning": "Q-learning",
  "usfa": "Successor features",
  "dyna": "Dyna",
  "dynaq_shared": "Multi-task preplay",
  "bfs": "Breadth-first search",
  "dfs": "Depth-first search",
}

model_order = [
  "human",
  "human_success",
  "human_terminate",
  "dyna",
  "dynaq_shared",
  "qlearning",
  "usfa",
  "bfs",
  "dfs",
]

maze_name = {
  "big_m2_maze2": "Start Manipulation",
  "big_m2_maze2_offpath": "Start Manipulation: Off-path",
  "big_m2_maze2_onpath": "Start Manipulation: On-path",
  "big_m3_maze1": "Path Manipulation",
  "big_m3_maze1_eval": "Path Manipulation: Evaluation",
  "big_m4_maze_long": "Plan Manipulation (long)",
  "big_m4_maze_long_eval_same": "Plan Manipulation (long): Same location",
  "big_m4_maze_long_eval_diff": "Plan Manipulation (long): New location",
  "big_m4_maze_short": "Plan Manipulation (short)",
  "big_m4_maze_short_eval_same": "Plan Manipulation (short): Same location",
  "big_m4_maze_short_eval_diff": "Plan Manipulation (short): New location",
}

# for tqdm both in notebook and terminal
try:
  from IPython import get_ipython

  if "IPKernelApp" in get_ipython().config:
    from tqdm.notebook import tqdm

    try:
      import ipywidgets
    except:
      pass
  else:
    from tqdm import tqdm
except (ImportError, AttributeError):
  from tqdm import tqdm


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
  assert rewards.ndim == 1, "this is only defined over vector, e.g. 1 episode"
  success = rewards > 0.5
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

  user = df["user_id"].unique().to_list()[0]
  nsuccess = int(sum(successes))
  remove = nsuccess < min_successes
  if remove:
    print(
      f"removed: user {user} rate: {np.mean(successes)} = {nsuccess}/{min_successes}/{len(successes)}"
    )
  return remove


def filter_outliers(
  df: DataFrame,
  filter_columns: List[str],
  method: str = "iqr",
  verbose: bool = False,
  threshold: float = 1.5,
) -> DataFrame:
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

    if method == "iqr":
      q1 = np.percentile(values, 25)
      q3 = np.percentile(values, 75)
      iqr = q3 - q1
      lower_bound = q1 - threshold * iqr
      upper_bound = q3 + threshold * iqr

    elif method == "zscore":
      mean = np.mean(values)
      std = np.std(values)
      lower_bound = mean - threshold * std
      upper_bound = mean + threshold * std

    elif method == "percentile":
      lower_bound = np.percentile(values, threshold)
      upper_bound = np.percentile(values, 100 - threshold)

    else:
      raise ValueError(f"Unknown method: {method}")

    bounds[col] = {"lower": lower_bound, "upper": upper_bound}

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
    print(
      f"Filtered {n_filtered} outliers from {original_size} rows using {method} method"
    )
    for row in removed_df.iter_rows(named=True):
      user = str(row["user_id"])
      users_removed.add(user)
      for col in filter_columns:
        val = row[col]
        bound = bounds[col]
        if val < bound["lower"] or val > bound["upper"]:
          print(
            f"\t{user}. {col}:{val} \t(bounds: {bound['lower']} to {bound['upper']})"
          )
    print(f"Users removed: {users_removed}")

  return filtered_df


def bar_plot_error(
  human_data, model_stats=None, ax=None, ylim=None, legend=True, xlabels=False
):
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
  all_data = {"human": human_data["means"]}
  # Add model data
  yerr = [human_data["se"]]  # Start with human SE
  if model_stats is not None:
    algos = model_stats["algo"].unique().to_list()
    for algo in model_order:
      if not algo in algos:
        continue
      row = model_stats.filter(algo=algo)
      all_data[algo] = row["mean"].to_numpy()[0]
      yerr.append(row["se"].to_numpy()[0])

  # Plot bars
  x_pos = np.arange(len(all_data))
  ordered_keys = [k for k in model_order if k in all_data]
  bars = ax.bar(
    x_pos,
    [all_data[k] for k in ordered_keys],
    yerr=yerr,
    capsize=5,
    color=[model_colors.get(k, "#333333") for k in ordered_keys],
  )

  # Update x-tick labels to match new ordering
  if xlabels:
    ax.set_xticks(x_pos)
    ax.set_xticklabels([model_names[k] for k in ordered_keys], rotation=45, ha="right")
  else:
    ax.set_xticks([])

  # Add individual dots for human data with jitter
  if "raw" in human_data:
    x_jitter = np.random.normal(0, 0.15, size=len(human_data["raw"]))
    ax.scatter(
      [0 + j for j in x_jitter],
      human_data["raw"],
      color="black",
      alpha=0.5,
      zorder=3,
    )

  # Add chance level line
  ax.axhline(y=50, color="r", linestyle="--", alpha=0.5, label="Chance level")

  if ylim is not None:
    ax.set_ylim(ylim)

  if legend:
    legend_elements = bars
    legend_labels = [model_names[k] for k in ordered_keys]
    ax.legend(legend_elements, legend_labels, loc="lower right")

  ax.grid(True, linestyle="--", alpha=0.7)

  return fig, ax


def plot_bar_rt_comparison(
  df,
  rt_column,
  ax=None,
  title=None,
  ylabel=None,
  xlabels=None,
  colors=None,
  stats_file=None,
  percentile_ylim: bool = True,
  n_simulations: int = 500,
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

  power_results = power_analysis_rt_across_groups(
    df, measure=rt_column, stats_file=stats_file, n_simulations=n_simulations
  )
  means = (
    power_results["descriptive"]["means"]["no_reuse"],
    power_results["descriptive"]["means"]["reuse"],
  )
  sems = (
    power_results["descriptive"]["ses"]["no_reuse"],
    power_results["descriptive"]["ses"]["reuse"],
  )

  # Create bar plot
  x_pos = np.arange(len(means))
  ax.bar(x_pos, means, yerr=sems, capsize=5, color=colors[: len(means)])

  # Add individual points with jitter
  all_data = []
  for i, key in enumerate(["no_reuse", "reuse"]):
    data = power_results["raw_means"][key]
    x_jitter = np.random.normal(i, 0.04, size=len(data))
    ax.scatter(x_jitter, data, alpha=0.3, color="black", s=20)
    all_data.append(data)

  # Customize plot
  ax.set_xticks(x_pos)
  if xlabels:
    ax.set_xticklabels(xlabels, ha="center")
  ax.set_ylabel(ylabel or f"Log {rt_column}", fontsize=DEFAULT_LABEL_SIZE)
  ax.set_title(title or f"{rt_column} Comparison", fontsize=DEFAULT_TITLE_SIZE)
  ax.tick_params(axis="both", which="major", labelsize=DEFAULT_LABEL_SIZE)
  ax.grid(True, linestyle="--", alpha=0.7)

  if percentile_ylim:
    all_data_combined = np.concatenate(all_data)
    y_min, y_max = np.percentile(all_data_combined, [1, 99])
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

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
    log_col = f"log_{col}"
    stats = (
      df.with_columns((1000 * pl.col(col)).log().alias(log_col))
      .group_by("user_id")
      .agg(pl.col(log_col).mean())
    )
    all_data.append(stats[log_col].to_numpy())

  # Calculate means and standard errors
  means = [np.mean(data) for data in all_data]
  sems = [np.std(data) / np.sqrt(len(data)) for data in all_data]

  # Create bar plot
  x_pos = np.arange(len(rt_columns))
  bars = ax.bar(x_pos, means, yerr=sems, capsize=5, color=colors[: len(rt_columns)])

  # Add individual points with jitter
  for i, data in enumerate(all_data):
    x_jitter = np.random.normal(i, 0.04, size=len(data))
    ax.scatter(x_jitter, data, alpha=0.3, color="black", s=20)

  # Customize plot
  ax.set_xticks(x_pos)
  if xlabels:
    ax.set_xticklabels(xlabels, ha="center")
  else:
    ax.set_xticklabels(rt_columns, ha="center")
  ax.set_ylabel(ylabel or "Log Reaction Time", fontsize=DEFAULT_LABEL_SIZE)
  ax.set_title(title or "Reaction Time Comparison", fontsize=DEFAULT_TITLE_SIZE)
  ax.tick_params(axis="both", which="major", labelsize=DEFAULT_LABEL_SIZE)
  ax.grid(True, linestyle="--", alpha=0.7)

  return ax


def plot_rt_differences(
  difference_df: pl.DataFrame,
  measures: List[str],
  title: str,
  ax: plt.Axes = None,
  colors=None,
  ylabel="Log RT Difference (Cond2 - Cond1)",
  stats_file=None,
  xlabels=None,
) -> Tuple[plt.Figure, plt.Axes]:
  """Plot RT differences between conditions.

  First compute a power analysis for each measure, seeing if its statistically significantly above 0.
  - note we have repeated measures per user, corresponding to the 'reversal column'.
  """
  # Calculate statistics for each measure
  means = []
  sems = []
  all_diffs = []
  for measure in measures:
    results = power_analysis_rt_differences(
      difference_df, measure, stats_file=stats_file
    )
    means.append(results["mean"])
    sems.append(results["se"])
    all_diffs.append(difference_df[measure].to_numpy())

  # Create/get axis
  if ax is None:
    fig, ax = plt.subplots(figsize=(5, 4))
  else:
    fig = ax.figure

  # Create bar plot
  x_pos = np.arange(len(measures))
  bars = ax.bar(
    x_pos,
    means,
    yerr=sems,
    capsize=5,
    color=colors,
    error_kw=dict(ecolor=default_colors["vermillion"]),
  )

  # Add individual points with jitter
  for i, diffs in enumerate(all_diffs):
    x_jitter = np.random.normal(i, 0.125, size=len(diffs))
    ax.scatter(x_jitter, diffs, alpha=0.3, color="black", s=20)

  # Add zero line
  ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)

  # Customize plot
  ax.set_xticks(x_pos)
  ax.set_xticklabels(xlabels or measures, ha="center")
  ax.set_ylabel(ylabel, fontsize=DEFAULT_LABEL_SIZE)
  if title:
    ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)
  ax.tick_params(axis="both", which="major", labelsize=DEFAULT_LABEL_SIZE)
  ax.grid(True, linestyle="--", alpha=0.7)

  # Set y-axis limits based on all data points
  all_data = np.concatenate(all_diffs)
  y_min, y_max = np.percentile(all_data, [1, 99])
  y_range = y_max - y_min
  ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

  return fig, ax


def plot_success_rate_comparison(
  df: DataFrame,
  model_df: DataFrame,
  ax=None,
  title="Success Rate Comparison",
  include_raw_data: bool = False,
  figsize=(6, 3),
) -> Tuple[plt.Figure, plt.Axes]:
  """Plot success rate comparison between human and model data.

  Args:
      df (DataFrame): DataFrame containing human data
      model_df (DataFrame): DataFrame containing model data
      ax (plt.Axes, optional): Matplotlib axes to plot on. If None, creates new figure
      title (str, optional): Plot title. Defaults to "Success Rate Comparison"
      include_raw_data (bool, optional): Whether to include individual human data points.
          Defaults to False
      figsize (tuple, optional): Figure size if creating new figure. Defaults to (6,3)

  Returns:
      tuple: (fig, ax) containing the matplotlib figure and axes objects
  """
  # Calculate human statistics
  human_successes = (
    df.group_by("user_id")
    .agg(pl.col("success").mean())
    .select("success")
    .to_numpy()
    .flatten()
  )
  # Calculate mean and standard error across users
  human_mean = np.mean(human_successes)
  human_se = np.sqrt((human_mean * (1 - human_mean)) / len(human_successes))

  # Calculate model statistics
  model_stats = model_df.group_by("algo").agg(
    mean=pl.col("success").mean() * 100,
    se=(
      pl.col("success").mean()
      * (1 - pl.col("success").mean())
      / pl.col("success").count()
    ).sqrt()
    * 100,
  )

  # Create figure if needed
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)
  else:
    fig = ax.figure

  # Create plot
  human_data = {
    "means": 100 * human_mean,
    "se": 100 * human_se,
  }
  if include_raw_data:
    human_data["raw"] = human_successes

  bar_plot_error(
    human_data=human_data,
    model_stats=model_stats,
    ax=ax,
    legend=True,
  )

  # Customize plot
  ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)
  ax.set_ylabel("Success Rate (%)", fontsize=DEFAULT_LABEL_SIZE)

  return fig, ax


def plot_path_reuse_comparison(
  df: DataFrame,
  model_df: DataFrame,
  stats_file=None,
  ax=None,
  measure: str = "reuse",
  title="Path Reuse Comparison",
  ylabel="Path Reuse (%)",
  include_raw_data: bool = True,
  figsize=(6, 3),
) -> Tuple[plt.Figure, plt.Axes]:
  """Plot path reuse comparison between human and model data with statistical analysis.

  Args:
      df (DataFrame): DataFrame containing human data
      model_df (DataFrame): DataFrame containing model data
      stats_file: Optional file handle for writing statistical analysis
      ax (plt.Axes, optional): Matplotlib axes to plot on. If None, creates new figure
      title (str, optional): Plot title. Defaults to "Path Reuse Comparison"
      include_raw_data (bool, optional): Whether to include individual human data points.
          Defaults to True
      figsize (tuple, optional): Figure size if creating new figure. Defaults to (6,3)

  Returns:
      tuple: (fig, ax) containing the matplotlib figure and axes objects
  """

  results = power_analysis_path_reuse(
    df, mu=0.5, alpha=0.05, plot=False, stats_file=stats_file
  )

  # Calculate model statistics
  model_stats = model_df.group_by("algo").agg(
    mean=pl.col(measure).mean() * 100,
    se=(
      pl.col(measure).mean() * (1 - pl.col(measure).mean()) / pl.col(measure).count()
    ).sqrt()
    * 100,
  )

  # Create figure if needed
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)
  else:
    fig = ax.figure

  # Create plot
  human_data = {
    "means": 100 * results["mean"],
    "se": 100 * results["se"],
  }
  if include_raw_data:
    human_data["raw"] = 100 * results["reuse_rates"]

  bar_plot_error(
    human_data=human_data,
    model_stats=model_stats,
    ax=ax,
    legend=False,
  )

  # Customize plot
  ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)
  ax.set_ylabel(ylabel, fontsize=DEFAULT_LABEL_SIZE)

  return fig, ax


def compute_condition_difference_df(
  df: pl.DataFrame, measures: List[str]
) -> pl.DataFrame:
  """Compute differences between conditions 1 and 2 for matched reversal conditions.

  Args:
      df: DataFrame containing columns [condition, reversal, setting, {measures}]
      setting: Which setting to filter for ('short' or 'long')
      measures: List of measure column names to compute differences for

  Returns:
      DataFrame with columns [user_id, reversal, diff_{measure1}, diff_{measure2}, ...]
  """

  # Get condition 1 and 2 data separately
  cond1_df = df.filter(condition=1)
  cond2_df = df.filter(condition=2)

  # Join the dataframes on user_id and reversal to match pairs
  diff_df = cond1_df.join(cond2_df, on=["user_id", "reversal"], suffix="_cond2")

  # Compute differences for each measure
  diff_exprs = [
    (pl.col(f"{measure}_cond2") - pl.col(measure)).alias(measure)
    for measure in measures
  ]

  # Select user_id, reversal and all difference columns
  diff_df = diff_df.select(["user_id", "reversal"] + diff_exprs)

  return diff_df


# define functions to run power analysis for linear mixed effects model (LME)


def simulate_mixed_effects_trial(args):
  """Simulate a single mixed effects trial and test for significance.

  Args:
      args: Tuple containing:
          - num_subjects: Number of subjects in simulation
          - trials_per_subject: Number of trials per subject
          - B0: Intercept coefficient
          - B1: Slope coefficient
          - random_effect_var: Variance of random effects
          - residual_var: Variance of residuals
          - alpha: Significance level for test

  Returns:
      bool: True if result is significant at alpha level
  """
  # Unpack arguments
  num_subjects, trials_per_subject, B0, B1, random_effect_var, residual_var, alpha = (
    args
  )

  # Generate data
  user_ids = np.repeat(range(num_subjects), trials_per_subject)
  reuse = np.random.choice([0, 1], num_subjects * trials_per_subject)
  random_intercepts = np.random.normal(0, np.sqrt(random_effect_var), num_subjects)
  random_intercepts = np.repeat(random_intercepts, trials_per_subject)
  residuals = np.random.normal(
    0, np.sqrt(residual_var), num_subjects * trials_per_subject
  )

  # Generate response variable
  RT = B0 + B1 * reuse + random_intercepts + residuals

  # Create and test model
  data = pd.DataFrame({"RT": RT, "reuse": reuse, "user_id": user_ids})
  model = smf.mixedlm("RT ~ reuse", data, groups=data["user_id"])
  result = model.fit(reml=True)

  return result.pvalues["reuse"] < alpha


def mixed_effects_compute_power(
  num_subjects: int,
  trials_per_subject: int,
  B0: float,
  B1: float,
  random_effect_var: float,
  residual_var: float,
  n_simulations: int = 500,
  alpha: float = 0.05,
  parallel: bool = False,
  n_jobs: int = -1,
  verbose: bool = False,
) -> float:
  """Compute power for mixed effects model using parallel or sequential processing.

  Args:
      num_subjects: Number of subjects in each simulation
      trials_per_subject: Number of trials per subject
      B0: Intercept coefficient
      B1: Slope coefficient
      random_effect_var: Variance of random effects
      residual_var: Variance of residuals
      n_simulations: Number of simulations to run (default: 500)
      alpha: Significance level (default: 0.05)
      parallel: Whether to use parallel processing (default: True)
      n_jobs: Number of processes to use if parallel (-1 for all cores, default: -1)
      **kwargs: Additional arguments to pass to simulate_mixed_effects_trial

  Returns:
      float: Computed power (proportion of significant results)
  """
  # Prepare simulation parameters
  sim_args = [
    (
      num_subjects,
      trials_per_subject,
      B0,
      B1,
      random_effect_var,
      residual_var,
      alpha,
    )
  ] * n_simulations

  if parallel:
    # Use all available cores if n_jobs is -1
    n_jobs = None if n_jobs == -1 else n_jobs

    # Run simulations in parallel
    with Pool(processes=n_jobs) as pool:
      results = list(
        tqdm(
          pool.imap(simulate_mixed_effects_trial, sim_args),
          total=n_simulations,
          desc="Simulating data",
          disable=not verbose,
        )
      )
  else:
    # Run simulations sequentially
    results = [
      simulate_mixed_effects_trial(args)
      for args in tqdm(sim_args, desc="Simulating data")
    ]

  return sum(results) / n_simulations


######################################
# Power Analysis function
######################################


def power_analysis_path_reuse(
  df: pl.DataFrame,
  measure: str = "reuse",
  mu: float = 0.5,
  alpha: float = 0.05,
  plot: bool = False,
  stats_file=None,
):
  """Analyze binary proportion data using appropriate statistical test based on normality.

  Args:
      df: DataFrame containing reuse column (binary 0/1) and user_id
      mu: null hypothesis value (default: 0.5)
      alpha: significance level for tests (default: 0.05)
      plot: whether to show diagnostic plots (default: False)
      stats_file: optional file handle to write stats output
  """
  # First aggregate by user to get their mean reuse rate
  user_means = df.group_by("user_id").agg(
    reuse_rate=pl.col(measure).mean(), n_trials=pl.col(measure).count()
  )
  reuse_rates = user_means["reuse_rate"].to_numpy()
  n_users = len(reuse_rates)
  n_trials = user_means["n_trials"].to_numpy()

  # Test normality using Shapiro-Wilk test
  _, normality_p = stats.shapiro(reuse_rates)
  is_normal = normality_p > alpha

  # Calculate mean and standard error
  p_obs = np.mean(reuse_rates)
  se = np.std(reuse_rates, ddof=1) / np.sqrt(n_users)

  if is_normal:
    # One-sided t-test
    t_stat, p_value = stats.ttest_1samp(reuse_rates, mu)
    # Convert to one-sided p-value if t-statistic is in predicted direction
    p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2

    # Calculate Cohen's d effect size
    d = (p_obs - mu) / np.std(reuse_rates, ddof=1)
    effect_size = {"name": "Cohen's d", "value": d}

    test_name = "One-sample t-test"
    test_stat = t_stat

  else:
    # One-sided Wilcoxon signed-rank test
    w_stat, p_value = stats.wilcoxon(reuse_rates - mu, alternative="greater")

    # Calculate r effect size (correlation coefficient) for Wilcoxon test
    z = stats.norm.ppf(1 - p_value)  # Convert p-value to Z score
    r = z / np.sqrt(n_users)  # Standardize by sample size
    effect_size = {"name": "r", "value": r}

    test_name = "Wilcoxon signed-rank test"
    test_stat = w_stat

  # Print summary
  summary = f"\nAnalysis Results:\n"
  summary += f"N = {n_users} participants (mean {np.mean(n_trials):.1f} trials per participant)\n"
  summary += f"Mean reuse rate: {p_obs:.3f} (SE: {se:.3f})\n\n"
  summary += f"Normality Test (Shapiro-Wilk):\n"
  summary += f"p = {normality_p:.3f} ({'Normal' if is_normal else 'Non-normal'} distribution)\n\n"
  summary += f"{test_name}:\n"
  summary += f"statistic = {test_stat:.3f}\n"
  summary += f"p = {p_value:.3f}\n\n"
  summary += f"Effect Size ({effect_size['name']}):\n"
  summary += f"{effect_size['value']:.3f}\n"

  # Calculate required sample sizes for different power levels
  power_levels = [0.8, 0.9, 0.95]
  summary += "\nRequired Sample Sizes:\n"

  if is_normal:
    # For t-test
    effect = effect_size["value"]  # Cohen's d
    for power in power_levels:
      analysis = TTestPower()
      n_required = analysis.solve_power(
        effect_size=effect, alpha=alpha, power=power, alternative="larger"
      )
      summary += f"Power {power * 100:g}%: N ≥ {ceil(n_required)} participants\n"
  else:
    # For Wilcoxon test (based on asymptotic relative efficiency)
    # Wilcoxon test is ~95% as efficient as t-test
    effect = effect_size["value"]  # r score
    # Convert r to d using formula: d = 2r/sqrt(1-r^2)
    d = 2 * effect / sqrt(1 - effect**2) if abs(effect) < 1 else float("inf")
    for power in power_levels:
      analysis = TTestPower()
      n_required = analysis.solve_power(
        effect_size=d, alpha=alpha, power=power, alternative="larger"
      )
      # Adjust for Wilcoxon efficiency
      n_required = ceil(n_required / 0.95)
      summary += f"Power {power * 100:g}%: N ≥ {n_required} participants\n"

  if stats_file:
    stats_file.write(summary)

  if plot:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram with density plot
    sns.histplot(data=user_means, x="reuse_rate", kde=True, ax=ax1)
    ax1.axvline(mu, color="r", linestyle="--", label=f"Null (μ={mu})")
    ax1.set_title("Distribution of User Reuse Rates")
    ax1.set_xlabel("Reuse Rate")
    ax1.legend()

    # Q-Q plot
    stats.probplot(reuse_rates, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot")

    plt.tight_layout()
    plt.show()

  return {
    "n_users": n_users,
    "mean_trials": np.mean(n_trials),
    "reuse_rates": reuse_rates,
    "mean": p_obs,
    "se": se,
    "normality": {"is_normal": is_normal, "p_value": normality_p},
    "test": {"name": test_name, "statistic": test_stat, "p_value": p_value},
    "effect_size": effect_size,
  }


def power_analysis_rt_across_groups(
  df: pl.DataFrame, measure: str, alpha=0.05, stats_file=None, n_simulations=500
):
  """Perform power analysis for between-groups comparison using linear mixed effects model.

  Args:
      df: DataFrame with columns [user_id, reuse, rt] where:
          - user_id: identifier for each participant
          - reuse: boolean indicating condition
          - rt: reaction time measurement (in log seconds)
      alpha: Significance level (default: 0.05)
      stats_file: Optional file handle to write stats output

  Returns:
      dict containing analysis results
  """
  # Convert to pandas for statsmodels compatibility
  data = df._df.select(["user_id", "reuse", measure]).to_pandas()
  data.columns = ["user_id", "reuse", "RT"]

  # Fit linear mixed effects model
  model = smf.mixedlm("RT ~ reuse", data, groups=data["user_id"])
  result = model.fit(reml=True)

  # Calculate descriptive statistics by group
  user_stats = df.group_by(["user_id", "reuse"]).agg(mean_val=pl.col(measure).mean())

  reuse_means = user_stats.filter(pl.col("reuse") == True)["mean_val"].to_numpy()
  no_reuse_means = user_stats.filter(pl.col("reuse") == False)["mean_val"].to_numpy()

  n1, n2 = len(no_reuse_means), len(reuse_means)
  mean1, mean2 = np.mean(no_reuse_means), np.mean(reuse_means)
  var1, var2 = np.var(no_reuse_means, ddof=1), np.var(reuse_means, ddof=1)

  # Get effect size (standardized coefficient)
  param_name = "reuse[T.True]" if "reuse[T.True]" in result.params else "reuse"
  effect_size = result.params[param_name] / np.std(data["RT"])

  # Calculate required sample sizes for different power levels
  power_levels = [0.8, 0.9, 0.95]
  n_required = {}

  # Binary search for each power level
  for target_power in tqdm(power_levels, desc="Power levels"):
    left = 10  # minimum sample size
    right = 200  # maximum sample size to try

    while left < right:
      n = (left + right) // 2
      power = mixed_effects_compute_power(
        num_subjects=n,
        trials_per_subject=len(data) // len(data["user_id"].unique()),
        B0=result.params["Intercept"],
        B1=result.params[param_name],
        random_effect_var=result.cov_re.iloc[0, 0],
        residual_var=result.scale,
        n_simulations=n_simulations,
      )

      if abs(power - target_power) < 0.01:  # within 1% of target
        break
      elif power < target_power:
        left = n + 1
      else:
        right = n - 1

    n_required[target_power] = n

  # Calculate actual power with current sample size
  current_power = mixed_effects_compute_power(
    num_subjects=len(data["user_id"].unique()),
    trials_per_subject=len(data) // len(data["user_id"].unique()),
    B0=result.params["Intercept"],
    B1=result.params[param_name],
    random_effect_var=result.cov_re.iloc[0, 0],
    residual_var=result.scale,
    n_simulations=n_simulations,
  )

  results = {
    "effect_size": {"name": "Standardized coefficient", "value": effect_size},
    "n_required": n_required,
    "current_power": current_power,
    "test_results": {
      "name": "Linear mixed effects model",
      "statistic": result.tvalues[param_name],
      "p_value": result.pvalues[param_name],
      "n1": n1,
      "n2": n2,
    },
    "descriptive": {
      "means": {"no_reuse": mean1, "reuse": mean2},
      "sds": {"no_reuse": np.sqrt(var1), "reuse": np.sqrt(var2)},
      "ses": {"no_reuse": np.sqrt(var1 / n1), "reuse": np.sqrt(var2 / n2)},
    },
    "raw_means": {"no_reuse": no_reuse_means, "reuse": reuse_means},
  }

  if stats_file:
    stats_file.write("\nLinear Mixed Effects Model Results:\n")
    stats_file.write("================================\n")
    stats_file.write(str(result.summary()) + "\n\n")

    stats_file.write("Sample Sizes:\n")
    stats_file.write(f"\tNo Reuse: {n1} users\n")
    stats_file.write(f"\tReuse: {n2} users\n")
    stats_file.write(
      f"\tTrials per user: {len(data) // len(data['user_id'].unique())}\n\n"
    )

    stats_file.write("Means:\n")
    stats_file.write(f"\tNo Reuse: {mean1:.3f}\n")
    stats_file.write(f"\tReuse: {mean2:.3f}\n")
    stats_file.write(f"\tDifference: {mean2 - mean1:.3f}\n\n")

    stats_file.write(f"Effect size: {effect_size:.3f}\n\n")

    stats_file.write("Power Analysis:\n")
    stats_file.write(f"Current power: {current_power:.3f}\n")
    for power, n in n_required.items():
      stats_file.write(f"Required sample size for {power * 100}% power: {n}\n")

  return results


def power_analysis_rt_differences(
  difference_df: pl.DataFrame, measure: str, alpha: float = 0.05, stats_file=None
) -> dict:
  """Analyze RT differences between conditions with appropriate statistical tests.

  Args:
      difference_df: DataFrame containing RT differences and user/reversal info
      measure: Name of RT measure column being analyzed
      alpha: significance level for tests (default: 0.05)
      stats_file: optional file handle to write stats output

  Returns:
      dict containing test results and effect size
  """
  # Get mean difference per user (averaging across reversals)
  user_means = (
    difference_df.group_by("user_id").agg(pl.col(measure).mean()).select(measure)
  )
  differences = user_means.to_numpy().flatten()

  n = len(differences)

  # Test normality using Shapiro-Wilk test
  _, normality_p = stats.shapiro(differences)
  is_normal = normality_p > alpha

  # Calculate mean and standard error
  mean = np.mean(differences)
  se = np.std(differences, ddof=1) / np.sqrt(n)

  if stats_file:
    stats_file.write(f"\n{measure}:\n")
    stats_file.write("=" * (len(measure) + 10) + "\n")
    stats_file.write(f"N = {n} participants\n")
    stats_file.write(f"Mean difference = {mean:.3f} (SE: {se:.3f})\n\n")
    stats_file.write("Normality Test (Shapiro-Wilk):\n")
    stats_file.write(
      f"p = {normality_p:.3f} ({'Normal' if is_normal else 'Non-normal'} distribution)\n\n"
    )

  if is_normal:
    # One-sided paired t-test (testing if condition 1 < condition 2)
    t_stat, p_value = stats.ttest_rel(differences, np.zeros_like(differences))
    # Convert to one-sided p-value if t-statistic is in predicted direction
    p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2

    # Calculate Cohen's d effect size for paired differences
    d = mean / np.std(differences, ddof=1)
    effect_size = {"name": "Cohen's d", "value": d}

    test_name = "Paired t-test"
    test_stat = t_stat

    # Power analysis for paired t-test
    analysis = TTestPower()
    actual_power = analysis.power(
      effect_size=abs(d), nobs=n, alpha=alpha, alternative="larger"
    )

    # Calculate required sample sizes for different power levels
    power_levels = [0.8, 0.9, 0.95]
    required_n = {}
    for power in power_levels:
      n_required = analysis.solve_power(
        effect_size=abs(d), alpha=alpha, power=power, alternative="larger"
      )
      required_n[power] = ceil(n_required)

  else:
    # One-sided Wilcoxon signed-rank test
    w_stat, p_value = stats.wilcoxon(differences, alternative="greater")
    test_name = "Wilcoxon signed-rank test"
    test_stat = w_stat

    # Calculate r effect size for Wilcoxon test
    # Convert p-value to z-score using inverse normal CDF
    z = stats.norm.ppf(1 - p_value)  # One-sided p-value
    r = z / np.sqrt(n)  # Standardize by sample size
    effect_size = {"name": "r", "value": r}

    # Convert r to d for power analysis
    # Formula: d = 2r/sqrt(1-r^2)
    d = 2 * r / sqrt(1 - r**2) if abs(r) < 1 else float("inf")

    # Power analysis using t-test as approximation (with 95% efficiency adjustment)
    analysis = TTestPower()
    actual_power = (
      analysis.power(effect_size=abs(d), nobs=n, alpha=alpha, alternative="larger")
      * 0.95
    )  # Adjust for Wilcoxon efficiency

    # Calculate required sample sizes for different power levels
    power_levels = [0.8, 0.9, 0.95]
    required_n = {}
    for power in power_levels:
      n_required = analysis.solve_power(
        effect_size=abs(d), alpha=alpha, power=power, alternative="larger"
      )
      # Adjust for Wilcoxon efficiency
      required_n[power] = ceil(n_required / 0.95)

  if stats_file:
    stats_file.write(f"{test_name}:\n")
    stats_file.write(f"statistic = {test_stat:.3f}\n")
    stats_file.write(f"p = {p_value:.3f}\n\n")
    stats_file.write(f"Effect Size ({effect_size['name']}):\n")
    stats_file.write(f"{effect_size['value']:.3f}\n\n")

    # Add power analysis results
    stats_file.write("Power Analysis:\n")
    stats_file.write(f"Achieved power with current N={n}: {actual_power:.3f}\n")
    stats_file.write("Required sample sizes:\n")
    for power, n_req in required_n.items():
      stats_file.write(f"  {power * 100:g}% power: N ≥ {n_req}\n")

  return {
    "n": n,
    "mean": mean,
    "se": se,
    "normality": {"is_normal": is_normal, "p_value": normality_p},
    "test": {"name": test_name, "statistic": test_stat, "p_value": p_value},
    "effect_size": effect_size,
    "power_analysis": {"actual_power": actual_power, "required_n": required_n},
  }


######################################
# Model Analysis
######################################


def episode_sf_value(e, idx=None):
  actions = e.actions
  preds = e.transitions.extras["preds"]
  sf_values = preds.sf  # [T, N, A, W]
  actions = e.actions  # [T]

  sf_values = jnp.take_along_axis(sf_values, actions[:, None, None, None], axis=-2)

  sf_values = jnp.squeeze(sf_values, axis=-2)  # [T, N, W]

  in_episode = get_in_episode(e.timesteps)
  sf_values = sf_values[in_episode]
  # [T', ... ]
  if idx is not None:
    sf_values = sf_values[:, idx]
  return sf_values


def plot_sf_values(
  e,
  idxs=None,
  line_mask=None,
  line_names=None,
  figsize=None,
  colors=None,
  styles=None,
  task_w=None,
  plot_q_values=True,
):
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

  line_mask = line_mask or [True, True, False, False, True, True, False, False]

  line_names = line_names or [
    "main",
    "off-task",
    "main2",
    "off-task2",
    "near main",
    "near off-task",
    "near-main2",
    "near-off-task2",
  ]
  # Get first half of line names and take every even index (0, 2)
  first_half = line_names[
    : len(line_names) // 2
  ]  # ['main', 'off-task', 'main2', 'off-task2']
  policy_names = first_half[::2]  # ['main', 'main2']

  colors = colors or ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
  styles = styles or ["-", "--"]

  in_episode = get_in_episode(e.timesteps)
  if task_w is None:
    task_w = e.timesteps.observation.task_w
    task_w = task_w[in_episode]
  max_value = -1000
  for panel_idx, idx in enumerate(idxs):
    sf_values = all_sf_values[:, idx]
    q_value = (sf_values * task_w).sum(-1)
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
      label = (
        line_names[i] if line_names and i < len(line_names) and panel_idx == 0 else None
      )
      ax.plot(
        time_steps,
        sf_values[:, i],
        label=label,
        color=colors[color_idx],
        linestyle=styles[style_idx],
      )

    # Add Q-value plot if plot_q_values is True
    if plot_q_values:
      ax.plot(
        time_steps,
        q_value,
        label="Q-value" if panel_idx == 0 else None,
        color="k",
        linestyle="-",
      )

    if len(idxs) > 1:
      ax.set_title(
        f"Successor Feature Predictions (task={policy_names[idx]})",
        fontsize=DEFAULT_TITLE_SIZE,
      )
    else:
      ax.set_title(f"Successor Feature Predictions", fontsize=DEFAULT_TITLE_SIZE)
    ax.set_xlabel("Time Step", fontsize=DEFAULT_LABEL_SIZE)
    ax.set_ylabel("Value", fontsize=DEFAULT_LABEL_SIZE)
    ax.set_xlim(0, sf_values.shape[0] - 1)
    ax.set_ylim(0, 1.1 * max_value)

  # Only show legend in first panel
  if line_names is not None:
    axs[0].legend()

  # Adjust spacing between subplots
  plt.tight_layout()

  return fig, axs


############################################################################
# Experiment results
############################################################################
def experiment_1_results(
  user_df: DataFrame,
  model_df: DataFrame,
  save_dir: str,
  filter_columns: List[str] = [],
  tell_reuse: int = 1,
  display_figs: bool = False,
  save_figs: bool = True,
  verbosity: int = 0,
  n_simulations: int = 500,
):
  """_summary_

  1. Filter out users with less than 16 successes during training

    Args:
      user_df (DataFrame): _description_
      model_df (DataFrame): _description_
  """
  save_dir = os.path.join(save_dir, f"exp1_tell_reuse={tell_reuse}")
  os.makedirs(save_dir, exist_ok=True)

  # Open stats file
  stats_file = open(os.path.join(save_dir, "stats.txt"), "w")
  stats_file.write("Experiment 1 Statistical Analysis\n\n")

  ##################
  # Get relevant simulations
  ##################
  mdf = model_df.filter(maze="big_m3_maze1", eval=True)

  ##################
  # get all episodes for users who achieved at least 16 successes during training
  ##################
  exp1_eval_df = user_df.filter_by_group(
    input_episode_filter=filter_train_by_min_success,
    input_settings=dict(eval=False),
    output_settings=dict(manipulation=3, tell_reuse=tell_reuse),
    group_key="user_id",
  ).filter(eval=True)

  # Convert reuse column from string to boolean
  if exp1_eval_df.schema["reuse"] == pl.String:
    exp1_eval_df = exp1_eval_df.with_columns(pl.col("reuse") == "true")
  elif exp1_eval_df.schema["reuse"] == pl.Boolean:
    pass
  else:
    raise ValueError("Reuse column is type: ", exp1_eval_df.schema["reuse"])

  ##################
  # filter outliers based on episode path length and max reaction time
  ##################
  # Check if reuse column is string type
  if filter_columns:
    exp1_eval_df = filter_outliers(
      exp1_eval_df,
      filter_columns=filter_columns,
    )

  ##################
  # Create success rate and path reuse plots
  ##################
  # compute the mean by user
  # Get mean success rate per user
  fig, ax = plt.subplots(figsize=(6, 3))
  plot_success_rate_comparison(
    df=exp1_eval_df, model_df=mdf, ax=ax, title="Exp 1 Generalization Success Rate"
  )

  if save_figs:
    fig.savefig(os.path.join(save_dir, "exp1_2_success_rate.pdf"), bbox_inches="tight")

  if display_figs:
    plt.show()

  ##################
  # Create path re-use analysis plots
  ##################
  stats_file.write("\nPath Reuse Analysis\n")
  stats_file.write("======================================\n")

  fig, ax = plt.subplots(figsize=(6, 3))
  plot_path_reuse_comparison(
    stats_file=stats_file,
    df=exp1_eval_df,
    model_df=mdf,
    ax=ax,
    title="Exp 1 Path Reuse",
  )

  if save_figs:
    fig.savefig(os.path.join(save_dir, "exp1_3_path_reuse.pdf"), bbox_inches="tight")

  if display_figs:
    plt.show()

  ######################
  # Plot reaction times when using new path vs. partial reuse
  ######################
  stats_file.write("\nReaction Time Analysis\n")
  stats_file.write("======================================\n")

  for idx, measure in enumerate(["log_first_rt"]):
    stats_file.write(f"\n{idx}. {measure}\n")
    stats_file.write("--------------------\n")

    fig, ax = plt.subplots(figsize=(4, 4))
    plot_bar_rt_comparison(
      exp1_eval_df,
      measure,
      title=dict(
        log_first_rt="First Reaction Time",
        log_avg_rt="Average Reaction Time",
        log_max_rt="Max Reaction Time",
      )[measure],
      ylabel="log seconds",
      xlabels=["New Path", "Partial Reuse"],
      colors=[default_colors["nice purple"], default_colors["bluish green"]],
      stats_file=stats_file,
      n_simulations=n_simulations,
      ax=ax,
    )
    if save_figs:
      fig.savefig(
        os.path.join(save_dir, f"exp1_3_bar_{measure}.pdf"), bbox_inches="tight"
      )
    if display_figs:
      plt.show()

  # Replace the separate success rate and path reuse plots with:
  fig, ax1, ax2 = plot_success_rate_path_reuse_metrics(
    df=exp1_eval_df,
    model_df=mdf,
    title="Exp 1 Success Rate and Path Reuse",
    figsize=(6, 4),
    include_raw_data=True,
  )

  if save_figs:
    fig.savefig(
      os.path.join(save_dir, "exp1_combined_metrics.pdf"), bbox_inches="tight"
    )

  if display_figs:
    plt.show()

  ######################
  # SF Model
  ######################
  sf_episodes = model_df.filter(maze="big_m3_maze1", eval=False, algo="usfa")
  fig, ax = plot_sf_values(
    sf_episodes.episodes[0], plot_q_values=False, figsize=(5, 4), idxs=[0]
  )
  if save_figs:
    fig.savefig(
      os.path.join(save_dir, "exp1_6_sf_predictions.pdf"), bbox_inches="tight"
    )
  if display_figs:
    plt.show()

  # Close stats file at the end
  stats_file.close()
  if verbosity > 0:
    with open(os.path.join(save_dir, "stats.txt"), "r") as f:
      print(f.read())


def experiment_2_results(
  user_df: DataFrame,
  model_df: DataFrame,
  save_dir: str,
  filter_columns: List[str] = None,
  display_figs: bool = False,
  tell_reuse: int = 1,
  save_figs: bool = True,
  verbosity: int = 0,
):
  """_summary_

  1. Filter out users with less than 16 successes during training

    Args:
      user_df (DataFrame): _description_
      model_df (DataFrame): _description_
  """
  save_dir = os.path.join(save_dir, f"exp2_tell_reuse={tell_reuse}")
  os.makedirs(save_dir, exist_ok=True)

  # Open stats file
  stats_file = open(os.path.join(save_dir, "stats.txt"), "w")
  stats_file.write("Experiment 2 Statistical Analysis\n")

  ##################
  # Get relevant simulations
  ##################
  mdf = model_df.filter(maze="big_m1_maze3_shortcut", eval=True)

  ##################
  # get all episodes for users who achieved at least 16 successes during training
  ##################
  exp2_eval_df = user_df.filter_by_group(
    input_episode_filter=filter_train_by_min_success,
    input_settings=dict(eval=False),
    output_settings=dict(manipulation=1, tell_reuse=tell_reuse),
    group_key="user_id",
  ).filter(eval=True)

  # Convert reuse column from string to boolean
  if exp2_eval_df.schema["reuse"] == pl.String:
    exp2_eval_df = exp2_eval_df.with_columns(pl.col("reuse") == "true")
  elif exp2_eval_df.schema["reuse"] == pl.Boolean:
    pass
  else:
    raise ValueError("Reuse column is type: ", exp2_eval_df.schema["reuse"])

  ##################
  # filter outliers based on episode path length and max reaction time
  ##################
  # TODO. QUESTION: should I separately filter out participants that reused the training path vs. though that took a new path?
  # Check if reuse column is string type
  if filter_columns:
    exp2_eval_df = filter_outliers(
      exp2_eval_df,
      filter_columns=filter_columns,
    )

  ##################
  # Create success rate and path reuse plots
  ##################

  fig, ax = plt.subplots(figsize=(6, 3))
  plot_success_rate_comparison(
    df=exp2_eval_df, model_df=mdf, ax=ax, title="Exp 3 Generalization Success Rate"
  )

  if save_figs:
    fig.savefig(os.path.join(save_dir, "exp2_2_success_rate.pdf"), bbox_inches="tight")

  if display_figs:
    plt.show()

  ##################
  # Create path re-use analysis plots
  ##################
  stats_file.write("\nPath Reuse Analysis\n")
  stats_file.write("======================================\n")

  fig, ax = plt.subplots(figsize=(6, 3))
  plot_path_reuse_comparison(
    df=exp2_eval_df,
    model_df=mdf,
    ax=ax,
    stats_file=stats_file,
    title="Exp 3 Path Reuse",
  )

  if save_figs:
    fig.savefig(os.path.join(save_dir, "exp2_3_path_reuse.pdf"), bbox_inches="tight")

  if display_figs:
    plt.show()

  # Replace the separate success rate and path reuse plots with:
  fig, ax1, ax2 = plot_success_rate_path_reuse_metrics(
    df=exp2_eval_df,
    model_df=mdf,
    title="Exp 3 Success Rate and Path Reuse",
    figsize=(6, 4),
    include_raw_data=True,
  )

  if save_figs:
    fig.savefig(
      os.path.join(save_dir, "exp3_combined_metrics.pdf"), bbox_inches="tight"
    )

  if display_figs:
    plt.show()

  # Close stats file at the end
  stats_file.close()
  if verbosity > 0:
    with open(os.path.join(save_dir, "stats.txt"), "r") as f:
      print(f.read())


def experiment_3_results(
  user_df: DataFrame,
  model_df: DataFrame,
  save_dir: str,
  filter_columns: List[str] = None,
  display_figs: bool = False,
  tell_reuse: int = 1,
  save_figs: bool = True,
  verbosity: int = 0,
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
  save_dir = os.path.join(save_dir, f"exp3_tell_reuse={tell_reuse}")
  os.makedirs(save_dir, exist_ok=True)
  # Default to ['avg_rt'] if no filter columns specified

  stats_file = open(os.path.join(save_dir, "stats.txt"), "w")
  stats_file.write("Experiment 3 Statistical Analysis\n")
  stats_file.write("===============================\n\n")

  ##################
  # get all episodes for users who achieved at least 16 successes during training
  ##################
  exp3_eval_df = user_df.filter_by_group(
    input_episode_filter=filter_train_by_min_success,
    input_settings=dict(eval=False),
    output_settings=dict(manipulation=2),
    group_key="user_id",
  )
  ##################
  # Create reaction time difference plot
  ##################
  # Create filter string for filename
  filter_columns = filter_columns or []
  filter_str = ",".join(filter_columns)
  difference_df = compute_condition_difference_df(
    exp3_eval_df._df.filter(tell_reuse=tell_reuse),
    measures=["log_first_rt", "log_max_rt", "log_avg_rt"],
  )
  xlabels = [
    "First",
    # "Max",
    "Average",
  ]
  measures = [
    "log_first_rt",
    # "log_max_rt",
    "log_avg_rt",
  ]
  colors = [
    default_colors["google blue"],
    # default_colors["sky blue"],
    default_colors["google orange"],
  ]
  fig, ax = plot_rt_differences(
    difference_df,
    measures=measures,
    title=f"Exp 4 RT Diff",
    colors=colors,
    ylabel="log seconds",
    xlabels=xlabels,
    stats_file=stats_file,
  )

  if save_figs:
    fig.savefig(
      os.path.join(save_dir, f"exp3_2_rt_diff_filter_{filter_str}.pdf"),
      bbox_inches="tight",
    )
  if display_figs:
    plt.show()
  stats_file.close()
  if verbosity > 0:
    with open(os.path.join(save_dir, "stats.txt"), "r") as f:
      print(f.read())


def experiment_4_results(
  user_df: DataFrame,
  # model_df: DataFrame,
  save_dir: str,
  filter_columns: List[str] = None,
  display_figs: bool = False,
  save_figs: bool = True,
  verbosity: int = 0,
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

  save_dir = os.path.join(save_dir, "exp4")
  os.makedirs(save_dir, exist_ok=True)
  # Default to ['avg_rt'] if no filter columns specified
  filter_columns = filter_columns or []

  # Open stats file
  stats_filename = os.path.join(save_dir, "stats.txt")
  stats_file = open(stats_filename, "w")
  stats_file.write("Experiment 4 Statistical Analysis\n\n")

  ##################
  # Add setting column based on maze name
  ##################
  user_df = user_df._df  # fancy merging will use regular df
  user_df = user_df.filter(manipulation=4)

  def get_maze_setting(maze_str: str) -> str:
    if "short" in maze_str.lower():
      return "short"
    elif "long" in maze_str.lower():
      return "long"
    raise ValueError(f"Could not determine setting from maze string: {maze_str}")

  # Add setting column based on maze name
  user_df = user_df.with_columns(
    setting=pl.col("maze").map_elements(get_maze_setting, return_dtype=pl.String)
  )

  ############################################
  # Create combined figure
  ############################################
  fig, axs = plt.subplots(1, 3, figsize=(15, 4))

  idx = 0

  xlabels = [
    "First",
    # "Max",
    "Average",
  ]
  measures = [
    "log_first_rt",
    # "log_max_rt",
    "log_avg_rt",
  ]
  colors = [
    default_colors["google blue"],
    # default_colors["sky blue"],
    default_colors["google orange"],
  ]
  for setting in ["short", "long"]:
    stats_file.write(f"\n\n=================={setting}===================\n")
    for tell_reuse in [1, 0]:
      if idx > 2:
        break
      difference_df = compute_condition_difference_df(
        user_df.filter(setting=setting, tell_reuse=tell_reuse),
        measures=["log_first_rt", "log_max_rt", "log_avg_rt"],
      )
      stats_file.write(f"\n\nRT Analysis for tell_reuse={tell_reuse}\n")
      stats_file.write(f"-----------------------------------------\n")

      label = dict(short="Near", long="Far")[setting]
      v = {0: "Unknown", 1: "Known"}[tell_reuse]

      plot_rt_differences(
        difference_df,
        ax=axs[idx],
        measures=measures,
        title=f"Exp 2 RT Diff ({label} x {v})",
        colors=colors,
        ylabel="log seconds",
        xlabels=xlabels,
        stats_file=stats_file,
      )
      idx += 1

      # Create and save individual figure
      if save_figs:
        ind_fig, ind_ax = plt.subplots(figsize=(6, 4))
        plot_rt_differences(
          difference_df,
          ax=ind_ax,
          measures=measures,
          title=f"Exp 2 RT Diff ({label} x {v})",
          colors=colors,
          ylabel="log seconds",
          xlabels=xlabels,
          stats_file=None,  # Don't write stats again
        )
        filter_str = ",".join(filter_columns)
        # Save individual figure in multiple formats
        base_path = os.path.join(
          save_dir, f"exp4_2_rt_diff_{setting}_{v}_filter_{filter_str}"
        )
        ind_fig.savefig(f"{base_path}.pdf", bbox_inches="tight")
        ind_fig.savefig(f"{base_path}.png", bbox_inches="tight", dpi=300)
        plt.close(ind_fig)  # Close individual figure

  # Adjust layout
  plt.tight_layout()

  # Save combined figure in multiple formats
  if save_figs:
    filter_str = ",".join(filter_columns)
    base_path = os.path.join(save_dir, f"exp4_2_rt_diff_combined_filter_{filter_str}")
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight")
    fig.savefig(f"{base_path}.png", bbox_inches="tight", dpi=300)
  if display_figs:
    plt.show()

  # Close stats file at the end
  stats_file.close()
  if verbosity > 0:
    with open(stats_filename, "r") as f:
      print(f.read())


def plot_success_rate_path_reuse_metrics(
  df: DataFrame,
  model_df: DataFrame,
  ax=None,
  title="Success Rate and Path Reuse",
  figsize=(8, 4),
  include_raw_data: bool = True,
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
  """Plot success rate and path reuse on the same axes with different y-axes.

  Args:
      df (DataFrame): DataFrame containing human data
      model_df (DataFrame): DataFrame containing model data
      ax (plt.Axes, optional): Matplotlib axes to plot on. If None, creates new figure
      title (str, optional): Plot title
      figsize (tuple, optional): Figure size if creating new figure
      include_raw_data (bool, optional): Whether to include individual human data points

  Returns:
      tuple: (fig, ax1, ax2) containing the figure and both axes objects
  """
  # Create figure if needed
  if ax is None:
    fig, ax1 = plt.subplots(figsize=figsize)
  else:
    fig = ax.figure
    ax1 = ax

  # Create second y-axis
  ax2 = ax1.twinx()

  # Calculate human success rate statistics
  human_successes = (
    df.group_by("user_id")
    .agg(pl.col("success").mean())
    .select("success")
    .to_numpy()
    .flatten()
  )
  human_success_mean = np.mean(human_successes)
  human_success_se = np.sqrt(
    (human_success_mean * (1 - human_success_mean)) / len(human_successes)
  )

  # Calculate human reuse statistics
  human_reuse = (
    df.group_by("user_id")
    .agg(pl.col("reuse").mean())
    .select("reuse")
    .to_numpy()
    .flatten()
  )
  human_reuse_mean = np.mean(human_reuse)
  human_reuse_se = np.sqrt(
    (human_reuse_mean * (1 - human_reuse_mean)) / len(human_reuse)
  )

  # Calculate model statistics
  model_stats = model_df.group_by("algo").agg(
    success_mean=pl.col("success").mean() * 100,
    success_se=(
      pl.col("success").mean()
      * (1 - pl.col("success").mean())
      / pl.col("success").count()
    ).sqrt()
    * 100,
    reuse_mean=pl.col("reuse").mean() * 100,
    reuse_se=(
      pl.col("reuse").mean() * (1 - pl.col("reuse").mean()) / pl.col("reuse").count()
    ).sqrt()
    * 100,
  )

  # Prepare data for plotting
  all_data = {
    "human": {"success": 100 * human_success_mean, "reuse": 100 * human_reuse_mean}
  }
  success_yerr = [100 * human_success_se]
  reuse_yerr = [100 * human_reuse_se]

  # Add model data
  algos = model_stats["algo"].unique().to_list()
  for algo in model_order:
    if not algo in algos:
      continue
    row = model_stats.filter(algo=algo)
    all_data[algo] = {
      "success": row["success_mean"].to_numpy()[0],
      "reuse": row["reuse_mean"].to_numpy()[0],
    }
    success_yerr.append(row["success_se"].to_numpy()[0])
    reuse_yerr.append(row["reuse_se"].to_numpy()[0])

  # Plot bars
  x_pos = np.arange(len(all_data))
  ordered_keys = [k for k in model_order if k in all_data]
  bar_width = 0.35

  # Success rate bars (left axis)
  success_bars = ax1.bar(
    x_pos - bar_width / 2,
    [all_data[k]["success"] for k in ordered_keys],
    bar_width,
    yerr=success_yerr,
    capsize=5,
    color=[model_colors.get(k, "#333333") for k in ordered_keys],
    label="Success Rate",
  )

  # Path reuse bars (right axis) - now using same colors without alpha
  reuse_bars = ax2.bar(
    x_pos + bar_width / 2,
    [all_data[k]["reuse"] for k in ordered_keys],
    bar_width,
    yerr=reuse_yerr,
    capsize=5,
    color=[model_colors.get(k, "#333333") for k in ordered_keys],
    hatch="///",
    label="Path Reuse",
  )

  # Add individual dots for human data if requested
  if include_raw_data:
    x_jitter = np.random.normal(0, 0.05, size=len(human_successes))
    # ax1.scatter(
    #    [-bar_width/2 + j for j in x_jitter],
    #    100 * human_successes,
    #    color="black",
    #    alpha=0.5,
    #    zorder=3,
    #    s=20
    # )
    ax2.scatter(
      [bar_width / 2 + j for j in x_jitter],
      100 * human_reuse,
      color="black",
      alpha=0.5,
      zorder=3,
      s=20,
    )

  # Customize axes
  ax1.set_ylabel("Success Rate (%)", fontsize=DEFAULT_LABEL_SIZE)
  ax2.set_ylabel("Path Reuse (%)", fontsize=DEFAULT_LABEL_SIZE)

  # Set x-ticks
  ax1.set_xticks(x_pos)
  ax1.set_xticklabels(["" for k in ordered_keys])

  # Add chance level line for success rate
  ax1.axhline(y=50, color="r", linestyle="--", alpha=0.5, label="Chance level")

  # Set title
  ax1.set_title(title, fontsize=DEFAULT_TITLE_SIZE)

  # Create custom legend
  # First create legend elements for models/human
  model_legend = []
  for i, key in enumerate(ordered_keys):
    # Create a patch with the model's color
    patch = mpatches.Patch(
      color=model_colors.get(key, "#333333"), label=model_names[key]
    )
    model_legend.append(patch)

  ## Create legend elements for metrics
  # chance_line = mlines.Line2D([], [], color='r', linestyle='--', label='Chance')
  # success_patch = mpatches.Patch(color='gray', label='Success Rate')
  # reuse_patch = mpatches.Patch(color='gray', hatch='///', label='Path Reuse')

  # Combine all legend elements and organize in one row below the plot
  ax1.legend(
    handles=model_legend,
    ncol=len(model_legend) // 2,  # Single row with all models
    bbox_to_anchor=(0.5, -0.2),  # Center horizontally, place below plot
    loc="lower center",
    columnspacing=1,
    handletextpad=0.5,
  )

  # Add grids to both axes
  ax1.grid(True, linestyle="--", alpha=0.7, which="major", axis="y")
  # ax2.grid(True, linestyle='--', alpha=0.3, which='major', axis='y')

  ax1.set_ylim(0, 110)  # Set fixed range for percentage
  ax2.set_ylim(0, 110)

  # Ensure grid lines are behind the bars
  # ax1.set_axisbelow(True)
  # ax2.set_axisbelow(True)

  # Adjust layout with more space at bottom for legend
  # plt.subplots_adjust(bottom=0.2)  # Increase bottom margin to accommodate legend

  # Remove the automatic legend from ax2
  ax2.get_legend().remove() if ax2.get_legend() else None

  return fig, ax1, ax2


if __name__ == "__main__":
  data_dir = "/Users/wilka/git/research/results/human_dyna_craftax/"

  USE_MODEL_DATA = False
  #################
  ## Load model data
  #################
  # TODO: implement model and get results
  model_df = None
  if USE_MODEL_DATA:
    model_df = get_model_data(
      qlearning_path=f"{data_dir}/model_data/ql/save_data/ql-big-2/tota=40000000,exp=exp2/seed=*",
      sf_path=f"{data_dir}/model_data/usfa/save_data/usfa-big-10-search/sf_h=1024,num_=2,tota=40000000,exp=exp2/seed=*",
      dyna_path=f"{data_dir}/model_data/dynaq_shared/save_data/dynaq-big-4/alg=dynaq_shared,agen=256,tota=100000000,exp=exp2/seed=*",
      search_path=f"{data_dir}/search_algos",
      overwrite_episodes=False,
      overwrite_df=False,
      cache_dir=f"{data_dir}/model_data/cache",
    )
  ################
  # Load user data
  ################
  searches = {
    "Paths": f"{data_dir}/user_data/*exps*/*v1*paths*.json",
    "Start": f"{data_dir}/user_data/*exps*/*v1*juncture*.json",
  }
  files = []
  for v in searches.values():
    files.extend(glob(v))

  #valid_files = get_valid_files(searches, verbose=True, plot=False)
  user_df = get_human_data(
    files, overwrite_episode_data=False, overwrite_episode_info=False
  )
  ################
  # Exp 1 analysis
  ################
  save_dir = f"{data_dir}/analysis_results/"
  os.makedirs(save_dir, exist_ok=True)

  experiment_1_results(user_df, model_df, save_dir=save_dir)

  experiment_2_results(user_df, model_df, save_dir=save_dir)

  experiment_3_results(user_df, model_df, save_dir=save_dir)

  experiment_4_results(user_df, model_df, save_dir=save_dir)
