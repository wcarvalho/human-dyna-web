"""
Functions for
(1) plotting experimental results
(2) doing power analysis
"""

from typing import List, Tuple, NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats
from flax import struct
import seaborn as sns

from math import sqrt, ceil
from statsmodels.stats.power import TTestPower
import pandas as pd
import statsmodels.formula.api as smf
from multiprocessing import Pool

from nicewebrl.dataframe import DataFrame

DEFAULT_TITLE_SIZE = 15
DEFAULT_LABEL_SIZE = 15
DEFAULT_LEGEND_SIZE = 10.5

from tqdm.auto import tqdm

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

default_colors["new_path"] = default_colors["nice purple"]
default_colors["reuse"] = default_colors["bluish green"]

model_colors = {
  #'human_success': '#0072B2',
  "human": default_colors["orange"],
  #'human_terminate': '#D55E00',
  "usfa": default_colors["nice purple"],
  "qlearning": default_colors["purple"],
  "dynaq_shared": default_colors["vermillion"],
  "bfs": default_colors["pretty blue"],
  "dfs": default_colors["sky blue"],
  "new_path": default_colors["new_path"],
  "reuse": default_colors["reuse"],
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

measures = [
  "success",
  "path_length",
  "termination",
  "log_first_rt",
  "log_avg_rt",
  "log_total_rt",
  "log_avg_post_rt",
  "log_max_rt",
  "log_max_post_rt",
  "log_max_init_post_rt",
  "log_max_end_rt",
  "log_max_final_rt",
]
measure_to_title = {
  "success": "Success Rate",
  "path_length": "Path Length",
  "termination": "Task Completion Rate",
  "log_first_rt": "First Action Response Time",
  "log_avg_rt": "Average Response Time",
  "log_total_rt": "Total Response Time",
  "log_avg_post_rt": "Post First Action Average Response Time",
  "log_max_rt": "Maximum Response Time",
  "log_max_post_rt": "Post First Action Maximum Response Time",
  "log_max_init_post_rt": "Initial Post First Action Maximum Response Time",
  "log_max_end_rt": "End-Phase Maximum Response Time",
  "log_max_final_rt": "Final Maximum Response Time",
}

measure_to_ylabel = {
  "success": "Success Rate (%)",
  "path_length": "Number of Steps",
  "termination": "Completion Rate (%)",
  "log_first_rt": "Log Response Time (milliseconds)",
  "log_avg_rt": "Log Response Time (milliseconds)",
  "log_total_rt": "Log Response Time (milliseconds)",
  "log_avg_post_rt": "Log Response Time (milliseconds)",
  "log_max_rt": "Log Response Time (milliseconds)",
  "log_max_post_rt": "Log Response Time (milliseconds)",
  "log_max_init_post_rt": "Log Response Time (milliseconds)",
  "log_max_end_rt": "Log Response Time (milliseconds)",
  "log_max_final_rt": "Log Response Time (milliseconds)",
}


class EpisodeData(NamedTuple):
  actions: jax.Array
  timesteps: struct.PyTreeNode
  positions: jax.Array = None
  reaction_times: jax.Array = None
  transitions: struct.PyTreeNode = None


def fix_reuse_column(df: pl.DataFrame):
  if df.schema["reuse"] == pl.String:
    df = df.with_columns(pl.col("reuse") == "true")
  elif df.schema["reuse"] in [pl.Boolean, pl.Int64]:
    pass
  else:
    raise ValueError("Reuse column is type: ", df.schema["reuse"])
  return df


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
  colors=None,
  reuse_column: str = "reuse",
  stats_file=None,
  percentile_ylim: bool = True,
  n_simulations: int = 1000,
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

  ylabel = ylabel or measure_to_ylabel[rt_column]
  title = title or measure_to_title[rt_column]
  len_before = len(df)
  df = df.filter(pl.col("success").is_not_null() & pl.col(reuse_column).is_not_null())
  len_after = len(df)
  print(f"Filtered {len_before - len_after} rows with null success or reuse")

  power_results = None
  if stats_file is not None:
    import os
    import pickle

    # Create a unique cache key based on the analysis parameters
    cache_key = f"{rt_column}_{reuse_column}_{n_simulations}"
    cache_path = f"{stats_file}.{cache_key}.pkl"

    if os.path.exists(cache_path):
      print(f"Loading cached results from {cache_path}")
      try:
        with open(cache_path, 'rb') as f:
          power_results = pickle.load(f)
      except Exception as e:
        print(f"Error loading cache: {e}")
        power_results = None

  # Run analysis if no cached results
  if power_results is None:
    if "rt" in rt_column:
      power_results = power_analysis_rt_across_groups(
        df,
        measure=rt_column,
        reuse_column=reuse_column,
        stats_file=stats_file,
        n_simulations=n_simulations,
      )
    elif "path_length" in rt_column:
      power_results = power_analysis_path_length_across_groups(
        df,
        measure=rt_column,
        reuse_column=reuse_column,
        stats_file=stats_file,
        n_simulations=n_simulations,
      )
    else:
      raise ValueError(f"Unknown rt_column: {rt_column}")
    
    # Save results to cache if cache_file is provided
    if stats_file is not None:
      import os
      import pickle
      
      # Create directory if it doesn't exist
      os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)

      print(f"Saving results to {cache_path}")
      with open(cache_path, 'wb') as f:
        pickle.dump(power_results, f)

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
  colors = [default_colors["nice purple"], default_colors["bluish green"]]
  ax.bar(x_pos, means, yerr=sems, capsize=5, color=colors[: len(means)])

  # Add individual points with jitter
  all_data = []
  for i, key in enumerate(["no_reuse", "reuse"]):
    data = power_results["raw_means"][key]
    print(f"n {key}: {len(data)}")
    x_jitter = np.random.normal(i, 0.04, size=len(data))
    ax.scatter(x_jitter, data, alpha=0.3, color="black", s=20)
    all_data.append(data)

  # Customize plot
  ax.set_xticks(x_pos)
  xlabels = ["New Path", "Partial Reuse"]
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

    # Get user-wide means instead of all individual data points
    user_means = (
      difference_df.group_by("user_id").agg(pl.col(measure).mean()).select(measure)
    )
    all_diffs.append(user_means.to_numpy().flatten())

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

  # Add individual points with jitter (now showing user means)
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
  model_df: DataFrame = None,
  ax=None,
  title="Success Rate Comparison",
  include_raw_data: bool = False,
  figsize=(6, 3),
) -> Tuple[plt.Figure, plt.Axes]:
  """Plot success rate comparison between human and model data.

  Args:
      df (DataFrame): DataFrame containing human data
      model_df (DataFrame, optional): DataFrame containing model data. If None, only plots human data.
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

  # Calculate model statistics if model_df is provided
  model_stats = None
  if model_df is not None:
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
    model_stats=model_stats,  # Will be None if model_df was None
    ax=ax,
    legend=True,  # Only show legend if we have model data
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


def plot_success_rate_path_reuse_metrics(
  df: DataFrame,
  model_df: DataFrame = None,
  stats_file=None,
  ax=None,
  reuse_column: str = "reuse",
  path_deviance_column: str = "path_length_deviance",
  title="Success Rate and Path Reuse",
  figsize=(8, 8),  # Changed to square figure for better 2D visualization
  include_raw_data: bool = True,
  min_circle_size: int = 10,
  max_circle_size: int = 100,
) -> Tuple[plt.Figure, plt.Axes]:
  """Plot success rate vs path reuse as a 2D scatter plot with error bars.

  Args:
      df (DataFrame): DataFrame containing human data
      model_df (DataFrame, optional): DataFrame containing model data. If None, only plots human data.
      stats_file (file, optional): File to write statistics to
      ax (plt.Axes, optional): Matplotlib axes to plot on. If None, creates new figure
      reuse_column (str, optional): Column name for path reuse metric
      path_deviance_column (str, optional): Column name for path length deviance
      title (str, optional): Plot title
      figsize (tuple, optional): Figure size if creating new figure
      include_raw_data (bool, optional): Whether to include individual human data points
      min_circle_size (int, optional): Minimum size for circles
      max_circle_size (int, optional): Maximum size for circles

  Returns:
      tuple: (fig, ax) containing the figure and axes object
  """

  # Create figure if needed
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)
  else:
    fig = ax.figure

  # Calculate human statistics
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

  results = power_analysis_path_reuse(
    df, measure=reuse_column, mu=0.5, alpha=0.05, plot=False, stats_file=stats_file
  )

  human_reuse = (
    df.group_by("user_id")
    .agg(pl.col(reuse_column).mean())
    .select(reuse_column)
    .to_numpy()
    .flatten()
  )
  human_reuse_mean = results["mean"]
  human_reuse_se = results["se"]

  # Calculate path length deviance per user if the column exists
  if path_deviance_column in df.columns:
    user_deviance = (
      df.group_by("user_id")
      .agg(pl.col(path_deviance_column).mean())
      .select(path_deviance_column)
      .to_numpy()
      .flatten()
    )
    # Scale deviance values to circle sizes
    mean_deviance = np.mean(user_deviance)
    # Map deviance values to circle sizes between min_circle_size and max_circle_size
    if np.max(user_deviance) > np.min(user_deviance):  # Avoid division by zero
      normalized_deviance = (user_deviance - np.min(user_deviance)) / (
        np.max(user_deviance) - np.min(user_deviance)
      )
      user_circle_sizes = min_circle_size + normalized_deviance * (
        max_circle_size - min_circle_size
      )
    else:
      user_circle_sizes = np.ones_like(user_deviance) * (
        (min_circle_size + max_circle_size) / 2
      )
  else:
    # If path_deviance_column doesn't exist, use a default size
    user_circle_sizes = np.ones(len(human_successes)) * 20
    mean_deviance = None

  # Prepare data for plotting
  all_data = {
    "human": {
      "success": 100 * human_success_mean,
      "reuse": 100 * human_reuse_mean,
      "success_se": 100 * human_success_se,
      "reuse_se": 100 * human_reuse_se,
      "deviance": mean_deviance,
    }
  }

  # Add model data if provided
  if model_df is not None:
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

    # Add model data
    algos = model_stats["algo"].unique().to_list()
    for algo in model_order:
      if algo not in algos:
        continue
      row = model_stats.filter(algo=algo)
      all_data[algo] = {
        "success": row["success_mean"].to_numpy()[0],
        "reuse": row["reuse_mean"].to_numpy()[0],
        "success_se": row["success_se"].to_numpy()[0],
        "reuse_se": row["reuse_se"].to_numpy()[0],
      }

  # Plot data points with error bars
  ordered_keys = [k for k in model_order if k in all_data]
  marker_size = 100  # Default size of the main scatter points

  # Add individual human data points if requested
  if include_raw_data:
    # Count occurrences of each unique (reuse, success) combination
    reuse_success_pairs = list(zip(100 * human_reuse, 100 * human_successes))
    unique_pairs, counts = np.unique(reuse_success_pairs, axis=0, return_counts=True)
    
    # Create a dictionary to store counts for each position
    position_counts = {tuple(pair): count for pair, count in zip(unique_pairs, counts)}
    
    # Plot each unique point with size based on count
    for (x, y), count in position_counts.items():
      # Scale point size based on count
      point_size = user_circle_sizes[0] * (1 + 0.5 * np.log(count))
      
      # Plot the point
      ax.scatter(
        x, y,
        color="black",
        alpha=0.2,
        zorder=1,
        s=point_size,
      )
      
      # Add annotation with count if more than 1 participant
      if count > 1:
        ax.annotate(
          f"{count}",
          xy=(x + 2, y - 5),  # Offset slightly from the point
          fontsize=10,
          color="black",
          alpha=0.7,
          zorder=2
        )
    
    # Add a single entry to the legend
    ax.scatter([], [], color="black", alpha=0.2, s=user_circle_sizes[0], label="Individual participants")

  # Plot each model/human data point with error bars
  for key in ordered_keys:
    data = all_data[key]

    ax.errorbar(
      data["reuse"],
      data["success"],
      xerr=data["reuse_se"],
      yerr=data["success_se"],
      fmt="none",
      color=model_colors.get(key, "#333333"),
      capsize=5,
      capthick=2,
      elinewidth=2,
      zorder=2,
    )

    # Set marker size based on deviance if available
    if "deviance" in data and data["deviance"] is not None:
      if mean_deviance is not None:  # Scale relative to human mean
        point_size = marker_size * (data["deviance"] / mean_deviance)
        # Ensure size is reasonable
        # point_size = max(min_circle_size, min(max_circle_size * 2, point_size))

        # Add annotation for the deviance value
        # ax.annotate(
        #    f"Dev: {data['deviance']:.2f}",
        #    xy=(data["reuse"], data["success"] - 5),  # Position slightly below the point
        #    xytext=(0, -15),  # Offset text below the point
        #    textcoords="offset points",
        #    ha='center',
        #    fontsize=12,
        #    color=model_colors.get(key, "#333333"),
        #    alpha=0.8
        # )
      else:
        point_size = marker_size
    else:
      point_size = marker_size

    ax.scatter(
      data["reuse"],
      data["success"],
      color=model_colors.get(key, "#333333"),
      s=point_size,
      label=model_names[key],
      zorder=3,
    )

  # Customize axes
  ax.set_xlabel("Path Reuse (%)", fontsize=DEFAULT_LABEL_SIZE)
  ax.set_ylabel("Success Rate (%)", fontsize=DEFAULT_LABEL_SIZE)
  ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)

  # Add chance level line for success rate
  ax.axhline(y=50, color="r", linestyle="--", alpha=0.5, label="Chance level")
  ax.axvline(x=50, color="r", linestyle="--", alpha=0.5)

  # Set axis limits with some padding
  ax.set_xlim(-5, 105)
  ax.set_ylim(-5, 105)

  # Add grid
  ax.grid(True, linestyle="--", alpha=0.7)

  # Add legend
  if model_df is not None:
    ax.legend(
      #bbox_to_anchor=(0.5, -0.15),  # Place legend below plot
      loc="lower right",
      #ncol=len(ordered_keys) // 2,  # Arrange in two rows
      ncol=2,
      columnspacing=1,
      handletextpad=0.5,
      fontsize=DEFAULT_LEGEND_SIZE,
    )

  return fig, ax


######################################
# Power Analysis function
######################################


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
  df: pl.DataFrame,
  measure: str,
  reuse_column: str = "reuse",
  alpha=0.05,
  stats_file=None,
  n_simulations=500,
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
  data = df._df.select(["user_id", reuse_column, measure]).to_pandas()
  data.columns = ["user_id", "reuse", "RT"]

  # Calculate descriptive statistics by group
  user_stats = df.group_by(["user_id", reuse_column]).agg(
    mean_val=pl.col(measure).mean()
  )

  reuse_means = user_stats.filter(pl.col(reuse_column) == True)["mean_val"].to_numpy()
  no_reuse_means = user_stats.filter(pl.col(reuse_column) == False)[
    "mean_val"
  ].to_numpy()

  n1, n2 = len(no_reuse_means), len(reuse_means)
  mean1, mean2 = np.mean(no_reuse_means), np.mean(reuse_means)
  var1, var2 = np.var(no_reuse_means, ddof=1), np.var(reuse_means, ddof=1)

  if stats_file is None:
    return {
      "descriptive": {
        "means": {"no_reuse": mean1, "reuse": mean2},
        "sds": {"no_reuse": np.sqrt(var1), "reuse": np.sqrt(var2)},
        "ses": {"no_reuse": np.sqrt(var1 / n1), "reuse": np.sqrt(var2 / n2)},
      },
      "raw_means": {"no_reuse": no_reuse_means, "reuse": reuse_means},
    }

  # Fit linear mixed effects model
  model = smf.mixedlm("RT ~ reuse", data, groups=data["user_id"])
  result = model.fit(reml=True)

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


def power_analysis_path_length_across_groups(
  df: pl.DataFrame,
  measure: str,
  alpha: float = 0.05,
  stats_file=None,
  reuse_column: str = "reuse",
  n_simulations: int = 500,
):
  """Perform power analysis for between-groups comparison of path lengths using linear mixed effects model.

  Args:
      df: DataFrame with columns [user_id, reuse, path_length] where:
          - user_id: identifier for each participant
          - reuse: boolean indicating condition
          - path_length: length of path taken
      alpha: Significance level (default: 0.05)
      stats_file: Optional file handle to write stats output
      n_simulations: Number of simulations for power analysis

  Returns:
      dict containing analysis results
  """
  # Convert to pandas for statsmodels compatibility
  data = df._df.select(["user_id", reuse_column, measure]).to_pandas()
  data.columns = ["user_id", "reuse", "path_length"]

  # Calculate descriptive statistics by group
  user_stats = df.group_by(["user_id", "reuse"]).agg(
    mean_val=pl.col(measure).mean(),
    median_val=pl.col(measure).median(),
    std_val=pl.col(measure).std(),
  )

  reuse_stats = user_stats.filter(pl.col("reuse") == True)
  no_reuse_stats = user_stats.filter(pl.col("reuse") == False)

  n1, n2 = len(no_reuse_stats), len(reuse_stats)
  mean1, mean2 = no_reuse_stats["mean_val"].mean(), reuse_stats["mean_val"].mean()
  median1, median2 = (
    no_reuse_stats["median_val"].median(),
    reuse_stats["median_val"].median(),
  )
  var1, var2 = (
    np.var(no_reuse_stats["mean_val"].to_numpy(), ddof=1),
    np.var(reuse_stats["mean_val"].to_numpy(), ddof=1),
  )

  if stats_file is None:
    return {
      "descriptive": {
        "means": {"no_reuse": mean1, "reuse": mean2},
        "medians": {"no_reuse": median1, "reuse": median2},
        "sds": {"no_reuse": np.sqrt(var1), "reuse": np.sqrt(var2)},
        "ses": {"no_reuse": np.sqrt(var1 / n1), "reuse": np.sqrt(var2 / n2)},
      },
      "raw_means": {
        "no_reuse": no_reuse_stats["mean_val"].to_numpy(),
        "reuse": reuse_stats["mean_val"].to_numpy(),
      },
    }

  # Fit linear mixed effects model
  model = smf.mixedlm("path_length ~ reuse", data, groups=data["user_id"])
  result = model.fit(reml=True)

  # Get effect size (standardized coefficient)
  param_name = "reuse[T.True]" if "reuse[T.True]" in result.params else "reuse"
  effect_size = result.params[param_name] / np.std(data["path_length"])

  # Calculate required sample sizes for different power levels
  power_levels = [0.8, 0.9, 0.95]
  n_required = {}

  # Binary search for each power level
  for target_power in tqdm(power_levels, desc="Power levels"):
    left = 10  # minimum sample size
    right = 200  # maximum sample size to try

    while left < right:
      n = (left + right) // 2
      # Use gamma distribution for simulation to better match path length distribution
      power = mixed_effects_compute_power_gamma(
        num_subjects=n,
        trials_per_subject=len(data) // len(data["user_id"].unique()),
        B0=result.params["Intercept"],
        B1=result.params[param_name],
        random_effect_var=result.cov_re.iloc[0, 0],
        shape=np.mean(data["path_length"]) ** 2 / np.var(data["path_length"]),
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
  current_power = mixed_effects_compute_power_gamma(
    num_subjects=len(data["user_id"].unique()),
    trials_per_subject=len(data) // len(data["user_id"].unique()),
    B0=result.params["Intercept"],
    B1=result.params[param_name],
    random_effect_var=result.cov_re.iloc[0, 0],
    shape=np.mean(data["path_length"]) ** 2 / np.var(data["path_length"]),
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
      "medians": {"no_reuse": median1, "reuse": median2},
      "sds": {"no_reuse": np.sqrt(var1), "reuse": np.sqrt(var2)},
      "ses": {"no_reuse": np.sqrt(var1 / n1), "reuse": np.sqrt(var2 / n2)},
    },
    "raw_means": {
      "no_reuse": no_reuse_stats["mean_val"].to_numpy(),
      "reuse": reuse_stats["mean_val"].to_numpy(),
    },
  }

  if stats_file:
    stats_file.write("\nLinear Mixed Effects Model Results (Path Length):\n")
    stats_file.write("=========================================\n")
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

    stats_file.write("Medians:\n")
    stats_file.write(f"\tNo Reuse: {median1:.3f}\n")
    stats_file.write(f"\tReuse: {median2:.3f}\n")
    stats_file.write(f"\tDifference: {median2 - median1:.3f}\n\n")

    stats_file.write(f"Effect size: {effect_size:.3f}\n\n")

    stats_file.write("Power Analysis:\n")
    stats_file.write(f"Current power: {current_power:.3f}\n")
    for power, n in n_required.items():
      stats_file.write(f"Required sample size for {power * 100}% power: {n}\n")

  return results


def mixed_effects_compute_power_gamma(
  num_subjects: int,
  trials_per_subject: int,
  B0: float,
  B1: float,
  random_effect_var: float,
  shape: float,
  n_simulations: int = 500,
  alpha: float = 0.05,
):
  """Simulate a mixed effects trial with gamma-distributed path lengths and test for significance.

  Args:
      num_subjects: Number of subjects in simulation
      trials_per_subject: Number of trials per subject
      B0: Intercept coefficient
      B1: Slope coefficient
      random_effect_var: Variance of random effects
      shape: Shape parameter for gamma distribution
      n_simulations: Number of simulations to run
      alpha: Significance level for test

  Returns:
      float: Computed power (proportion of significant results)
  """
  significant_results = 0

  for _ in range(n_simulations):
    # Generate data
    user_ids = np.repeat(range(num_subjects), trials_per_subject)
    reuse = np.random.choice([0, 1], num_subjects * trials_per_subject)
    random_intercepts = np.random.normal(0, np.sqrt(random_effect_var), num_subjects)
    random_intercepts = np.repeat(random_intercepts, trials_per_subject)

    # Generate path lengths using gamma distribution
    mean = np.exp(B0 + B1 * reuse + random_intercepts)
    scale = mean / shape  # scale parameter for gamma distribution
    path_length = np.random.gamma(shape, scale)

    # Create and test model
    data = pd.DataFrame(
      {"path_length": path_length, "reuse": reuse, "user_id": user_ids}
    )
    model = smf.mixedlm("path_length ~ reuse", data, groups=data["user_id"])
    try:
      result = model.fit(reml=True)
      param_name = "reuse[T.True]" if "reuse[T.True]" in result.params else "reuse"
      if result.pvalues[param_name] < alpha:
        significant_results += 1
    except:
      continue

  return significant_results / n_simulations


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


def plot_success_rate_efficient_reuse_metrics(
  df: DataFrame,
  model_df: DataFrame = None,
  stats_file=None,
  ax=None,
  reuse_columns: List[str] = [
    "efficient_reuse_1.25",
    "efficient_reuse_1.5",
    "efficient_reuse_1.75",
    "efficient_reuse_2",
  ],
  title="Success Rate and Efficient Path Reuse",
  figsize=(8, 8),
  color=default_colors["orange"],
) -> Tuple[plt.Figure, plt.Axes]:
  """Plot success rate vs different efficient path reuse metrics as a 2D scatter plot with error bars.

  Args:
      df (DataFrame): DataFrame containing human data
      model_df (DataFrame, optional): DataFrame containing model data. If None, only plots human data.
      stats_file (file, optional): File to write statistics to
      ax (plt.Axes, optional): Matplotlib axes to plot on. If None, creates new figure
      reuse_columns (List[str], optional): List of efficient reuse column names to plot
      title (str, optional): Plot title
      figsize (tuple, optional): Figure size if creating new figure
      color (str, optional): Color to use for all data points

  Returns:
      tuple: (fig, ax) containing the figure and axes object
  """

  # Create figure if needed
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)
  else:
    fig = ax.figure

  # Calculate human success statistics (shared across all reuse metrics)
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

  # Prepare data for plotting
  reuse_data = []

  # Calculate statistics for each reuse metric
  for reuse_column in reuse_columns:
    results = power_analysis_path_reuse(
      df, measure=reuse_column, mu=0.5, alpha=0.05, plot=False, stats_file=stats_file
    )

    # Extract threshold value from column name (e.g., "efficient_reuse_1.25" -> "1.25")
    threshold = reuse_column.split("_")[-1]

    reuse_data.append(
      {"threshold": threshold, "reuse_mean": results["mean"], "reuse_se": results["se"]}
    )

  # Plot each reuse metric data point with error bars
  marker_size = 100  # Size of the scatter points

  for data in reuse_data:
    ax.errorbar(
      100 * data["reuse_mean"],
      100 * human_success_mean,
      xerr=100 * data["reuse_se"],
      yerr=100 * human_success_se,
      fmt="none",
      color=color,
      capsize=5,
      capthick=2,
      elinewidth=2,
      zorder=2,
    )
    ax.scatter(
      100 * data["reuse_mean"],
      100 * human_success_mean,
      color=color,
      s=marker_size,
      zorder=3,
    )
    # Add threshold label next to each point
    ax.annotate(
      data["threshold"],
      xy=(100 * data["reuse_mean"] + 2, 100 * human_success_mean + 2),
      fontsize=10,
      zorder=4,
    )

  # Customize axes
  ax.set_xlabel("Efficient Path Reuse (%)", fontsize=DEFAULT_LABEL_SIZE)
  ax.set_ylabel("Success Rate (%)", fontsize=DEFAULT_LABEL_SIZE)
  ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)

  # Add chance level lines
  ax.axhline(y=50, color="r", linestyle="--", alpha=0.5, label="Chance level")
  ax.axvline(x=50, color="r", linestyle="--", alpha=0.5)

  # Set axis limits with some padding
  ax.set_xlim(-5, 105)
  ax.set_ylim(-5, 105)

  # Add grid
  ax.grid(True, linestyle="--", alpha=0.7)

  return fig, ax
