from typing import List, Tuple
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import os.path


# from analysis.housemaze_analysis_garbarge import plot_rt_condition_differences
from housemaze.human_dyna import utils


from analysis.housemaze_model_data import get_model_data
from analysis.housemaze_user_data import get_human_data
from analysis import experiment_analysis
from nicewebrl.dataframe import DataFrame
import matplotlib.patches as mpatches
from tqdm.auto import tqdm

DEFAULT_TITLE_SIZE = 15
DEFAULT_LABEL_SIZE = 15
DEFAULT_LEGEND_SIZE = 10.5

image_dict = utils.load_image_dict()


# for tqdm both in notebook and terminal
#try:
#  from IPython import get_ipython

#  if "IPKernelApp" in get_ipython().config:
#    from tqdm.notebook import tqdm

#    try:
#      import ipywidgets
#    except:
#      pass
#  else:
#    from tqdm import tqdm
#except (ImportError, AttributeError):
#  from tqdm import tqdm

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

  in_episode = experiment_analysis.get_in_episode(e.timesteps)
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

  in_episode = experiment_analysis.get_in_episode(e.timesteps)
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
      ax.set_title("Successor Feature Predictions", fontsize=DEFAULT_TITLE_SIZE)
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
def path_reuse_results(
  user_df: DataFrame,
  model_df: DataFrame,
  save_dir: str,
  filter_columns: List[str] = [],
  tell_reuse: int = 1,
  display_figs: bool = False,
  save_figs: bool = True,
  verbosity: int = 0,
  n_simulations: int = 1000,
  rereun_analysis: bool = False,
  rt_ylim: Tuple[float, float] = None,
):
  """_summary_

  1. Filter out users with less than 16 successes during training

    Args:
      user_df (DataFrame): _description_
      model_df (DataFrame): _description_
  """
  save_dir = os.path.join(save_dir, f"ll_reuse_plots={tell_reuse}")
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
  sub_df = user_df.filter_by_group(
    input_episode_filter=experiment_analysis.filter_train_by_min_success,
    input_settings=dict(eval=False),
    output_settings=dict(manipulation=3, tell_reuse=tell_reuse),
    group_key="user_id",
  ).filter(eval=True)

  ##################
  # Create success rate and path reuse plot
  ##################
  fig, ax = experiment_analysis.plot_success_rate_path_reuse_metrics(
    df=sub_df,
    model_df=mdf,
    stats_file=stats_file,
    title= "Path Reuse & Generalization Success",
    figsize=(6, 4),
    include_raw_data=False,
    legend_ncol=1,
  )

  if save_figs:
    fig.savefig(os.path.join(save_dir, "path_reuse_plots.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(save_dir, "path_reuse_plots.png"), bbox_inches="tight", dpi=300)

  if display_figs:
    from IPython.display import display
    display(fig)

  ######################
  # Plot reaction times when using new path vs. partial reuse
  ######################
  stats_file.write("\nReaction Time Analysis\n")
  stats_file.write("======================================\n")

  for idx, measure in enumerate(['log_max_rt', 'log_first_rt', 'first_rt', 'max_rt', 'total_rt']):
    stats_file.write(f"\n{measure}\n")
    stats_file.write("--------------------\n")

    fig, ax = plt.subplots(figsize=(4, 4))

    #if rt_ylim:
    #  if isinstance(rt_ylim[0], list):
    #    rt_ylim_ = rt_ylim[idx]
    #  else:
    #    rt_ylim_ = rt_ylim

    experiment_analysis.plot_bar_rt_comparison(
      sub_df.filter(success=1),
      measure,
      n_simulations=n_simulations,
      stats_file=stats_file,
      ax=ax,
      rereun_analysis=rereun_analysis,
      #ylim=rt_ylim_,
    )
    plt.show()

    if save_figs:
      fig.savefig(
        os.path.join(save_dir, f"path_reuse_bar_plots_{measure}.pdf"), bbox_inches="tight"
      )
      fig.savefig(
        os.path.join(save_dir, f"path_reuse_bar_plots_{measure}.png"), bbox_inches="tight", dpi=300
      )
    if display_figs:
      plt.show()

  # Close stats file at the end
  stats_file.close()
  if verbosity > 0:
    with open(os.path.join(save_dir, "stats.txt"), "r") as f:
      print(f.read())

def sf_analysis_results(
  model_df: DataFrame,
  save_dir: str,
  display_figs: bool = False,
  save_figs: bool = True,
):
  sf_episodes = model_df.filter(maze="big_m3_maze1", eval=False, algo="usfa")
  fig, ax = plot_sf_values(
    sf_episodes.episodes[0], plot_q_values=False, figsize=(5, 4), idxs=[0]
  )
  if save_figs:
    fig.savefig(
      os.path.join(save_dir, "sf_predictions_plots.pdf"), bbox_inches="tight"
    )
  if display_figs:
    from IPython.display import display
    display(fig)


def juncture_results(
  user_df: DataFrame,
  # model_df: DataFrame,
  save_dir: str,
  filter_columns: List[str] = None,
  display_figs: bool = False,
  save_figs: bool = True,
  verbosity: int = 0,
  tell_reuse_options=[1, 0],
  figsize=(5.5, 4),
  include_raw_data: bool = False,
  show_legend: bool = True,
  options: List[Tuple[str, int]] = None,
  measure = "log_first_rt",
  ylim=None,
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
  # Create combined figure with all conditions on one plot
  ############################################
  fig, ax = plt.subplots(figsize=figsize)

  # We'll focus only on first RT
  
  
  # Define colors and labels for each condition
  condition_colors = {
    ('short', 1): experiment_analysis.default_colors["sky blue"],     # Near x Known
    ('long', 1): experiment_analysis.default_colors["vermillion"],    # Far x Known
    ('short', 0): experiment_analysis.default_colors["bluish green"], # Near x Unknown
  }
  
  condition_labels = {
    ('short', 1): "Near, Known Test goal",
    ('long', 1): "Far, Known Test goal",
    ('short', 0): "Near, Unknown Test goal",
  }
  
  # Store all data for combined plot
  all_diffs = []
  all_means = []
  all_sems = []
  all_labels = []
  all_colors = []
  
  options = options or [
    ('short', 1),
    ('short', 0),
    ('long', 1),
  ]

  # Collect data for each condition
  for setting, tell_reuse in options:
    stats_file.write(f"\n\n=================={setting}===================\n")
    difference_df = experiment_analysis.compute_condition_difference_df(
      user_df.filter(setting=setting, tell_reuse=tell_reuse),
      measures=[measure],
    )
    stats_file.write(f"\n\nRT Analysis for tell_reuse={tell_reuse}\n")
    stats_file.write("-----------------------------------------\n")

    # Get statistics for this condition
    results = experiment_analysis.power_analysis_rt_differences(
      difference_df, measure, stats_file=stats_file
    )

    # Store data for plotting
    all_diffs.append(difference_df[measure].to_numpy())
    all_means.append(results["mean"])
    all_sems.append(results["se"])
    all_labels.append(condition_labels[(setting, tell_reuse)])
    all_colors.append(condition_colors[(setting, tell_reuse)])

  # Create bar plot with all conditions
  x_pos = np.arange(len(all_means))
  ax.bar(
    x_pos,
    all_means,
    yerr=all_sems,
    capsize=5,
    color=all_colors,
    #error_kw=dict(ecolor=experiment_analysis.default_colors["vermillion"]),
    
  )

  # Add individual points with jitter
  if include_raw_data:
    for i, diffs in enumerate(all_diffs):
      x_jitter = np.random.normal(i, 0.125, size=len(diffs))
      ax.scatter(x_jitter, diffs, alpha=0.3, color="black", s=20)

  # Add zero line
  ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)

  # Customize plot
  ax.set_xticks(x_pos)
  ax.set_xticklabels([])
  ax.set_ylabel(experiment_analysis.measure_to_ylabel[measure], fontsize=DEFAULT_LABEL_SIZE)
  ax.set_title("Juncture Manipulation\nReaction Time Difference", fontsize=DEFAULT_TITLE_SIZE)
  ax.tick_params(axis="both", which="major", labelsize=DEFAULT_LABEL_SIZE)
  ax.grid(True, linestyle="--", alpha=0.7)

  # Create legend with colored patches
  legend_elements = [
    mpatches.Patch(color=all_colors[i], label=all_labels[i])
    for i in range(len(all_labels))
  ]
  if show_legend:
    ax.legend(handles=legend_elements, loc="lower right", fontsize=DEFAULT_LEGEND_SIZE)

  # Set y-axis limits based on all data points
  if ylim is None:
    all_data = np.concatenate(all_diffs)
    y_min, y_max = np.percentile(all_data, [1, 99])
  else:
    y_min, y_max = ylim
  y_range = y_max - y_min
  ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

  # Adjust layout
  plt.tight_layout()

  # Save combined figure in multiple formats
  if save_figs:
    base_path = os.path.join(save_dir, "exp4_2_rt_diff_combined")
    fig.savefig(f"{base_path}_{measure}.pdf", bbox_inches="tight")
    fig.savefig(f"{base_path}_{measure}.png", bbox_inches="tight", dpi=300)
  if display_figs:
    from IPython.display import display
    display(fig)

  # Close stats file at the end
  stats_file.close()
  if verbosity > 0:
    with open(stats_filename, "r") as f:
      print(f.read())


def shortcut_results(
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
  sub_df = user_df.filter_by_group(
    input_episode_filter=experiment_analysis.filter_train_by_min_success,
    input_settings=dict(eval=False),
    output_settings=dict(manipulation=1, tell_reuse=tell_reuse),
    group_key="user_id",
  ).filter(eval=True)

  ## Convert reuse column from string to boolean
  #if sub_df.schema["reuse"] == pl.String:
  #  sub_df = sub_df.with_columns(pl.col("reuse") == "true")
  #elif sub_df.schema["reuse"] == pl.Boolean:
  #  pass
  #else:
  #  raise ValueError("Reuse column is type: ", sub_df.schema["reuse"])

  ###################
  ## filter outliers based on episode path length and max reaction time
  ###################
  ## TODO. QUESTION: should I separately filter out participants that reused the training path vs. though that took a new path?
  ## Check if reuse column is string type
  #if filter_columns:
  #  sub_df = filter_outliers(
  #    sub_df,
  #    filter_columns=filter_columns,
  #  )

  ##################
  # Create success rate and path reuse plots
  ##################
  fig, ax = experiment_analysis.plot_success_rate_path_reuse_metrics(
    df=sub_df,
    model_df=mdf,
    stats_file=stats_file,
    title= "Shortcut Path Reuse & Generalization Success",
    figsize=(6, 4),
    include_raw_data=False,
  )
  #fig, ax = plt.subplots(figsize=(6, 3))
  #experiment_analysis.plot_success_rate_comparison(
  #  df=sub_df, model_df=mdf, ax=ax, title="Exp 3 Generalization Success"
  #)

  if save_figs:
    fig.savefig(os.path.join(save_dir, "exp2_2_success_rate.pdf"), bbox_inches="tight")

  if display_figs:
    from IPython.display import display
    display(fig)

  ###################
  ## Create path re-use analysis plots
  ###################
  #stats_file.write("\nPath Reuse Analysis\n")
  #stats_file.write("======================================\n")

  #fig, ax = plt.subplots(figsize=(6, 3))
  #experiment_analysis.plot_path_reuse_comparison(
  #  df=sub_df,
  #  model_df=mdf,
  #  ax=ax,
  #  stats_file=stats_file,
  #  title="Exp 3 Path Reuse",
  #)

  #if save_figs:
  #  fig.savefig(os.path.join(save_dir, "exp2_3_path_reuse.pdf"), bbox_inches="tight")

  #if display_figs:
  #  plt.show()

  ## Replace the separate success rate and path reuse plots with:
  #fig, ax1, ax2 = experiment_analysis.plot_success_rate_path_reuse_metrics(
  #  df=sub_df,
  #  model_df=mdf,
  #  title="Exp 3 Success Rate and Path Reuse",
  #  figsize=(6, 4),
  #  include_raw_data=True,
  #)

  #if save_figs:
  #  fig.savefig(
  #    os.path.join(save_dir, "exp3_combined_metrics.pdf"), bbox_inches="tight"
  #  )

  #if display_figs:
  #  plt.show()

  # Close stats file at the end
  stats_file.close()
  if verbosity > 0:
    with open(os.path.join(save_dir, "stats.txt"), "r") as f:
      print(f.read())


def start_results(
  user_df: DataFrame,
  save_dir: str,
  filter_columns: List[str] = None,
  display_figs: bool = False,
  tell_reuse: int = 1,
  save_figs: bool = True,
  verbosity: int = 0,
  ylim: Tuple[float, float] = None,
):
  """_summary_

  1. Filter out users with less than 16 successes during training

    Args:
      user_df (DataFrame): _description_
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
    input_episode_filter=experiment_analysis.filter_train_by_min_success,
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
  difference_df = experiment_analysis.compute_condition_difference_df(
    exp3_eval_df._df.filter(tell_reuse=tell_reuse),
    measures=["log_first_rt", "log_max_rt", "log_avg_rt"],
  )
  xlabels = [
    "First",
    # "Max",
    #"Average",
  ]
  measures = [
    "log_first_rt",
    # "log_max_rt",
    #"log_avg_rt",
  ]
  colors = [
    experiment_analysis.default_colors["google blue"],
    # experiment_analysis.default_colors["sky blue"],
    #default_colors["google orange"],
  ]
  fig, ax = plt.subplots(figsize=(3, 4))
  fig, ax = experiment_analysis.plot_rt_differences(
    difference_df,
    measures=measures,
    title="Start Manipulation\nReaction Time Difference",
    colors=colors,
    ylabel="log ms",
    xlabels=xlabels,
    stats_file=stats_file,
    ax=ax,
    ylim=ylim,
  )

  if save_figs:
    fig.savefig(
      os.path.join(save_dir, f"exp3_2_rt_diff_filter_{filter_str}.pdf"),
      bbox_inches="tight",
    )
  if display_figs:
    from IPython.display import display
    display(fig)
  stats_file.close()
  if verbosity > 0:
    with open(os.path.join(save_dir, "stats.txt"), "r") as f:
      print(f.read())




if __name__ == "__main__":
  data_dir = "/Users/wilka/git/research/results/human_dyna/"

  ################
  # Load model data
  ################
  # %debug
  model_df = get_model_data(
    qlearning_path=f"{data_dir}/model_data/ql/save_data/ql-big-2/tota=40000000,exp=exp2/seed=*",
    sf_path=f"{data_dir}/model_data/usfa/save_data/usfa-big-10-search/sf_h=1024,num_=2,tota=40000000,exp=exp2/seed=*",
    dyna_path=f"{data_dir}/model_data/dynaq_shared/save_data/dynaq-big-4/alg=dynaq_shared,agen=256,tota=100000000,exp=exp2/seed=*",
    search_path=f"{data_dir}/search_algos",
    overwrite_episodes=False,
    overwrite_df=False,
  )

  ################
  # Load user data
  ################
  from glob import glob

  # ON COMPUTER
  RESULTS_DIR = "/Users/wilka/git/research/"
  USER_RESULTS_DIR = os.path.join(RESULTS_DIR, "results/human_dyna/user_data/exps")

  human_data_pattern = "final*v2*"
  files = f"{USER_RESULTS_DIR}/*{human_data_pattern}*.json"
  # files = 'jaxemaze_data/*.json'
  valid_files = list(set(glob(files)))
  user_df = get_human_data(
    valid_files,
    overwrite_episode_data=False,
    overwrite_episode_info=False,
    require_finished=False,
    load_df_only=True,
  )
  ################
  # Exp 1 analysis
  ################
  save_dir = f"{data_dir}/housemaze_analysis_results_final/"
  os.makedirs(save_dir, exist_ok=True)

  path_reuse_results(user_df, model_df, save_dir=save_dir)

  shortcut_results(user_df, model_df, save_dir=save_dir)

  start_results(user_df, model_df, save_dir=save_dir)

  juncture_results(user_df, model_df, save_dir=save_dir)
