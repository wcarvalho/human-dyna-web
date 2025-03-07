"""
Functions for
(1) getting dataframes related to different experiments
(2) plotting related metrics
"""

from typing import List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import os.path
from analysis import experiment_analysis

from nicewebrl import DataFrame
import craftax_utils
import craftax_experiment_configs
import craftax_experiment_structure as experiment
from IPython.display import HTML, display

OPTIMAL_TEST_PATHS = {}
for config in craftax_experiment_configs.PATHS_CONFIGS:
  # Create cache path
  cache_dir = "craftax_cache/optimal_paths"
  os.makedirs(cache_dir, exist_ok=True)
  cache_file = os.path.join(cache_dir, f"path_{hash(str(config))}.npy")

  # Try to load from cache
  if os.path.exists(cache_file):
    path = np.load(cache_file)
  else:
    # Calculate path and save to cache
    env_params = craftax_experiment_configs.make_block_env_params(
      config, experiment.default_params
    ).replace(
      # goal_locations=config.test_object_location,
      # current_goal=jnp.asarray(config.test_objects[0], dtype=jnp.int32),
      start_positions=experiment.make_start_position(config.start_eval_positions),
    )
    timestep = experiment.jax_web_env.reset(jax.random.PRNGKey(0), env_params)
    goal_position = config.test_object_location
    path, _ = craftax_utils.astar(timestep.state, goal_position)
    path = np.array(path)

    np.save(cache_file, path)
  OPTIMAL_TEST_PATHS[config.world_seed] = path

OPTIMAL_TEST_LENGTHS = {k: len(v) - 1 for k, v in OPTIMAL_TEST_PATHS.items()}


def filter_to_str(filter: dict):
  return "".join([f"{k}={v}" for k, v in filter.items()])


def get_path_reuse_df(
  user_df: DataFrame,
  tell_reuse: int = 1,
  eval_map: bool = False,
):
  sub_df = user_df.filter_by_group(
    input_episode_filter=experiment_analysis.filter_train_by_min_success,
    input_settings=dict(eval=False),
    output_settings=dict(
      manipulation="paths", tell_reuse=tell_reuse, eval_map=int(eval_map)
    ),
    group_key="user_id",
  ).filter(eval=True)
  sub_df = experiment_analysis.fix_reuse_column(sub_df)
  sub_df = sub_df.filter(
    pl.col("success").is_not_null() & pl.col("reuse").is_not_null()
  )
  return sub_df


def get_path_reuse_stats_file(save_dir: str, tell_reuse: int):
  save_dir = os.path.join(save_dir, f"path_reuse_tell_goal={tell_reuse}")
  os.makedirs(save_dir, exist_ok=True)
  stats_file = os.path.join(save_dir, "stats.txt")
  return stats_file


def path_similarity(path1, path2):
  path1 = path1[-10:]
  path2 = path2[-10:]

  # Convert to direction vectors
  def to_directions(path):
    dirs = []
    for i in range(len(path) - 1):
      dx = path[i + 1][0] - path[i][0]
      dy = path[i + 1][1] - path[i][1]
      mag = (dx * dx + dy * dy) ** 0.5
      if mag > 0:
        dirs.append((dx / mag, dy / mag))
    return dirs

  dirs1 = to_directions(path1)
  dirs2 = to_directions(path2)

  # Resample longer to match shorter
  min_len = min(len(dirs1), len(dirs2))
  if len(dirs1) > min_len:
    step = len(dirs1) / min_len
    dirs1 = [dirs1[int(i * step)] for i in range(min_len)]
  elif len(dirs2) > min_len:
    step = len(dirs2) / min_len
    dirs2 = [dirs2[int(i * step)] for i in range(min_len)]

  # Average dot product
  dots = sum(d1[0] * d2[0] + d1[1] * d2[1] for d1, d2 in zip(dirs1, dirs2))
  similarity = (dots / min_len + 1) / 2  # Map from [-1,1] to [0,1]

  return similarity


def visualize_user_path_reuse(df: DataFrame, user_id: int, idx=None, **kwargs):
  user_df = df.filter(user_id=user_id)
  test_mazes = user_df["name"].unique()
  test_mazes = [t for t in test_mazes if "eval" in t]
  test_mazes = sorted(test_mazes)

  for i in range(len(test_mazes)):
    if idx is not None:
      if i != idx:
        continue
    # get random test maze
    i = int(i)
    test_maze = test_mazes[i]
    test_df = user_df.filter(eval=True, name=test_maze, **kwargs)
    if len(test_df.episodes) == 0:
      print(f"No test episodes for maze: {test_maze}")
      continue

    # get corresponding train maze
    train_maze = test_maze.replace("eval1", "training")
    start_pos = test_df["start_pos"].to_list()[0]
    train_df = user_df.filter(
      name=train_maze, room=0, eval=False, success=1, start_pos=start_pos
    )

    if len(train_df.episodes) == 0:
      print(f"No train episodes for maze: {train_maze}")
      continue

    ############################################################
    # plot {train, test} paths for test reactio times
    ############################################################
    width = 5
    fig, axs = plt.subplots(2, 2, figsize=(2 * width, 2 * width))
    axs = axs.flatten()

    def make_image_path_panel(episode, ax):
      first_state = first(episode.timesteps.state)
      image = craftax_utils.render_fn(first_state, show_agent=False)
      path = episode.positions
      actions = craftax_utils.actions_from_path(path)
      craftax_utils.place_arrows_on_image(
        image=image,
        positions=path,
        actions=actions,
        maze_height=first_state.map.shape[1],
        maze_width=first_state.map.shape[2],
        ax=ax,
        display_image=True,
        arrow_color="red",
        show_path_length=False,
        start_color="red",
      )

    # plot {train, test} images
    first = lambda t: jax.tree_map(lambda x: x[0], t)
    train_episode = train_df.episodes[0]
    test_episode = test_df.episodes[0]
    with jax.disable_jit():
      display(HTML(test_df.to_pandas().to_html()))
      make_image_path_panel(train_episode, axs[0])
      title = f"User: {user_id}. {train_maze}"
      path_length = len(train_episode.positions)
      title += f"\nSuccess: {train_df['success'][0]}. Path Length: {path_length}"
      axs[0].set_title(title, fontsize=13)
      if len(train_df.episodes) > 1:
        make_image_path_panel(train_df.episodes[1], axs[1])
      else:
        axs[1].remove()

      make_image_path_panel(test_episode, axs[2])
      train_path = train_episode.positions
      test_path = test_episode.positions
      # similarity = path_similarity(train_path, test_path)
      world_seed = test_df["world_seed"].to_list()[0]
      path_length = len(test_episode.positions) - 1
      title = f"{test_maze}. Path Length: {path_length}"
      title += (
        f"\nOverlap: {test_df['overlap'][0]:.2f}. Reuse: {bool(test_df['reuse'][0])}"
      )
      title += f"\nOptimal Path Length: {OPTIMAL_TEST_LENGTHS[world_seed]}"
      axs[2].set_title(title, fontsize=13)

      # plot reaction times
      reaction_times = test_episode.reaction_times
      axs[3].bar(range(len(reaction_times)), reaction_times, color="lightblue")
      axs[3].set_xlabel("Time")
      axs[3].set_title("Reaction Times")
      axs[3].set_ylim(0, max(reaction_times) * 1.1)

    plt.show()


def path_reuse_manipulation_analysis(
  sub_df: DataFrame,
  save_dir: str,
  filter: dict,
  save_figs: bool = True,
  display_figs: bool = True,
  verbosity: int = 1,
  reuse_column: str = "reuse",
  n_simulations: int = 1000,
):
  ############################################################
  # Create stats file
  ############################################################
  suffix = filter_to_str(filter)
  stats_filename = get_path_reuse_stats_file(save_dir, suffix)
  stats_file = open(stats_filename, "w")
  stats_file.write("Statistical Analysis\n\n")

  ############################################################
  # Plot success rate and path reuse
  ############################################################
  tell_reuse = filter.get("tell_reuse")
  if reuse_column == "reuse":
    title = "Path Reuse & Generalization Success Rate"
  else:
    title = "Efficient Path Reuse & Generalization Success Rate"
  if tell_reuse is not None:
    title += f"\nTell Reuse: {bool(tell_reuse)}"
  fig, ax = experiment_analysis.plot_success_rate_path_reuse_metrics(
    df=sub_df,
    model_df=None,
    stats_file=stats_file,
    title=title,
    figsize=(6, 4),
    include_raw_data=True,
    reuse_column=reuse_column,
  )
  if reuse_column == "efficient_reuse":
    ax.set_xlabel(
      "Efficient Path Reuse (%)", fontsize=experiment_analysis.DEFAULT_LABEL_SIZE
    )

  if save_figs:
    fig.savefig(os.path.join(save_dir, "success_path_reuse.pdf"), bbox_inches="tight")
  if display_figs:
    plt.show()

  #############################################################
  ## Plot path length
  #############################################################
  #fig, ax = plt.subplots(figsize=(4, 4))
  #experiment_analysis.plot_bar_rt_comparison(
  #sub_df.filter(success=1),
  #"path_length",
  ##title="Path Length",
  ##ylabel="Length",
  ##xlabels=["New Path", "Partial Reuse"],
  ##colors=[experiment_analysis.default_colors["nice purple"], experiment_analysis.default_colors["bluish green"]],
  #n_simulations=n_simulations,
  #stats_file=stats_file,
  #ax=ax,
  #)
  #plt.show()

  stats_file.close()
  if verbosity > 0:
    with open(stats_filename, "r") as f:
      print(f.read())
