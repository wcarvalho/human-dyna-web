"""
Used for converting craftax data to dataframes.

Key functions:
  - `get_human_data`: wrapper around `make_all_episode_data` that creates env
  - `make_all_episode_data`: loops (in parallel) across files and parses them
  - `make_episode_data`: creates (1) dataframe and (2) EpisodeData for episode

MOST OF THE LOGIC IS IN `make_episode_data`
"""

from glob import glob
from joblib import Parallel, delayed
from typing import Optional, Tuple
import polars as pl
import os.path
from collections import defaultdict
from typing import NamedTuple, List
from flax import struct
from datetime import datetime
import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from flax import serialization

import jax.tree_util as jtu

import nicewebrl
from nicewebrl import TimeStep
from nicewebrl.dataframe import DataFrame
import craftax_experiment_configs as configs

import asyncio


class EpisodeData(NamedTuple):
  actions: jax.Array
  timesteps: TimeStep
  positions: jax.Array = None
  reaction_times: jax.Array = None
  transitions: struct.PyTreeNode = None


def get_task_object(timesteps: TimeStep):
  goal = int(timesteps.state.current_goal[0])
  return configs.GOAL_TO_BLOCK[goal]


def get_agent_position(timesteps: TimeStep):
  return timesteps.state.player_position


def get_step_number(timesteps: TimeStep):
  return timesteps.state.timestep


def success(e: EpisodeData):
  rewards = e.timesteps.reward
  # return rewards
  assert rewards.ndim == 1, "this is only defined over vector, e.g. 1 episode"
  success = rewards > 0.5
  return success.any().astype(np.float32)


def user_id_from_filename(filename: str):
  return int(filename.split("/")[-1].split(".")[0].split("_")[0].split("=")[1])


def time_diff(t1, t2) -> float:
  # Convert string timestamps to datetime objects
  if t1 is None or t2 is None:
    return np.nan
  t1 = datetime.strptime(t1, "%Y-%m-%dT%H:%M:%S.%fZ")
  t2 = datetime.strptime(t2, "%Y-%m-%dT%H:%M:%S.%fZ")

  # Calculate the time difference
  time_difference = t2 - t1

  # Convert the time difference to milliseconds
  return time_difference.total_seconds()


def compute_reaction_time(datum) -> float:
  # Calculate the time difference
  return time_diff(datum["data"]["image_seen_time"], datum["data"]["action_taken_time"])


def make_row(
  datum: dict,
  timesteps: TimeStep,
  file: str,
  episode_info: Optional[dict],
  user_storage: dict,
):
  """THIS IS WHERE YOU'LL WANT TO INSERT OTHER EPISODE LEVEL INFO TO TRACK IN DATAFRAME!!!

  Args:
      datum (dict): _description_
      timesteps (TimeStep): _description_
      file (str): _description_

  Returns:
      _type_: _description_
  """
  is_eval = datum["metadata"]["eval"]
  task_object = int(get_task_object(timesteps))
  if is_eval:
    room = 0
  else:
    train_objects = datum["metadata"]["block_metadata"]["train_objects"]
    room = train_objects.index(task_object)

  row = dict(
    world_seed=datum["metadata"].get("world_seed"),
    condition=datum["metadata"].get("condition", 0),
    name=datum["name"],
    block=datum["metadata"]["block_metadata"]["desc"],
    manipulation=datum["metadata"]["block_metadata"].get("manipulation", None),
    global_episode_idx=episode_info["user_episode_idx"],
    episode_idx=datum["metadata"]["nepisodes"],
    eval=datum["metadata"]["eval"],
    task=int(get_task_object(timesteps)),
    room=room,
    start_pos=str(timesteps.state.player_position[0]),
  )
  row.update(datum["user_data"])

  ##########
  # get experiment name from file
  ##########
  row.update(
    exp_name=user_storage["env_vars"]["NAME"],
    tell_reuse=user_storage["env_vars"]["SAY_REUSE"],
    eval_map=user_storage["env_vars"]["EVAL_SHOW_MAP"],
    timer=0,
  )
  ####################
  # add version, tell_reuse, timer
  ####################
  name = row.get("exp_name")
  if name is not None:
    # example 'exp4-v1-r1-t0-plan'
    # split on '-' and take the first element
    # if v--> version
    # if there's a word at the end, it's the manipulation
    # create a dictionary according to this legend
    legend = dict(v="version")
    name_info = dict()
    for k, v in legend.items():
      if k in name:
        name_info[v] = name.split(k)[1].split("-")[0]
    row.update(name_info)

  # Convert all numeric strings to integers
  for key, value in row.items():
    if isinstance(value, str) and value.isdigit():
      row[key] = int(value)

  #####################
  ## add optimal path length - with caching
  #####################
  #from craftax_utils import astar
  #import os

  ## Create cache directory if it doesn't exist
  #cache_dir = os.path.join(os.path.dirname(file), "path_length_cache")
  #os.makedirs(cache_dir, exist_ok=True)
  
  ## Create cache key from relevant state information
  #cache_key = f"{row['task']}_{timesteps.state.player_position[0]}"
  #cache_file = os.path.join(cache_dir, f"optimal_length_{cache_key}.npy")

  #if os.path.exists(cache_file):
  #  # Load cached length
  #  row["optimal_length"] = np.load(cache_file)
  #  #print(f"Loaded optimal length from cache for {cache_key}")
  #else:
  #  # Calculate and cache length
  #  path = astar(
  #    state=jax.tree_map(lambda x: x[0], timesteps.state),  # first time-step
  #    goal=row["task"],
  #  )
  #  optimal_length = len(path) - 1  # includes done
  #  np.save(cache_file, optimal_length)
  #  row["optimal_length"] = optimal_length

  #print(f"Optimal length: {row['optimal_length']} for {row['task']}")

  return row


def dict_to_string(data):
  # Convert each key-value pair to "key=value" format
  pairs = [f"{key}={value}" for key, value in data.items()]

  # Join all pairs with ", " separator
  return ", ".join(pairs)


def get_block_stage_description(datum):
  # TODO: make craftax-specific
  ####################
  # block information
  ####################
  block_metadata = datum["metadata"]["block_metadata"]
  # e.g. manipulation = 4
  block_manipulation = block_metadata.get("manipulation", -1)
  # e.g. desc = 'off-task object regular'
  block_desc = block_metadata.get("desc", "unknown")

  ####################
  # stage information
  ####################
  return dict(
    world_seed=datum["metadata"].get("world_seed"),
    condition=datum["metadata"].get("condition", 0),
    name=datum["name"],
    block=block_desc,
    manipulation=block_manipulation,
    episode_idx=datum["metadata"]["nepisodes"],
    eval=datum["metadata"]["eval"],
  )


def separate_data_by_block_stage(data: List[dict]):
  """This function will group episodes by the values from get_block_stage_description

  The input i
  So for example, each episode with {'stage': "'not obvious' shortcut",
   'block': 'shortcut',
   'manipulation': 1,
   'episode_idx': 1,
   'eval': True}
   with go into its own list.
  """
  grouped_data = defaultdict(list)
  episode_idx = -1
  keys = set()
  infos = dict()
  # first group all of the data based on which (stage, block) its in
  for datum in data:
    info = get_block_stage_description(datum)
    key = dict_to_string(info)
    if key not in keys:
      episode_idx += 1
      keys.add(key)
    info["user_episode_idx"] = episode_idx

    updated_key = dict_to_string(info)
    grouped_data[updated_key].append(datum)
    infos[updated_key] = info
  return grouped_data, infos


def compute_overlap(map1: np.ndarray, map2: np.ndarray, final_t: int = None):
  """map1: HxW, map2: HxW"""
  """Calculate the overlap between two maps."""
  nonzero_indices = np.argwhere(map1 > 0)
  values_map1 = (map1[nonzero_indices[:, 0], nonzero_indices[:, 1]] > 0).astype(
    np.float32
  )
  values_map2 = (map2[nonzero_indices[:, 0], nonzero_indices[:, 1]] > 0).astype(
    np.float32
  )

  overlap = ((values_map1 + values_map2) > 1)[:final_t]
  if final_t is not None:
    overlap = overlap[-final_t:]
  return overlap


def add_reuse_columns(df: DataFrame, overlap_threshold=0.15) -> DataFrame:
  """Add 'reuse' and 'overlap' columns to the DataFrame.

  Args:
      df (DataFrame): Input DataFrame
      overlap_threshold (float, optional): Threshold for path reuse. Defaults to 0.15.

  Returns:
      DataFrame: DataFrame with added 'reuse' and 'overlap' columns
  """
  # Create dictionaries to store values
  reuse_dict = {}
  overlap_dict = {}

  def update_reuse_dict(train_mazes, test_mazes):
    for train_maze, test_maze in zip(train_mazes, test_mazes):
      # Get train episodes
      test = df.filter(name=test_maze, eval=True)
      start_pos = test["start_pos"].to_list()[0]

      train = df.filter(
        name=train_maze, room=0, eval=False, success=1, start_pos=start_pos
      )

      if len(train.episodes) == 0:
        print(f"No training episodes for {(train_maze, test_maze)}")
        continue

      # Create map for training episodes
      train_map = create_maps(train.episodes).sum(0)

      # Get test episodes

      # Process each test episode
      for idx, row in enumerate(test._df.iter_rows(named=True)):
        global_index = row["global_episode_idx"]
        episode = test.episodes[idx]
        # Create map for single test episode
        test_map = create_maps([episode]).sum(0)
        overlap = compute_overlap(train_map, test_map)
        overlap_mean = overlap.mean()

        # Store both raw overlap and binary reuse values
        episode_id = (test_maze, global_index)
        overlap_dict[episode_id] = overlap_mean
        reuse_dict[episode_id] = overlap_mean > overlap_threshold

  all_mazes = df["name"].unique()
  train_mazes = sorted([m for m in all_mazes if "training" in m])
  test_mazes = sorted([m for m in all_mazes if "eval" in m])

  assert len(train_mazes) + len(test_mazes) == len(all_mazes)

  update_reuse_dict(train_mazes, test_mazes)

  # Create Series for both columns
  reuse_values = pl.Series(
    [
      reuse_dict.get((row["name"], row["global_episode_idx"]), None)
      for row in df.iter_rows(named=True)
    ]
  )
  overlap_values = pl.Series(
    [
      overlap_dict.get((row["name"], row["global_episode_idx"]), None)
      for row in df.iter_rows(named=True)
    ]
  )

  # Add both columns to the DataFrame
  new_df = df.with_columns(
    [
      pl.Series("reuse", reuse_values).cast(pl.Boolean),
      pl.Series("overlap", overlap_values).cast(pl.Float64),
    ]
  )

  return new_df


async def make_episode_data(
  file: str,
  example_timestep: TimeStep,
  debug: bool = False,
  overwrite_episode_data: bool = False,
  overwrite_episode_info: bool = False,
  verbose: bool = False,
) -> Tuple[pl.DataFrame, EpisodeData]:
  """This groups all of the data by block/stage information and prepares
      (1) a list of EpisodeData objects per block/stage
      (2) a dataframe which summarizes all episode information.

  The dataframe can be used to get indices into the list of EpisodeData for further computation.
  """
  try:
    data = await nicewebrl.read_all_records(file)
  except Exception as e:
    logging.warning(f"Failed to read records from {file}: {str(e)}")
    return file, None, None

  if len(data) == 0:
    return file, None, None

  file_metadata = data[-1]
  finished = file_metadata.get("finished", False)
  if not finished:
    return file, None, None
  if debug:
    n = max(1, int(len(data) * 0.05))
    data = data[:n]

  #####################
  # filenames
  #####################
  user_filename = file.split("/")[-1].split(".json")[0]
  base_path = file.split(user_filename)[0]
  if debug:
    episode_data_filename = f"{base_path}/{user_filename}_debug_episode_data.bytes"
    episode_info_filename = f"{base_path}/{user_filename}_debug_episode_info.csv"
  else:
    episode_data_filename = f"{base_path}/{user_filename}_episode_data.bytes"
    episode_info_filename = f"{base_path}/{user_filename}_episode_info.csv"

  #####################
  # filter out practice not-manipulation data
  #####################
  def filter_fn(datum):
    if "metadata" not in datum:
      return True
    desc = datum["metadata"]["block_metadata"]["desc"]
    manipulation = datum["metadata"]["block_metadata"].get("manipulation", None)
    if manipulation is None:
      return True
    if "practice" in desc:
      return True

    return False

  nbefore = len(data)
  data = [datum for datum in data if not filter_fn(datum)]
  if len(data) == 0:
    return file, None, None

  if verbose:
    print(f"Filtered {nbefore - len(data)} data points")

  #####################
  # separate data by block/stage
  #####################
  gds, gd_infos = separate_data_by_block_stage(data)
  idxs = [k["user_episode_idx"] for k in gd_infos.values()]
  assert len(idxs) == len(set(idxs)), (
    f"user_episode_idx is not unique. {len(idxs)} vs {len(set(idxs))}. max={max(idxs)}"
  )
  #####################
  # Load or create episode_data
  #####################
  import ipdb

  ipdb.set_trace()
  example_timestep = example_timestep.replace(
    state=jax.tree_map(lambda t: t[:1], example_timestep.state)
  )
  if os.path.exists(episode_data_filename) and not overwrite_episode_data:
    with open(episode_data_filename, "rb") as f:
      serialized_data = f.read()
      # Create template episode for deserialization
      example_episode = EpisodeData(
        actions=jnp.zeros((1,)),
        positions=jnp.zeros((1, 2)),
        timesteps=example_timestep,
        reaction_times=None,
        transitions=None,
      )
      # Two-step deserialization
      attempt1 = serialization.from_bytes(None, serialized_data)
      nepisodes = len(attempt1)
      episode_data = serialization.from_bytes(
        [example_episode] * nepisodes, serialized_data
      )
      print(f"Loaded episode data from {episode_data_filename}")
  else:
    episode_data = [None] * len(gds.keys())

    def get_timestep(datum, example_timestep):
      timestep = datum["data"]["timestep"]
      timestep = serialization.from_bytes(example_timestep, timestep)
      return timestep

    for key in gds.keys():
      red = raw_episode_data = gds[key]
      actions = jnp.asarray([datum["data"]["action_idx"] for datum in red])
      timesteps = [get_timestep(datum, example_timestep) for datum in red]
      timesteps = jtu.tree_map(lambda *v: jnp.stack(v), *timesteps)

      expected_step_num = jnp.arange(len(get_step_number(timesteps)))
      correct = jnp.all(get_step_number(timesteps) == expected_step_num)
      if not correct:
        # Get sorting indices
        sort_indices = jnp.argsort(timesteps.state.step_num)

        # Check if sorting fixes the sequence
        sorted_steps = timesteps.state.step_num[sort_indices]
        if jnp.all(sorted_steps == expected_step_num):
          # Fix the ordering of all relevant data
          actions = actions[sort_indices]
          timesteps = jtu.tree_map(
            lambda x: (
              x[sort_indices] if isinstance(x, (jnp.ndarray, np.ndarray)) else x
            ),
            timesteps,
          )
          logging.info(
            f"{user_filename}: Fixed step indices for episode {key} through sorting"
          )
        else:
          logging.warning(
            f"{user_filename}: Skipping episode {key} due to invalid step indices that cannot be fixed"
          )
          raise RuntimeError(
            f"{user_filename}: episode {key} has faulty step indices: {timesteps.state.step_num}"
          )
      positions = get_agent_position(timesteps)
      episode_idx = gd_infos[key]["user_episode_idx"]
      episode_data[episode_idx] = EpisodeData(
        actions=actions,
        positions=positions,
        timesteps=timesteps,
      )
    # Save using Flax serialization
    with open(episode_data_filename, "wb") as f:
      serialized_data = serialization.to_bytes(episode_data)
      f.write(serialized_data)

  #####################
  # Load or create episode_info
  #####################

  if os.path.exists(episode_info_filename) and not overwrite_episode_info:
    episode_info = pl.read_csv(episode_info_filename)
    print(f"Loaded episode info from {episode_info_filename}")
  else:
    # --------------
    # first make df with raw data from file
    # --------------
    episode_info = [None] * len(gds.keys())
    for key in gds.keys():
      raw_episode_data = gds[key]
      episode_idx = gd_infos[key]["user_episode_idx"]
      timesteps = episode_data[episode_idx].timesteps

      episode_info[episode_idx] = make_row(
        datum=raw_episode_data[0],
        episode_info=gd_infos[key],
        timesteps=timesteps,
        file=file,
        user_storage=file_metadata["user_storage"],
      )

      reaction_times = [compute_reaction_time(datum) for datum in raw_episode_data]
      reaction_times = jnp.asarray(reaction_times)

      episode_data[episode_idx] = episode_data[episode_idx]._replace(
        reaction_times=reaction_times,
      )

    episode_info = pl.DataFrame(episode_info)
    # --------------
    # next, augment df with success, termination, first_rt, avg_rt, total_rt
    # --------------

    def success(e: EpisodeData):
      rewards = e.timesteps.reward
      # return rewards
      assert rewards.ndim == 1, "this is only defined over vector, e.g. 1 episode"
      success = rewards > 0.5
      return success.any().astype(np.float32)

    def features_achieved(e):
      features = e.timesteps.state.achievements
      achieved = features.sum(-1) > 0
      return achieved.any().astype(np.float32)

    def terminated(e):
      return features_achieved(e)

    def get_rt(e: EpisodeData):
      return np.log(e.reaction_times + 1e-5)

    def total_rt(e: EpisodeData):
      return np.sum(get_rt(e)[:-1])

    def avg_rt(e: EpisodeData):
      return np.mean(get_rt(e)[:-1])

    def first_rt(e: EpisodeData):
      return get_rt(e)[0]

    def max_rt(e: EpisodeData):
      return np.max(get_rt(e)[:-1])

    def max_post_rt(e: EpisodeData):
      return np.max(get_rt(e)[1:-1])

    def max_init_post_rt(e: EpisodeData):
      n = len(get_rt(e)[:-1]) // 2 + 1
      return np.max(get_rt(e)[1:n])

    def max_final_rt(e: EpisodeData):
      n = len(get_rt(e)[:-1]) // 2 + 1
      return np.max(get_rt(e)[-n:-1])

    def max_end_rt(e: EpisodeData):
      return np.max(get_rt(e)[-11:-1])

    def avg_post_rt(e: EpisodeData):
      return np.mean(get_rt(e)[1:-1])

    def path_length(e: EpisodeData):
      return len(e.actions[:-1])

    measures = {
      "success": success,
      "path_length": path_length,
      "termination": terminated,
      "log_first_rt": first_rt,
      #"log_avg_rt": avg_rt,
      #"log_total_rt": total_rt,
      #"log_avg_post_rt": avg_post_rt,
      #"log_max_rt": max_rt,
      #"log_max_post_rt": max_post_rt,
      #"log_max_init_post_rt": max_init_post_rt,
      #"log_max_end_rt": max_end_rt,
      #"log_max_final_rt": max_final_rt,
    }
    computed_values = {key: [] for key in measures}

    # Calculate values for each episode
    for episode in episode_data:
      for key, fn in measures.items():
        computed_values[key].append(fn(episode))

    # Create a new DataFrame with the additional columns
    episode_info = episode_info.with_columns(
      [pl.Series(key, values) for key, values in computed_values.items()]
    )
    episode_info = episode_info.with_columns(
      pl.col("path_length")
      .sub(pl.col("optimal_length"))
      .alias("optimal_length_deviance")
    )
    _temp_df = DataFrame(episode_info, episode_data)
    _temp_df = add_reuse_columns(_temp_df, overlap_threshold=0.15)
    episode_info = _temp_df._df

    episode_info.write_csv(episode_info_filename)

    # Save using Flax serialization
    with open(episode_data_filename, "wb") as f:
      serialized_data = serialization.to_bytes(episode_data)
      f.write(serialized_data)

  return file, episode_info, episode_data


async def make_all_episode_data(
  files,
  example_timestep,
  debug=False,
  overwrite_episode_data=False,
  overwrite_episode_info=False,
):
  """Synchronous version of make_all_episode_data that processes files sequentially."""

  if debug:
    files = files[: max(int(len(files) * 0.1), 10)]

  all_episode_data = []
  episode_df_list = []

  # Process files sequentially
  for enum, file in enumerate(files):
    try:
      # Convert the async make_episode_data to sync by running it in an event loop
      file, episode_df, episode_data = await make_episode_data(
        file,
        example_timestep,
        overwrite_episode_data=overwrite_episode_data,
        overwrite_episode_info=overwrite_episode_info,
        debug=debug,
        )

      print(f"{enum}/{len(files)}", file)

      if episode_df is not None and episode_data is not None:
        all_episode_data.extend(episode_data)
        episode_df_list.append(episode_df)
      else:
        print(f"skipping {file} because one of the episode_df or episode_data is None")

    except Exception as e:
      print(f"error processing {file}: {str(e)}")
      continue

  episode_df = pl.concat(episode_df_list, how="diagonal_relaxed")
  return DataFrame(episode_df, all_episode_data)


def create_maps(episode_data_list: List[EpisodeData]):
  maps = []

  for episode_data in episode_data_list:
    timesteps = episode_data.timesteps

    # [T, N, H, W]
    # Assuming grid is 3D with time as first dimension
    grid_shape = timesteps.state.map.shape

    # skip the time dimension and final channel dimension
    grid = jnp.zeros(grid_shape[2:], dtype=jnp.int32)

    # go through each position and set the corresponding index to 1
    for pos in episode_data.positions:
      grid = grid.at[pos[0], pos[1]].set(1)
    maps.append(grid)
  return np.array(maps)


async def get_human_data(
  valid_files,
  overwrite_episode_data=False,
  overwrite_episode_info=True,
  debug=False,
):
  import craftax_experiment_structure as experiment

  example_web_timestep = experiment.jax_web_env.reset(
    jax.random.PRNGKey(0), experiment.dummy_params
  )

  ################
  # Load data
  ################
  initial_user_df = await make_all_episode_data(
    files=valid_files,
    example_timestep=example_web_timestep,
    overwrite_episode_data=overwrite_episode_data,
    overwrite_episode_info=overwrite_episode_info,
    debug=debug,
  )

  def bad_episode(e):
    empty = len(e.actions) == 0
    empty |= (np.array(e.actions[0]) == -1).any()
    remove = empty
    return remove

  initial_user_df = initial_user_df.filter(episode_filter=bad_episode)

  return initial_user_df


async def main():
  # Define searches
  data_dir = "/Users/wilka/git/research/results/human_dyna_craftax/"

  searches = {
    "Paths": f"{data_dir}/user_data/*exps*/*v1*paths*.json",
    "Start": f"{data_dir}/user_data/*exps*/*v1*juncture*.json",
  }
  files = []
  for v in searches.values():
    files.extend(glob(v))

  # valid_files = get_valid_files(searches, verbose=True, plot=False)
  user_df = await get_human_data(
    files, overwrite_episode_data=False, overwrite_episode_info=True
  )
  return user_df


if __name__ == "__main__":
  import asyncio

  user_df = asyncio.run(main())
