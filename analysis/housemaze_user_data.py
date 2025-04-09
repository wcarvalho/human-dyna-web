from joblib import Parallel, delayed
from typing import Optional
import polars as pl
import json
from glob import glob
import os.path
from collections import defaultdict
from typing import NamedTuple, List
from flax import struct
from datetime import datetime
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from absl import logging
from flax import serialization
import nicewebrl

import jax.tree_util as jtu
from housemaze import utils
from housemaze.human_dyna import multitask_env
from housemaze.human_dyna import mazes
from housemaze.human_dyna import web_env

from nicewebrl import nicejax
from nicewebrl.dataframe import DataFrame

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

image_data = utils.load_image_dict()
image_keys = image_data["keys"]

groups = [
  # room 1
  [image_keys.index("orange"), image_keys.index("potato")],
  # room 2
  [image_keys.index("knife"), image_keys.index("spoon")],
  # room 3
  # [image_keys.index('tomato'), image_keys.index('lettuce')],
]
groups = np.array(groups, dtype=np.int32)
task_objects = groups.reshape(-1)

char2idx = mazes.groups_to_char2key(groups)


class EpisodeData(NamedTuple):
  actions: jax.Array
  timesteps: multitask_env.TimeStep
  positions: jax.Array = None
  reaction_times: jax.Array = None
  transitions: struct.PyTreeNode = None


def success(e: EpisodeData):
  rewards = e.timesteps.reward
  # return rewards
  assert rewards.ndim == 1, "this is only defined over vector, e.g. 1 episode"
  success = rewards > 0.5
  return success.any().astype(np.float32)


def reversal_label(reversal):
  if reversal == [False, False]:
    return "F,F"
  elif reversal == [True, False]:
    return "T,F"
  elif reversal == [False, True]:
    return "F,T"
  elif reversal == [True, True]:
    return "T,T"
  else:
    raise ValueError(f"reversal: {reversal}")


def make_env_params(maze_str):
  return mazes.get_maze_reset_params(
    groups=groups,
    char2key=char2idx,
    maze_str=maze_str,
    randomize_agent=False,
    make_env_params=True,
  )


def read_dict_list_from_file(filename: str):
  dictionaries = []
  with open(filename, "r") as f:
    for line in f:
      # Parse each line as a JSON object (dictionary) and append it to the list
      dictionaries.append(json.loads(line.strip()))
  return dictionaries


def user_id_from_filename(filename: str):
  return int(filename.split("/")[-1].split(".")[0].split("_")[0].split("=")[1])


def compute_experiment_lengths(
  files, plot: bool = False, condition_name: str = "", verbose: bool = False
):
  experiment_lengths = {}

  for file in files:
    data = read_dict_list_from_file(file)
    user_id = user_id_from_filename(file)
    if len(data) < 2 or not data[-1].get("finished", False):
      if verbose:
        print(f"Skipping {user_id} because it's not finished")
      continue

    try:
      start_time = datetime.strptime(
        data[0]["data"]["image_seen_time"], "%Y-%m-%dT%H:%M:%S.%fZ"
      )

      if "noticed_difference" in data[-2]["data"].keys():
        end_time = datetime.strptime(
          data[-3]["data"]["action_taken_time"], "%Y-%m-%dT%H:%M:%S.%fZ"
        )
      else:
        end_time = datetime.strptime(
          data[-2]["data"]["action_taken_time"], "%Y-%m-%dT%H:%M:%S.%fZ"
        )
    except Exception as e:
      print(f"Error processing file {file}: {e}")
      raise e

    total_length = (end_time - start_time).total_seconds() / 60  # Convert to minutes

    user_id = data[0]["user_data"]["user_id"]
    experiment_lengths[user_id] = total_length

  if plot:
    lengths = np.array(list(experiment_lengths.values()))
    mean = np.mean(lengths)
    std = np.std(lengths)

    plt.figure(figsize=(10, 6))
    sns.histplot(lengths, kde=True)
    plt.axvline(mean, color="r", linestyle="--", label="Mean")
    plt.axvline(mean + 2 * std, color="g", linestyle="--", label="2 Std Dev")
    plt.axvline(mean + 3 * std, color="b", linestyle="--", label="3 Std Dev")
    plt.axvline(mean - 2 * std, color="g", linestyle="--")
    plt.axvline(mean - 3 * std, color="b", linestyle="--")
    plt.title(f"Distribution of Experiment Lengths - {condition_name}")
    plt.xlabel("Experiment Length (minutes)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

  return experiment_lengths


def get_valid_files(
  searches, filter: bool = False, plot: bool = False, verbose: bool = False
):
  all_valid_files = {}

  for condition_name, search in searches.items():
    files = list(set(glob(search)))
    if filter:
      experiment_lengths = compute_experiment_lengths(
        files, condition_name=condition_name, plot=plot
      )

      # Calculate mean and standard deviation
      lengths = np.array(list(experiment_lengths.values()))
      mean = np.mean(lengths)
      std = np.std(lengths)

      # Filter files within 3 standard deviations
      def good_user(file):
        user_id = user_id_from_filename(file)
        if user_id not in experiment_lengths:
          if verbose:
            print(f"User {user_id} not in experiment_lengths")
          return False
        user_val = experiment_lengths[user_id]
        good = abs(user_val - mean) <= 3 * std
        if not good:
          if verbose:
            print(
              f"User {user_id} > 3 std. x: {user_val}, mean: {mean}, 3*std: {mean + 3 * std}"
            )
        return good

      valid_files = [file for file in files if good_user(file)]
    else:
      valid_files = files

    all_valid_files[condition_name] = valid_files
    print(f"{condition_name}: {len(valid_files)}/{len(files)} valid files")

  # Convert the dictionary of valid files to a long list
  all_valid_files_list = []
  for condition_files in all_valid_files.values():
    all_valid_files_list.extend(condition_files)

  return all_valid_files_list


def get_timestep(datum, example_timestep):
  timestep = datum["data"]["timestep"]
  timestep = serialization.from_bytes(example_timestep, timestep)
  return timestep
  # timestep = nicejax.deserialize_bytes(
  #  cls=multitask_env.TimeStep, encoded_data=datum["data"]["timestep"]
  # )

  ## `deserialize_bytes` infers the types so it might be slightly wrong. you can enforce the correct types by matching them to example data.
  # timestep = nicejax.match_types(example=example_timestep, data=timestep)

  # return timestep


def time_diff(t1, t2) -> float:
  # Convert string timestamps to datetime objects
  t1 = datetime.strptime(t1, "%Y-%m-%dT%H:%M:%S.%fZ")
  t2 = datetime.strptime(t2, "%Y-%m-%dT%H:%M:%S.%fZ")

  # Calculate the time difference
  time_difference = t2 - t1

  # Convert the time difference to milliseconds
  return time_difference.total_seconds()


def compute_reaction_time(datum) -> float:
  # Calculate the time difference
  return time_diff(datum["data"]["image_seen_time"], datum["data"]["action_taken_time"])


def get_task_object(timesteps: multitask_env.TimeStep):
  return timesteps.state.task_object[0]


def get_task_room(timesteps: multitask_env.TimeStep, task_groups=None):
  task_object = get_task_object(timesteps)
  task_groups = task_groups or groups
  # Find the room (row) that contains the task object
  task_room = next((i for i, row in enumerate(task_groups) if task_object in row), None)
  return task_room


def dict_to_string(data):
  # Convert each key-value pair to "key=value" format
  pairs = [f"{key}={value}" for key, value in data.items()]

  # Join all pairs with ", " separator
  return ", ".join(pairs)


def get_block_stage_description(datum):
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
  reversal = datum["metadata"]["block_metadata"].get("reversal", [False, False])
  return dict(
    maze=datum["metadata"].get("maze"),
    condition=datum["metadata"].get("condition", 0),
    name=datum["name"],
    block=block_desc,
    manipulation=block_manipulation,
    episode_idx=datum["metadata"]["nepisodes"],
    eval=datum["metadata"]["eval"],
    reversal=reversal_label(reversal),
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
    if not key in keys:
      episode_idx += 1
      keys.add(key)
    info["user_episode_idx"] = episode_idx

    updated_key = dict_to_string(info)
    grouped_data[updated_key].append(datum)
    infos[updated_key] = info
  return grouped_data, infos


def make_row(
  datum: dict,
  timesteps: multitask_env.TimeStep,
  file: str,
  episode_info: Optional[dict],
  user_storage: dict,
):
  """THIS IS WHERE YOU'LL WANT TO INSERT OTHER EPISODE LEVEL INFO TO TRACK IN DATAFRAME!!!

  Args:
      datum (dict): _description_
      timesteps (multitask_env.TimeStep): _description_
      file (str): _description_

  Returns:
      _type_: _description_
  """
  groups = datum["metadata"]["block_metadata"].get("groups")
  row = dict(
    maze=datum["metadata"].get("maze"),
    condition=datum["metadata"].get("condition", 0),
    name=datum["name"],
    block=datum["metadata"]["block_metadata"]["desc"],
    manipulation=datum["metadata"]["block_metadata"].get("manipulation", None),
    global_episode_idx=episode_info["user_episode_idx"],
    episode_idx=datum["metadata"]["nepisodes"],
    eval=datum["metadata"]["eval"],
    task=int(get_task_object(timesteps)),
    room=int(get_task_room(timesteps, task_groups=groups)),
    start_pos=str(timesteps.state.agent_pos[0]),
  )
  row.update(datum["user_data"])
  row.update(user_storage['user_info'])
  ##########
  # get experiment name from file
  ##########
  # '/path/data_user=3712207029_name=exp3-v2-r1-t30_exp=3_debug=0.json'
  # e.g. ['data', 'user=3712207029', 'name=exp3-v2-r1-t30', 'exp=3', 'debug=0.json']
  pieces = os.path.splitext(os.path.basename(file))[0].split("_")
  pieces = [p.split("=") for p in pieces if "=" in p]
  new_vals = {p[0]: p[1] for p in pieces}
  # Rename 'name' key to 'exp_name' if it exists
  if "name" in new_vals:
    new_vals["exp_name"] = new_vals.pop("name")
  row.update(new_vals)

  ####################
  # add version, tell_reuse, timer
  ####################
  # name = new_vals.get("exp_name")
  # if name is not None:
  #  # example 'exp4-v1-r1-t0-plan'
  #  # split on '-' and take the first element
  #  # if v--> version
  #  # if r--> tell_reuse
  #  # if t--> timer
  #  # if there's a word at the end, it's the manipulation
  #  # create a dictionary according to this legend
  #  legend = dict(v="version", r="tell_reuse", t="timer")
  #  name_info = dict()
  #  for k, v in legend.items():
  #    if k in name:
  #      name_info[v] = name.split(k)[1].split("-")[0]
  #  print(name_info)
  #  row.update(name_info)
  ## Convert all numeric strings to integers
  # for key, value in row.items():
  #  if isinstance(value, str) and value.isdigit():
  #    row[key] = int(value)
  env_vars = user_storage.get("env_vars", {})
  row["tell_reuse"] = int(env_vars.get("SAY_REUSE", 0))
  reversal = datum["metadata"]["block_metadata"].get("reversal", [False, False])
  row["reversal"] = reversal_label(reversal)

  ####################
  # add optimal path length
  ####################
  path = utils.find_optimal_path(
    grid=timesteps.state.grid[0],  # first time-step
    agent_pos=tuple([int(i) for i in timesteps.state.agent_pos[0]]),
    goal=timesteps.state.task_object[0],
  )
  row["optimal_length"] = len(path) - 1  # includes done

  return row


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
  """Add a 'reuse' column to the DataFrame indicating whether each episode reused paths.

  TODO: move this function into make_episode_data at end. just once per user. then saved.c
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

  def update_reuse_dict(train_mazes, test_mazes):
    # Get unique users

    for train_maze, test_maze in zip(train_mazes, test_mazes):
      # Get train episodes
      train = df.filter(maze=train_maze, room=0, eval=False, success=1)

      if len(train.episodes) == 0:
        continue

      # Create map for training episodes
      train_map = create_maps(train.episodes).sum(0)

      # Get test episodes
      test = df.filter(maze=test_maze, eval=True)

      # Process each test episode
      for idx, row in enumerate(test._df.iter_rows(named=True)):
        global_index = row["global_episode_idx"]
        episode = test.episodes[idx]
        # Create map for single test episode
        test_map = create_maps([episode]).sum(0)
        overlap = compute_overlap(train_map, test_map)

        # Store the reuse value
        episode_id = (test_maze, global_index)
        reuse_dict[episode_id] = int(overlap.mean() > overlap_threshold)

  # -----------------
  # paths manipulation (3)
  # -----------------
  # Define mazes if not provided
  # manipulation = 3
  train_mazes = test_mazes = [
    "big_m3_maze1_(F,F)",
    "big_m3_maze1_(F,T)",
    "big_m3_maze1_(T,F)",
    "big_m3_maze1_(T,T)",
  ]
  update_reuse_dict(train_mazes, test_mazes)
  # -----------------
  # shortcut manipulation (1)
  # -----------------
  # Define mazes if not provided
  # manipulation = 1
  train_mazes = [
    "big_m1_maze3_(F,F)",
    "big_m1_maze3_(F,T)",
    "big_m1_maze3_(T,F)",
    "big_m1_maze3_(T,T)",
  ]

  test_mazes = [
    "big_m1_maze3_shortcut_(F,F)",
    "big_m1_maze3_shortcut_(F,T)",
    "big_m1_maze3_shortcut_(T,F)",
    "big_m1_maze3_shortcut_(T,T)",
  ]
  update_reuse_dict(train_mazes, test_mazes)

  # -----------------
  # add everything
  # -----------------
  # Create a new column with reuse values
  reuse_values = pl.Series(
    [
      reuse_dict.get((row["maze"], row["global_episode_idx"]), None)
      for row in df.iter_rows(named=True)
    ]
  )
  # Add the new column to the DataFrame
  new_df = df.with_columns([pl.Series("reuse", reuse_values).cast(pl.Int32)])

  return new_df


def make_episode_data(
  file: str,
  example_timestep: multitask_env.TimeStep,
  debug: bool = False,
  overwrite_episode_data: bool = False,
  overwrite_episode_info: bool = False,
  verbose: bool = False,
  require_finished: bool = True,
):
  """This groups all of the data by block/stage information and prepares
      (1) a list of EpisodeData objects per block/stage
      (2) a dataframe which summarizes all episode information.

  The dataframe can be used to get indices into the list of EpisodeData for further computation.
  """
  try:
    data = nicewebrl.read_all_records_sync(file)
  except Exception as e:
    logging.warning(f"Failed to read records from {file}: {str(e)}")
    return None, None

  if len(data) < 2:
    return None, None

  file_metadata = data[-1]
  finished = file_metadata.get("finished", False)
  if require_finished and not finished:
    return None, None
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
  else:
    episode_data = [None] * len(gds.keys())
    for key in tqdm(gds.keys(), desc="Processing episodes"):
      red = raw_episode_data = gds[key]
      actions = jnp.asarray([datum["data"]["action_idx"] for datum in red])
      timesteps = [get_timestep(datum, example_timestep) for datum in red]
      timesteps = jtu.tree_map(lambda *v: jnp.stack(v), *timesteps)

      expected_step_num = jnp.arange(len(timesteps.state.step_num))
      correct = jnp.all(timesteps.state.step_num == expected_step_num)
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
      positions = timesteps.state.agent_pos
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
        user_storage=file_metadata.get("user_storage", {}),
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
      features = e.timesteps.state.task_state.features
      achieved = features.sum(-1) > 0
      return achieved.any().astype(np.float32)

    def terminated(e):
      return features_achieved(e)

    def get_log_rt(e: EpisodeData):
      return np.log(1000 * e.reaction_times + 1e-5)

    def total_rt(e: EpisodeData):
      return np.sum(e.reaction_times[:-1])

    def log_total_rt(e: EpisodeData):
      return np.sum(get_log_rt(e)[:-1])

    def log_avg_rt(e: EpisodeData):
      return np.mean(get_log_rt(e)[:-1])

    def log_first_rt(e: EpisodeData):
      return get_log_rt(e)[0]

    def log_max_rt(e: EpisodeData):
      return np.max(get_log_rt(e)[:-1])

    def log_max_post_rt(e: EpisodeData):
      return np.max(get_log_rt(e)[1:-1])

    def log_max_init_post_rt(e: EpisodeData):
      n = len(get_log_rt(e)[:-1]) // 2 + 1
      return np.max(get_log_rt(e)[1:n])

    def log_max_final_rt(e: EpisodeData):
      n = len(get_log_rt(e)[:-1]) // 2 + 1
      return np.max(get_log_rt(e)[-n:-1])

    def log_max_end_rt(e: EpisodeData):
      return np.max(get_log_rt(e)[-11:-1])

    def log_avg_post_rt(e: EpisodeData):
      return np.mean(get_log_rt(e)[1:-1])

    def path_length(e: EpisodeData):
      return len(e.actions[:-1])

    measures = {
      "success": success,
      "path_length": path_length,
      "termination": terminated,
      "log_first_rt": log_first_rt,
      "first_rt": lambda e: e.reaction_times[0],  # Convert to milliseconds
      "log_avg_rt": log_avg_rt,
      "avg_rt": lambda e: np.mean(e.reaction_times[:-1]),  # Non-log version in ms
      "log_total_rt": log_total_rt,
      "total_rt": lambda e: np.sum(e.reaction_times[:-1]),  # Already existed, now in ms
      "log_avg_post_rt": log_avg_post_rt,
      "avg_post_rt": lambda e: np.mean(e.reaction_times[1:-1]),  # Non-log version in ms
      "log_max_rt": log_max_rt,
      "max_rt": lambda e: np.max(e.reaction_times[:-1]),  # Non-log version in ms
      "log_max_post_rt": log_max_post_rt,
      #"max_post_rt": lambda e: np.max(e.reaction_times[1:-1]),  # Non-log version in ms
      "log_max_init_post_rt": log_max_init_post_rt,
      #"max_init_post_rt": lambda e: np.max(e.reaction_times[1:(len(e.reaction_times[:-1]) // 2 + 1)]),  # Non-log version in ms
      "log_max_end_rt": log_max_end_rt,
      #"max_end_rt": lambda e: np.max(e.reaction_times[-11:-1]),  # Non-log version in ms
      "log_max_final_rt": log_max_final_rt,
      #"max_final_rt": lambda e: np.max(e.reaction_times[-(len(e.reaction_times[:-1]) // 2 + 1):-1]),  # Non-log version in ms
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

  return episode_info, episode_data


def make_all_episode_data(
  files,
  example_timestep,
  debug=False,
  overwrite_episode_data=False,
  overwrite_episode_info=False,
  require_finished: bool = True,
  parallel: bool = True,
):
  def process_file(file):
    try:
      return make_episode_data(
        file,
        example_timestep,
        overwrite_episode_data=overwrite_episode_data,
        overwrite_episode_info=overwrite_episode_info,
        debug=debug,
      require_finished=require_finished,
      )
    except Exception as e:
      logging.error(f"Error processing file {file}: {e}")
      return None, None

  if debug:
    files = files[: max(int(len(files) * 0.1), 10)]
  if parallel:
    results = Parallel(n_jobs=-1)(delayed(process_file)(file) for file in files)
  else:
    results = [process_file(file) for file in files]

  all_episode_data = []
  episode_df_list = []

  for episode_df, episode_data in tqdm(
    results, desc="Combining results", total=len(files)
  ):
    if episode_df is not None and episode_data is not None:
      all_episode_data.extend(episode_data)
      episode_df_list.append(episode_df)

  episode_df = pl.concat(episode_df_list, how="diagonal_relaxed")

  return DataFrame(episode_df, all_episode_data)


def create_maps(episode_data_list: List[EpisodeData]):
  maps = []
  for episode_data in episode_data_list:
    timesteps = episode_data.timesteps

    # [T, H, W, 1]
    # Assuming grid is 3D with time as first dimension
    grid_shape = timesteps.state.grid.shape

    # skip the time dimension and final channel dimension
    grid = jnp.zeros(grid_shape[1:-1], dtype=jnp.int32)

    # go through each position and set the corresponding index to 1
    for pos in episode_data.positions:
      grid = grid.at[pos[0], pos[1]].set(1)
    maps.append(grid)
  return np.array(maps)


def get_human_data(
  valid_files,
  overwrite_episode_data=False,
  overwrite_episode_info=True,
  load_df_only: bool = True,
  require_finished: bool = True,
  debug=False,
):
  from experiment_utils import SuccessTrackingAutoResetWrapper

  ################
  # Setup environment
  ################
  dummy_rng = jax.random.PRNGKey(42)
  dummy_env_params = make_env_params(mazes.big_practice_maze)
  task_runner = multitask_env.TaskRunner(task_objects=task_objects)
  base_env = web_env.HouseMaze(
    task_runner=task_runner,
    num_categories=200,
  )
  end = SuccessTrackingAutoResetWrapper(base_env)
  example_web_timestep = end.reset(dummy_rng, dummy_env_params)

  data_dir = "/Users/wilka/git/research/results/human_dyna/"
  if debug:
    df_location = os.path.join(data_dir, "user_data/exps/all_data_debug.csv")
  else:
    df_location = os.path.join(data_dir, "user_data/exps/all_data.csv")
  ################
  # Load data
  ################

  # Try to load existing dataframe if load_df_only is True
  if load_df_only and os.path.exists(df_location) and not overwrite_episode_info:
    try:
      import polars as pl
      print(f"Loading existing dataframe from {df_location}")
      return pl.read_csv(df_location)
    except Exception as e:
      print(f"Failed to load existing dataframe: {e}")
      # Fall through to creating a new dataframe

  initial_user_df = make_all_episode_data(
    files=valid_files,
    example_timestep=example_web_timestep,
    overwrite_episode_data=overwrite_episode_data,
    overwrite_episode_info=overwrite_episode_info,
    require_finished=require_finished,
    debug=debug,
  )

  def bad_episode(e):
    empty = len(e.actions) == 0
    empty |= (np.array(e.actions[0]) == -1).any()
    remove = empty
    return remove

  initial_user_df = initial_user_df.filter(episode_filter=bad_episode)

  # Save the dataframe for future use
  initial_user_df._df.write_csv(df_location)
  if load_df_only:
    return initial_user_df._df

  return initial_user_df


if __name__ == "__main__":
  # Define searches
  data_dir = "/Users/wilka/git/research/results/human_dyna/"

  searches = {
    "Paths": f"{data_dir}/user_data/*exps*/*v1*paths*.json",
    "Path-notell": f"{data_dir}/user_data/*exps*/*v2*r0*paths*.json",
    "Start": f"{data_dir}/user_data/*exps*/*v3*start*.json",
    #'Start-notell': f'{data_dir}/user_data/*exps*/*v2*r0*start*.json',
    "Plan (Tell)": f"{data_dir}/user_data/*exps*/*v2*r1-t0-plan*.json",
    "Plan (Don't Tell)": f"{data_dir}/user_data/*exps*/*v2*r0-t0-plan*.json",
    "Shortcut": f"{data_dir}/user_data/*exps*/*v3*shortcut*.json",
    #'Shortcut-notell': f'{data_dir}/user_data/*exps*/*v2*r0*shortcut*.json',
  }

  valid_files = get_valid_files(searches, verbose=True, plot=False)
  user_df = get_human_data(
    valid_files, overwrite_episode_data=False, overwrite_episode_info=True
  )
