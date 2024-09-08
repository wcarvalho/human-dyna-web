import polars as pl
from collections import defaultdict
import json
from typing import List, Dict, Any, NamedTuple
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
from nicewebrl import nicejax
from housemaze.human_dyna import multitask_env as maze
import jax
import jax.numpy as jnp
import numpy as np
import experiment_1 as experiment


class EpisodeData(NamedTuple):
    actions: jax.Array
    positions: jax.Array
    reaction_times: jax.Array
    timesteps: maze.TimeStep

########################
# Utilities for analyzing episode data
########################


def filter_episodes(episode_data_list: List[EpisodeData], df: pl.DataFrame, **kwargs):
    idxs = df.filter(**kwargs)['global_episode_idx']
    return [episode_data_list[i] for i in idxs]


def render_episode(timesteps: maze.TimeStep, t: int = 0):
    # pick idx to visualize
    timestep = jax.tree_map(lambda x: x[t], timesteps)
    image = experiment.render_fn(timestep)
    return image


def get_task_object(timesteps: maze.TimeStep):
    return timesteps.state.task_object[0]

def get_task_room(timesteps: maze.TimeStep):
    task_object = get_task_object(timesteps)
    # Find the room (row) that contains the task object
    task_room = next((i for i, row in enumerate(experiment.groups) if task_object in row), None)
    return task_room


def object_idx_to_name(object_idx: int):
    image_keys = experiment.image_keys
    return image_keys[object_idx]


def success_fn(timesteps: maze.TimeStep):
    # did any timestep have a reward > .5?
    rewards = timesteps.reward
    assert rewards.ndim == 1, 'this is only defined over vector, e.g. 1 episode'
    success = rewards > .5
    return success.any()


def create_maps(episode_data_list: List[EpisodeData]):
    maps = []
    for episode_data in episode_data_list:
        timesteps = episode_data.timesteps

        # [T, H, W, 1]
        grid_shape = timesteps.state.grid.shape  # Assuming grid is 3D with time as first dimension

        # skip the time dimension and final channel dimension
        grid = jnp.zeros(grid_shape[1:-1], dtype=jnp.int32)

        # go through each position and set the corresponding index to 1
        for pos in episode_data.positions:
            grid = grid.at[pos[0], pos[1]].set(1)
        maps.append(grid)
    return np.array(maps)


def overlap(map1: np.ndarray, map2: np.ndarray, final_t: int = None):
    """map1: HxW, map2: HxW"""
    """Calculate the overlap between two maps."""
    nonzero_indices = np.argwhere(map1 > 0)
    values_map1 = map1[nonzero_indices[:, 0], nonzero_indices[:, 1]]
    values_map2 = map2[nonzero_indices[:, 0], nonzero_indices[:, 1]]

    overlap = ((values_map1 + values_map2) > 1)[:final_t]
    if final_t is not None:
        overlap = overlap[-final_t:]
    return overlap

########################
# files for loading episode data
########################

def get_timestep(datum):
    timestep = nicejax.deserialize_bytes(
        cls=maze.TimeStep, encoded_data=datum['data'])

    # `deserialize_bytes` infers the types so it might be slightly wrong. you can enforce the correct types by matching them to example data.
    timestep = nicejax.match_types(
        example=experiment.dummy_timestep, data=timestep)

    return timestep

def dict_to_string(data):
    # Convert each key-value pair to "key=value" format
    pairs = [f"{key}={value}" for key, value in data.items()]

    # Join all pairs with ", " separator
    return ", ".join(pairs)

def get_block_stage_description(datum):
    ####################
    # block information
    ####################
    block_metadata = datum['metadata']['block_metadata']
    # e.g. manipulation = 4
    block_manipulation = block_metadata.get('manipulation', -1)
    # e.g. desc = 'off-task object regular'
    block_desc = block_metadata.get('desc', 'unknown')

    ####################
    # stage information
    ####################
    stage_desc = datum['metadata'].get('desc')

    return dict(
        stage=stage_desc,
        block=block_desc,
        manipulation=block_manipulation,
        episode_idx=datum['metadata']['episode_idx'],
        eval=datum['metadata']['eval'],
    )


def time_diff(t1, t2) -> float:
    # Convert string timestamps to datetime objects
    t1 = datetime.strptime(t1, '%Y-%m-%dT%H:%M:%S.%fZ')
    t2 = datetime.strptime(t2, '%Y-%m-%dT%H:%M:%S.%fZ')

    # Calculate the time difference
    time_difference = t2 - t1

    # Convert the time difference to milliseconds
    return time_difference.total_seconds() * 1000


def compute_reaction_time(datum) -> float:
    # Calculate the time difference
    return time_diff(datum['image_seen_time'], datum['action_taken_time'])


def load_experiment_data(file_path: str):
    with open(file_path, 'r') as f:
        data_dicts = json.load(f)

    data_dicts = [row for row in data_dicts if not 'practice' in row['metadata']['block_metadata']['desc']]
    datum = data_dicts[0]

    action_taken: int = datum['action_idx']
    image_seen_time: str = datum['image_seen_time']
    action_taken_time: str = datum['action_taken_time']
    reaction_time: float = compute_reaction_time(datum)

    print(f"Reaction time: {reaction_time:.3f} milliseconds")


    # as long as you give it the right class, it will deserialize it correctly
    timestep = nicejax.deserialize_bytes(cls=maze.TimeStep, encoded_data=datum['data'])

    # `deserialize_bytes` infers the types so it might be slightly wrong. you can enforce the correct types by matching them to example data.
    timestep = nicejax.match_types(
        example=experiment.dummy_timestep, data=timestep)

    # convert all jax arrays to numpy arrays
    timestep = jax.tree_map(lambda x: np.asarray(x), timestep)

    # Group data by block_metadata['manipulation'] and block_metadata['desc']
    grouped_data = defaultdict(list)
    for datum in data_dicts:
        block_metadata = datum['metadata']['block_metadata']
        # e.g. manipulation = 4
        manipulation = block_metadata.get('manipulation', -1)
        # e.g. desc = 'off-task object regular'
        desc = block_metadata.get('desc', 'unknown')
        key = f"{manipulation}. {desc}"

        grouped_data[key].append(datum)

    # Print summary of grouped data
    for key, group in grouped_data.items():
        print(f"Group: {key}")
        print(f"Number of entries: {len(group)}")
        print("---")

    
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    import sys
    import argparse
    import os

    def parse_arguments():
        parser = argparse.ArgumentParser(description="Load and summarize experiment data from a JSON file.")
        parser.add_argument('file_path', nargs='?', default="data/data_user=42_exp=1_debug=1.json",
                            help="Path to the JSON file containing experiment data")
        return parser.parse_args()

    def check_file_exists(file_path):
        if not os.path.isfile(file_path):
            print(f"Error: The file '{file_path}' does not exist.")
            sys.exit(1)

    args = parse_arguments()
    check_file_exists(args.file_path)

    load_experiment_data(args.file_path)