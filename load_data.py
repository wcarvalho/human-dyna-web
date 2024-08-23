from collections import defaultdict
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from nicewebrl import nicejax
from housemaze.human_dyna import env as maze
import jax
import numpy as np

def compute_reaction_time(image_seen_time: str, action_taken_time: str) -> float:
    # Convert string timestamps to datetime objects
    image_seen_datetime = datetime.strptime(image_seen_time, '%Y-%m-%dT%H:%M:%S.%fZ')
    action_taken_datetime = datetime.strptime(action_taken_time, '%Y-%m-%dT%H:%M:%S.%fZ')

    # Calculate the time difference
    time_difference = action_taken_datetime - image_seen_datetime

    # Convert the time difference to milliseconds
    reaction_time_milliseconds = time_difference.total_seconds() * 1000

    return reaction_time_milliseconds

def load_experiment_data(file_path: str):
    with open(file_path, 'r') as f:
        data_dicts = json.load(f)

    data_dicts = [row for row in data_dicts if not 'practice' in row['metadata']['block_metadata']['desc']]
    datum = data_dicts[0]

    action_taken = datum['action_idx']
    image_seen_time = datum['image_seen_time']
    action_taken_time = datum['action_taken_time']
    reaction_time = compute_reaction_time(image_seen_time, action_taken_time)

    print(f"Reaction time: {reaction_time:.3f} seconds")


    # as long as you give it the right class, it will deserialize it correctly
    timestep = nicejax.deserialize_bytes(cls=maze.TimeStep, encoded_data=datum['data'])

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