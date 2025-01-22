from typing import Union, Tuple
import jax
import jax.numpy as jnp
from collections import deque
from craftax.craftax.constants import Action, BlockType
from craftax_fullmap_renderer import render_craftax_pixels
import craftax_fullmap_constants as constants
import matplotlib.pyplot as plt
import os
import numpy as np

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm
CACHE_DIR = 'craftax_cache'

def bfs(state, goal: Union[int, Tuple[int, int]], key = None, walkable_blocks = None, budget=1e8):
  """Performs Breadth-First Search to find a path from the player's position to a goal.

  Args:
      state: Game state object containing the map and player position.
      goal: Either a tuple of (row, col) coordinates or a BlockType value to search for.
      key: Optional JAX PRNG key for random direction shuffling. Defaults to key(42).
      walkable_blocks: List of block types that can be traversed. Defaults to [GRASS, STONE, PATH].
      budget: Maximum number of iterations before giving up. Defaults to 1e8.

  Returns:
      Tuple of:
          - JAX array of coordinates forming the path to the goal, or None if no path found
          - Number of iterations performed during search
  """
  map = state.map[state.player_level]  # Current level's map
  agent_pos = state.player_position

  if key is None:
    key = jax.random.PRNGKey(42)
  if walkable_blocks is None:
    walkable_blocks = [
      BlockType.GRASS.value,
      BlockType.STONE.value,
      BlockType.PATH.value,
    ]

  rows, cols = map.shape
  queue = deque([(agent_pos, [agent_pos])])
  visited = set()
  iterations = 0

  # Handle different goal types
  is_position_goal = isinstance(goal, (tuple, jnp.ndarray)) and len(goal) == 2
  if is_position_goal:
    goal_object_type = map[goal[0], goal[1]]
    walkable_blocks.append(goal_object_type)

  passible = jnp.array(walkable_blocks)

  # Create progress bar without total to show raw counts
  pbar = tqdm(desc="BFS Iterations")
  
  while queue:
    key, subkey = jax.random.split(key)
    iterations += 1
    pbar.update(1)  # Update progress bar
    
    if iterations >= budget:
      pbar.close()  # Close progress bar before returning
      return [], iterations

    current_pos, path = queue.popleft()
    # Check goal condition based on goal type
    if is_position_goal:
      if tuple(current_pos) == tuple(goal):  # Convert both to tuples for comparison
        pbar.close()  # Close progress bar before returning
        return jnp.array([p for p in path]), iterations
    else:  # integer (BlockType.value)
      if map[current_pos[0], current_pos[1]] == goal:
        pbar.close()  # Close progress bar before returning
        return jnp.array([p for p in path]), iterations
    visited.add(tuple([int(i) for i in current_pos]))

    # Shuffle the order of directions
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    key, subkey = jax.random.split(key)
    directions = jax.random.permutation(subkey, jnp.array(directions))

    for dx, dy in directions:
      new_x, new_y = int(current_pos[0] + dx), int(current_pos[1] + dy)
      if (
        0 <= new_x < rows
        and 0 <= new_y < cols
        and (new_x, new_y) not in visited
        and (map[new_x, new_y] == passible).any()
      ):
        new_path = path + [(new_x, new_y)]
        iterations += 1
        queue.append(((new_x, new_y), new_path))

  pbar.close()  # Close progress bar before returning
  return [], iterations


def manhattan_distance(pos1, pos2):
  """Calculate Manhattan distance between two positions."""
  return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def astar(state, goal: Union[int, Tuple[int, int]], key=None, walkable_blocks=None, budget=1e8):
  """Performs A* Search to find a path from the player's position to a goal.
  
  Args: [same as before]
  """
  map = state.map[state.player_level]
  agent_pos = tuple(int(i) for i in state.player_position)

  if key is None:
    key = jax.random.PRNGKey(42)
  if walkable_blocks is None:
    walkable_blocks = [
      BlockType.GRASS.value,
      BlockType.STONE.value,
      BlockType.PATH.value,
    ]

  rows, cols = map.shape
  
  # Handle different goal types
  is_position_goal = isinstance(goal, (tuple, jnp.ndarray)) and len(goal) == 2
  if is_position_goal:
    goal_object_type = map[goal[0], goal[1]]
    walkable_blocks.append(goal_object_type)
    goal_pos = goal
  else:
    # For BlockType goals, find the closest matching block
    matches = jnp.where(map == goal)
    if len(matches[0]) == 0:
      return [], 0
    # Use Manhattan distance to find closest goal
    distances = [manhattan_distance(agent_pos, (y, x)) for y, x in zip(matches[0], matches[1])]
    closest_idx = jnp.argmin(jnp.array(distances))
    goal_pos = (int(matches[0][closest_idx]), int(matches[1][closest_idx]))

  passible = jnp.array(walkable_blocks)

  # Priority queue implemented as a list of (priority, count, pos, path) tuples
  import heapq
  count = 0  # Tiebreaker for equal priorities
  open_set = [(0, count, agent_pos, [agent_pos])]
  closed_set = set()
  g_scores = {tuple(agent_pos): 0}  # Cost from start to node
  iterations = 0

  pbar = tqdm(desc="A* Iterations")

  while open_set:
    iterations += 1
    pbar.update(1)

    if iterations >= budget:
      pbar.close()
      return [], iterations

    # Get node with lowest f_score
    current = heapq.heappop(open_set)
    current_pos = current[2]
    current_path = current[3]

    if tuple(current_pos) == tuple(goal_pos):
      pbar.close()
      return jnp.array(current_path), iterations

    if tuple(current_pos) in closed_set:
      continue

    closed_set.add(tuple(current_pos))

    # Shuffle directions for randomness
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    key, subkey = jax.random.split(key)
    directions = jax.random.permutation(subkey, jnp.array(directions))

    for dx, dy in directions:
      new_x, new_y = int(current_pos[0] + dx), int(current_pos[1] + dy)
      new_pos = (new_x, new_y)
      
      if (
        0 <= new_x < rows
        and 0 <= new_y < cols
        and new_pos not in closed_set
        and (map[new_x, new_y] == passible).any()
      ):
        # g_score is distance from start
        tentative_g = g_scores[tuple(current_pos)] + 1

        if new_pos not in g_scores or tentative_g < g_scores[new_pos]:
          # This path is better than previous ones
          g_scores[new_pos] = tentative_g
          f_score = tentative_g + manhattan_distance(new_pos, goal_pos)
          count += 1
          heapq.heappush(open_set, (f_score, count, new_pos, current_path + [new_pos]))

  pbar.close()
  return [], iterations


def actions_from_path(path):
  if path is None or len(path) < 2:
    return jnp.array([Action.NOOP.value])

  actions = []
  for i in range(1, len(path)):
    prev_pos = path[i - 1]
    curr_pos = path[i]

    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]

    if dx == 1:
      actions.append(Action.DOWN.value)
    elif dx == -1:
      actions.append(Action.UP.value)
    elif dy == 1:
      actions.append(Action.RIGHT.value)
    elif dy == -1:
      actions.append(Action.LEFT.value)

  actions.append(Action.NOOP.value)
  return jnp.array(actions)

def place_arrows_on_image(
  image,
  positions,
  actions,
  maze_height,
  maze_width,
  arrow_scale=5,
  arrow_color="b",
  ax=None,
  display_image=True,
  ):
  # Get the dimensions of the image and the maze
  image_height, image_width, _ = image.shape

  # Calculate the scaling factors for mapping maze coordinates to image coordinates
  scale_y = image_height / maze_height
  scale_x = image_width / maze_width

  # No need for wall offsets anymore
  offset_y = 0
  offset_x = 0

  # Create a figure and axis
  if ax is None:
    fig, ax = plt.subplots(1, figsize=(5, 5))

  # Display the rendered image
  if display_image:
    ax.imshow(image)

  # Iterate over each position and action
  for (y, x), action in zip(positions, actions):
    # Calculate the center coordinates of the cell in the image
    center_y = offset_y + (y + 0.5) * scale_y
    center_x = offset_x + (x + 0.5) * scale_x

    # Define the arrow directions based on the action
    if action == Action.UP.value:
      dx, dy = 0, -scale_y / 2
    elif action == Action.DOWN.value:
      dx, dy = 0, scale_y / 2
    elif action == Action.LEFT.value:
      dx, dy = -scale_x / 2, 0
    elif action == Action.RIGHT.value:
      dx, dy = scale_x / 2, 0
    else:  # KeyboardActions.done
      continue  # Skip drawing an arrow for the 'done' action

    # Draw the arrow on the image with specified color
    ax.arrow(
      center_x,
      center_y,
      dx,
      dy,
      head_width=scale_x / (arrow_scale * 0.7),  # Increased head width by ~40%
      head_length=scale_y / (arrow_scale * 0.7),  # Increased head length by ~40%
      width=scale_x / (arrow_scale * 2),
      fc=arrow_color,
      ec=arrow_color,
    )

  # Remove the axis ticks and labels

  ax.set_xticks([])
  ax.set_yticks([])
  return ax

def get_object_positions(state, block_type):
  map = state.map
  # Find all positions where the block type matches
  matches = jnp.where(map[0] == block_type.value)
  # Stack the y and x coordinates into a single array of shape (N, 2)
  # where N is the number of matching positions
  positions = jnp.stack([matches[0], matches[1]], axis=1)
  return positions


def render_fn(state, block_pixel_size=constants.BLOCK_PIXEL_SIZE_IMG):
  image = render_craftax_pixels(
    state, block_pixel_size=block_pixel_size
  )
  return image.astype(jnp.uint8)

def get_cached_path(start_pos, goal_pos):
    """Load cached path if it exists."""
    cache_file = os.path.join(CACHE_DIR, f"path_{start_pos}_{goal_pos}.npy")
    if os.path.exists(cache_file):
        try:
            print(f"Loading path from cache: {cache_file}")
            return np.load(cache_file, allow_pickle=True)
        except Exception as e:
            return None
    return None

def save_path_to_cache(path, start_pos, goal_pos):
    """Save path to cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"path_{start_pos}_{goal_pos}.npy")
    print(f"Saving path to cache: {cache_file}")
    np.save(cache_file, path)

def display_map_with_paths(
  state,
  goal,
  params,
  block_pixel_size=constants.BLOCK_PIXEL_SIZE_IMG,
  display_paths=True,
  ):

  world = int(params.world_seeds[0])
  fig, ax = plt.subplots(1, figsize=(10, 10))

  image = render_fn(state, block_pixel_size=block_pixel_size)
  ax.imshow(image)

  ax.axis("off")  # This removes the axes and grid
  if not display_paths:
    ax.set_title(f"World {world}")
    return ax

  goal_positions = get_object_positions(state, goal)

  paths = []
  start_pos = tuple(state.player_position)
  for goal_position in goal_positions:
    goal_pos = tuple(goal_position)
    # Try to load from cache first
    path = get_cached_path(start_pos, goal_pos)
    
    if path is None:
      path, _ = astar(state, goal_position)
      save_path_to_cache(path, start_pos, goal_pos)

    if path is not None and len(path) > 0:
      # Check if path exists before getting actions
      actions = actions_from_path(path)
      paths.append((path, actions))

  # sort paths by length
  paths.sort(key=lambda x: len(x[0]))

  colors = [
    "#FFB700", # google orange
    "#D55E00", # vermillion
    "#CC79A7", # reddish purple
    "#9B80E6", # nice purple
    "#679FE5", # pretty blue
    "#186CED", # google blue
    (86 / 255, 180 / 255, 233 / 255), # sky blue
  ]
  path_lengths = [len(path) for path, _ in paths]
  ax.set_title(f"World {world}\nPath lengths: {path_lengths}")
  for idx, (path, actions) in enumerate(paths):
    place_arrows_on_image(
      image,
      path,
      actions,
      state.map.shape[1],
      state.map.shape[2],
      ax=ax,
      display_image=False,
      arrow_color=colors[idx % len(colors)],
    )
  return ax
