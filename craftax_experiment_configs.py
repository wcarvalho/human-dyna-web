from typing import List, Tuple
from flax import struct
import numpy as np
import os.path
import matplotlib.pyplot as plt
from craftax.craftax.constants import BlockType
from craftax.craftax.renderer import render_craftax_pixels as render_partial
from craftax_fullmap_renderer import render_craftax_pixels as render_full
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_IMG, BLOCK_PIXEL_SIZE_HUMAN
import jax


class BlockConfig(struct.PyTreeNode):
  """Configuration for a single experimental block"""

  world_seed: int
  start_train_positions: List[Tuple[int, int]]
  start_eval_positions: List[Tuple[int, int]]
  train_objects: List[int]
  test_objects: List[int]
  start_eval2_positions: List[Tuple[int, int]] = None
  type: str = ""


# Common objects used across configurations
TRAIN_OBJECTS = [BlockType.DIAMOND.value, BlockType.STONE.value]
TEST_OBJECTS = [BlockType.CRAFTING_TABLE.value]

# Paths manipulation configs
PATHS_CONFIGS = [
  BlockConfig(
    world_seed=3,
    start_train_positions=[
      (32, 30),
      (17, 28),
      (29, 35),
      (25, 18),
      (4, 33),
      (22, 29),
      (4, 5),
    ],
    start_eval_positions=[(28, 25)],
    train_objects=TRAIN_OBJECTS,
    test_objects=TEST_OBJECTS,
    type="paths",
  ),
  BlockConfig(
    world_seed=15,
    start_train_positions=[
      (46, 8),
      (45, 32),
      (35, 25),
      (30, 31),
      (32, 30),
      (43, 6),
      (46, 10),
    ],
    start_eval_positions=[(24, 24)],
    train_objects=TRAIN_OBJECTS,
    test_objects=TEST_OBJECTS,
    type="paths",
  ),
  BlockConfig(
    world_seed=20,
    start_train_positions=[
      (25, 22),
      (15, 15),
      (10, 21),
      (12, 20),
      (38, 16),
      (29, 2),
      (12, 10),
    ],
    start_eval_positions=[(18, 30)],
    train_objects=TRAIN_OBJECTS,
    test_objects=TEST_OBJECTS,
    type="paths",
  ),
  BlockConfig(
    world_seed=95,
    start_train_positions=[
      (9, 3),
      (3, 19),
      (2, 27),
      (2, 20),
      (2, 26),
      (15, 21),
      (2, 2),
    ],
    start_eval_positions=[(7, 16)],
    train_objects=TRAIN_OBJECTS,
    test_objects=TEST_OBJECTS,
    type="paths",
  ),
]

# Juncture manipulation configs
JUNCTURE_CONFIGS = [
  BlockConfig(
    world_seed=1,
    start_train_positions=[
      (31, 32),
      (25, 46),
      (24, 46),
      (14, 46),
      (9, 46),
      (11, 46),
      (28, 34),
    ],
    start_eval_positions=[(23, 40)],
    start_eval2_positions=[(19, 23)],
    train_objects=TRAIN_OBJECTS,
    test_objects=TEST_OBJECTS,
    type="juncture",
  ),
  BlockConfig(
    world_seed=2,
    start_train_positions=[
      (23, 18),
      (17, 34),
      (6, 35),
      (2, 41),
      (3, 40),
      (21, 35),
      (15, 16),
    ],
    start_eval_positions=[(15, 30)],
    start_eval2_positions=[(34, 24)],
    train_objects=TRAIN_OBJECTS,
    test_objects=TEST_OBJECTS,
    type="juncture",
  ),
  BlockConfig(
    world_seed=16,
    start_train_positions=[
      (26, 25),
      (25, 33),
      (15, 26),
      (10, 32),
      (12, 31),
      (38, 27),
      (23, 7),
    ],
    start_eval_positions=[(24, 21)],
    start_eval2_positions=[(15, 8)],
    train_objects=TRAIN_OBJECTS,
    test_objects=TEST_OBJECTS,
    type="juncture",
  ),
  BlockConfig(
    world_seed=21,
    start_train_positions=[
      (2, 46),
      (27, 37),
      (9, 40),
      (4, 46),
      (23, 40),
      (25, 46),
      (5, 39),
    ],
    start_eval_positions=[(5, 43)],
    start_eval2_positions=[(12, 32)],
    train_objects=TRAIN_OBJECTS,
    test_objects=TEST_OBJECTS,
    type="juncture",
  ),
]


def get_fullmap_image(world_seed, type="paths"):
  # Use same cache directory as defined in craftax_utils
  subdir = "single" if type == "paths" else "juncture"
  cache_dir = os.path.join("craftax_cache", subdir)
  image_path = os.path.join(cache_dir, f"world_{world_seed}_paths.png")

  if not os.path.exists(image_path):
    raise FileNotFoundError(
      f"No cached map found for world seed {world_seed} at {image_path}"
    )

  # Read image using matplotlib to maintain consistency with how images were saved
  image = plt.imread(image_path)

  # Convert to uint8 if needed
  if image.dtype == np.float32:
    image = (image * 255).astype(np.uint8)

  # Ensure image has exactly 3 channels (RGB)
  if image.ndim == 3 and image.shape[2] == 4:  # RGBA image
    image = image[:, :, :3]  # Keep only RGB channels
  elif image.ndim == 2:  # Grayscale image
    image = np.stack([image] * 3, axis=-1)  # Convert to RGB

  assert image.ndim == 3 and image.shape[2] == 3, "Image must have exactly 3 channels"
  return image


def visualize_block_config(config: BlockConfig, jax_env):
  """Visualizes a block configuration showing the full map and agent views from all starting positions.

  Args:
      config: BlockConfig instance containing world seed and start positions
      jax_env: The Craftax environment instance

  Returns:
      Tuple of (full_map_figure, agent_views_figure)
  """
  import matplotlib.pyplot as plt

  # Calculate number of start positions
  n_train = len(config.start_train_positions)
  n_eval = len(config.start_eval_positions)
  n_eval2 = len(config.start_eval2_positions) if config.start_eval2_positions else 0
  total_positions = n_train + n_eval + n_eval2

  # Create full map figure with two subplots side by side
  fig_map = plt.figure(figsize=(14, 7))

  # Left subplot - rendered environment
  plt.subplot(1, 2, 1)
  env_params = jax_env.default_params.replace(
    world_seeds=(config.world_seed,),
    max_timesteps=100000,
  )
  key = jax.random.PRNGKey(0)
  obs, state = jax_env.reset(key, env_params)
  with jax.disable_jit():
    full_map = render_full(state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN).astype(
      np.uint8
    )
  plt.imshow(full_map)
  plt.title(f"Rendered Environment (World Seed: {config.world_seed})")
  plt.axis("off")

  # Right subplot - cached full map
  plt.subplot(1, 2, 2)
  cached_map = get_fullmap_image(config.world_seed, config.type)
  plt.imshow(cached_map)
  plt.title(f"Cached Full Map (World Seed: {config.world_seed})")
  plt.axis("off")

  # Create agent views figure
  columns = 4
  rows = (total_positions + 2) // columns  # Round up division
  fig_views = plt.figure(figsize=(15, 4 * rows))

  # Function to render from a position
  def render_from_position(pos, idx, title_prefix):
    # Create environment params with this start position
    env_params = jax_env.default_params.replace(
      world_seeds=(config.world_seed,),
      max_timesteps=100000,
      start_position=(pos,),
    )

    # Reset environment
    key = jax.random.PRNGKey(0)
    obs, state = jax_env.reset(key, env_params)

    # Get partial observation using partial renderer
    obs = render_partial(state, block_pixel_size=BLOCK_PIXEL_SIZE_IMG).astype(np.uint8)

    # Plot
    plt.figure(fig_views.number)  # Ensure we're plotting on the views figure
    ax = plt.subplot(rows, columns, idx + 1)
    ax.imshow(obs)
    ax.set_title(f"{title_prefix} Start Pos: {pos}")
    ax.axis("off")

  # Plot train positions
  for idx, pos in enumerate(config.start_train_positions):
    render_from_position(pos, idx, "Train")

  # Plot eval positions
  offset = n_train
  for idx, pos in enumerate(config.start_eval_positions):
    render_from_position(pos, idx + offset, "Eval")

  # Plot eval2 positions if they exist
  if config.start_eval2_positions:
    offset = n_train + n_eval
    for idx, pos in enumerate(config.start_eval2_positions):
      render_from_position(pos, idx + offset, "Eval2")

  plt.figure(fig_views.number)  # Ensure we're adjusting the views figure
  plt.tight_layout()
  return fig_map, fig_views
