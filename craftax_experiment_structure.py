
import asyncio
from functools import partial
from typing import Callable, List, Tuple, Optional
from dotenv import load_dotenv
from flax import struct
import jax
import distrax
from jax.experimental import io_callback
import jax.numpy as jnp
import numpy as np
import os
from skimage.transform import resize

import matplotlib.pyplot as plt
from nicegui import ui, app
import nicewebrl
from nicewebrl import JaxWebEnv, base64_npimage, TimestepWrapper
from nicewebrl import Stage, EnvStage
from nicewebrl import get_logger
from craftax.craftax.constants import BlockType
from craftax.craftax.renderer import render_craftax_pixels, TEXTURES
from craftax.craftax.constants import (
  Action,
  BLOCK_PIXEL_SIZE_HUMAN,
  BLOCK_PIXEL_SIZE_IMG,
  Achievement,
)
from craftax_experiment_configs import (
  PATHS_CONFIGS,
  JUNCTURE_CONFIGS,
)

load_dotenv()

logger = get_logger(__name__)
VERBOSITY = int(os.environ.get("VERBOSITY", 2))
DEBUG = int(os.environ.get("DEBUG", 1))
DEBUG_DISPLAY = int(os.environ.get("DEBUG_DISPLAY", 1))
MANIPULATION = os.environ.get("MANIPULATION", "paths")
SAY_REUSE = int(os.environ.get("SAY_REUSE", 1))
GIVE_INSTRUCTIONS = int(os.environ.get("GIVE_INSTRUCTIONS", 0))

PRECOMPILE = int(os.environ.get("PRECOMPILE", 1))

DUMMY_ENV = int(os.environ.get("DUMMY_ENV", 1))
MONSTERS = int(os.environ.get("MONSTERS", 1))
NAME = os.environ.get("NAME", "exp")
DATA_DIR = os.environ.get("DATA_DIR", "data")

MAX_STAGE_EPISODES = 100 if DEBUG == 0 else 2
MIN_SUCCESS_TASK = 8 if DEBUG == 0 else 2
MAX_START_POSITIONS = 10


class WorldConfig(struct.PyTreeNode):
  world_seed: int
  start_positions: List[Tuple[int, int]]
  goals: List[int]


if DUMMY_ENV:
  from craftax_dummy_env import CraftaxSymbolicWebEnvNoAutoReset
else:
  from simulations.craftax_web_env import CraftaxSymbolicWebEnvNoAutoReset


def get_user_save_file_fn():
  return (
    f"{DATA_DIR}/user={app.storage.user.get('seed')}_name={NAME}_debug={DEBUG}.json"
  )


########################################
# Utility functions
########################################
EnvParams = struct.PyTreeNode
EvaluateSuccessFn = Callable[[nicewebrl.TimeStep, EnvParams], jnp.bool_]


def get_remaining(possible_goals, num_success):
  possible_goals = [int(i) for i in possible_goals]
  num_success = [int(i) for i in num_success]
  try:
    stage_idx = app.storage.user.get("stage_idx", 0)
  except Exception:
    # no page setup yet
    return jnp.ones(len(possible_goals), dtype=jnp.int32)
  key = f"{stage_idx}_remaining"
  default = {str(g): n for g, n in zip(possible_goals, num_success)}
  remaining = app.storage.user.get(key, default)
  app.storage.user[key] = remaining
  logger.info(f"remaining={remaining}")

  # maintain order of goals
  output = jnp.array(
    [remaining.get(g, n) for g, n in zip(possible_goals, num_success)], dtype=jnp.int32
  )
  return output


def update_successes(current_goal, success):
  try:
    stage_idx = app.storage.user["stage_idx"]
  except Exception:
    # no page setup yet
    return
  key = f"{stage_idx}_remaining"
  remaining = app.storage.user.get(key)

  current_goal = str(int(current_goal))
  remaining[current_goal] -= int(success)
  remaining[current_goal] = max(remaining[current_goal], 0)
  app.storage.user[key] = remaining


class GoalSettingSuccessTrackingEnvWrapper(TimestepWrapper):
  """
  Wraps an environment to (1) sample new goals and (2) track the number of successful episodes.

  This is in tracker because goals are sampled until enough successful episodes are completed.

  NOTE: this intercepts step in TimestepWrapper and replaces its reset with this reset. This is how you get goal-control automatically while you step and automatically reset.
  """

  def __init__(
    self,
    env,
    num_success: int = 5,
    possible_goals: jnp.ndarray = None,
    autoreset: bool = False,
    **kwargs,
  ):
    super().__init__(env, autoreset=autoreset, **kwargs)
    self.num_success = num_success
    self.possible_goals = possible_goals

  # provide proxy access to regular attributes of wrapped object
  def __getattr__(self, name):
    return getattr(self._env, name)

  def reset(self, key, params):
    """Sample goals according to user successes"""
    #########################################
    # Compute how many successful episodes must be completed
    # for each goal
    #########################################
    remaining = io_callback(
      get_remaining,
      jax.ShapeDtypeStruct(shape=(len(self.possible_goals),), dtype=jnp.int32),
      self.possible_goals,
      self.num_success * params.active_goals,
    )

    #########################################
    # Sample goal in proportion to remaining successes
    # more success = sample less often
    #########################################
    # e.g. [5, 0, 5, 0] --> [0.5, 0, 0.5, 0]
    # e.g. [1, 5, 0, 0] --> [0.2, 0.8, 0, 0]
    goal_probs = remaining / (remaining.sum())
    goal_sampler = distrax.Categorical(probs=goal_probs)
    key, key_ = jax.random.split(key)
    goal_idx = goal_sampler.sample(seed=key_)
    goal = jax.lax.dynamic_index_in_dim(params.possible_goals, goal_idx, keepdims=False)
    params = params.replace(current_goal=goal.astype(jnp.int32))
    timestep = super().reset(key, params)
    return timestep


########################################
# Define actions and corresponding keys
########################################
actions = [Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP, Action.DO]
action_array = jnp.array([a.value for a in actions])
action_keys = ["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", " "]
action_to_name = [a.name for a in actions]

########################################
# Define goals
########################################
possible_goals = jnp.array(
  (
    Achievement.COLLECT_DIAMOND.value,
    Achievement.MAKE_WOOD_SWORD.value,
    Achievement.COLLECT_STONE.value,
    Achievement.COLLECT_DRINK.value,
    # Achievement.COLLECT_STONE.value,
  )
)

block_type_2_goal = {
  BlockType.DIAMOND.value: Achievement.COLLECT_DIAMOND.value,
  BlockType.CRAFTING_TABLE.value: Achievement.MAKE_WOOD_SWORD.value,
  BlockType.STONE.value: Achievement.COLLECT_STONE.value,
  BlockType.WATER.value: Achievement.COLLECT_DRINK.value,
}

def goals_from_blocktypes(blocks: List[BlockType], ngoals: int = 4) -> jnp.ndarray:
  """Creates a binary vector indicating which goals are active based on the provided Achievements."""
  goals = jnp.zeros(ngoals, dtype=jnp.float32)
  for block in blocks:
    if block in (BlockType.DIAMOND, BlockType.DIAMOND.value):
      # Collect diamond
      goals = goals.at[0].set(1)
    elif block in (BlockType.CRAFTING_TABLE, BlockType.CRAFTING_TABLE.value):
      # Make wood sword
      goals = goals.at[1].set(1)
    elif block in (BlockType.STONE, BlockType.STONE.value):
      # Collect stone
      goals = goals.at[2].set(1)
    elif block in (BlockType.WATER, BlockType.WATER.value):
      # Collect drink
      goals = goals.at[3].set(1)
    else:
      raise RuntimeError(f"Don't know how to handle blocktype {BlockType(block).name}")
  return goals


# random default
all_goals_active = jnp.ones(len(possible_goals), dtype=jnp.float32)


########################################
# Define Craftax environment
########################################

static_env_params = CraftaxSymbolicWebEnvNoAutoReset.default_static_params()
static_env_params = static_env_params.replace(
  max_melee_mobs=MONSTERS,
  max_ranged_mobs=MONSTERS,
  max_passive_mobs=10,  # cows
  initial_crafting_tables=True,
  initial_strength=20,
  map_size=(48, 48),
)
jax_env = CraftaxSymbolicWebEnvNoAutoReset(
  static_env_params=static_env_params,
)

dummy_start_position = jnp.zeros((MAX_START_POSITIONS, 2), dtype=jnp.int32)
dummy_start_position = dummy_start_position.at[:1].set(jnp.array((24, 24)))

default_params = jax_env.default_params.replace(
  day_length=100000,
  max_timesteps=500 if DEBUG == 0 else 2,
  mob_despawn_distance=100000,
  possible_goals=possible_goals,
  active_goals=all_goals_active,
  world_seeds=(0,),
  start_position=dummy_start_position,
)

jax_env = GoalSettingSuccessTrackingEnvWrapper(
  env=jax_env,
  num_success=MIN_SUCCESS_TASK,
  possible_goals=possible_goals,
  autoreset=False,
)

# create web environment wrapper
jax_web_env = JaxWebEnv(env=jax_env, actions=action_array)


# Define rendering function
if DUMMY_ENV:

  def render_fn(timestep: nicewebrl.TimeStep):
    return timestep.observation.astype(jnp.uint8)

else:

  def render_fn(timestep: nicewebrl.TimeStep):
    return render_craftax_pixels(
      timestep.state, block_pixel_size=BLOCK_PIXEL_SIZE_IMG
    ).astype(jnp.uint8)


# precompile vmapped render fn that will vmap over all actions
vmap_render_fn = None
# pre-compile jax functions before experiment starts.
if PRECOMPILE:
  jax_web_env.precompile(dummy_env_params=default_params)
  vmap_render_fn = jax_web_env.precompile_vmap_render_fn(render_fn, default_params)
  render_fn = (
    jax.jit(render_fn)
    .lower(jax_web_env.reset(jax.random.PRNGKey(0), default_params))
    .compile()
  )


########################################
# Preload images for displaying
########################################
def get_fullmap_image(world_seed):
  # Use same cache directory as defined in craftax_utils
  if DEBUG:
    subdir = "single" if MANIPULATION == "paths" else "juncture"
    cache_dir = os.path.join("craftax_cache", subdir)
    image_path = os.path.join(cache_dir, f"world_{world_seed}_paths.png")
  else:
    cache_dir = os.path.join("craftax_cache", "maps")
    image_path = os.path.join(cache_dir, f"world_{world_seed}.png")

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


if MANIPULATION == "paths":
  FULLMAP_IMAGES = {
    # paths manipulation
    3: get_fullmap_image(3),
    15: get_fullmap_image(15),
    20: get_fullmap_image(20),
    95: get_fullmap_image(95),
  }
elif MANIPULATION == "juncture":
  FULLMAP_IMAGES = {
    # juncture manipulation
    1: get_fullmap_image(1),
    2: get_fullmap_image(2),
    16: get_fullmap_image(16),
    21: get_fullmap_image(21),
  }
else:
  raise RuntimeError


def evaluate_success_fn(timestep: nicewebrl.TimeStep, params: EnvParams):
  success = timestep.reward > 0.5
  update_successes(timestep.state.current_goal, success)
  return success


########################################
# Utils for defining stages of experiment
########################################
def remove_extra_spaces(text):
  """For each line, remove extra space."""
  return "\n".join([i.strip() for i in text.strip().split("\n")])


# ------------------
# Environment stage
# ------------------
def make_image_html(src, id="stateImage", percent=50):
  html = f"""
  <div id="{id}Container" style="display: flex; justify-content: center; align-items: center;">
      <img id="{id}" src="{src}" style="width: {percent}%; height: {percent}%; object-fit: contain;">
  </div>
  """
  return html


def get_goal_image(
  achievement_idx: int, block_pixel_size: int = BLOCK_PIXEL_SIZE_HUMAN
):
  """Get the image for a goal Achievement."""
  textures = TEXTURES[block_pixel_size]

  # Map Achievement to corresponding BlockType/ItemType
  achievement = Achievement(achievement_idx)

  if achievement == Achievement.COLLECT_DIAMOND:
    texture = textures["full_map_block_textures"][BlockType.DIAMOND.value]
    # Extract just one cell (block_pixel_size x block_pixel_size)
    return texture[:block_pixel_size, :block_pixel_size]
  elif achievement == Achievement.COLLECT_IRON:
    texture = textures["full_map_block_textures"][BlockType.IRON.value]
    return texture[:block_pixel_size, :block_pixel_size]
  elif achievement == Achievement.MAKE_WOOD_SWORD:
    # sword_idx = constants.ItemType.WOOD_SWORD.value
    return textures["sword_textures"][Achievement.MAKE_WOOD_SWORD.value]
  elif achievement == Achievement.COLLECT_DRINK:
    texture = textures["full_map_block_textures"][BlockType.WATER.value]
    return texture[:block_pixel_size, :block_pixel_size]
  elif achievement == Achievement.COLLECT_STONE:
    texture = textures["full_map_block_textures"][BlockType.STONE.value]
    return texture[:block_pixel_size, :block_pixel_size]
  else:
    raise ValueError(f"Unknown achievement: {achievement}")


def debug_info(stage):
  stage_state = stage.get_user_data("stage_state")
  debug_info = (
    f"**Manipulation**: {stage.metadata['block_metadata'].get('manipulation')}. "
  )
  if stage_state is not None:
    # ------------
    # stage, world information
    # ------------
    debug_info += f"**Eval**: {stage.metadata['eval']}. "
    # debug_info += f"**World**: {stage.metadata['world_seed']}. "
    # debug_info += f"**Episode** idx: {stage_state.nepisodes}. "
    # debug_info += f"**Step**: {stage_state.nsteps}/{stage.env_params.max_timesteps}. "
    # ui.markdown(debug_info)
    # ------------
    # position information
    # ------------
    start_positions = stage.env_params.start_position
    debug_info += f"**start_positions**: {start_positions}. "
    debug_info += f"**position**: {stage_state.timestep.state.player_position}. "
    ui.markdown(debug_info)
    # ------------
    # remaining goals information
    # ------------
    stage_idx = app.storage.user.get("stage_idx", 0)
    key = f"{stage_idx}_remaining"
    remaining = app.storage.user.get(key, {})
    name = lambda i: Achievement(int(i)).name.replace("_", " ").title()
    remaining = {name(i): r for i, r in remaining.items()}
    debug_info = f"**remaining**: {remaining}. "
    ui.markdown(debug_info)

  return debug_info


# ------------------
# Instruction stage
# ------------------
async def experiment_instructions_display_fn(stage, container):
  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)

    ui.markdown(f"## {stage.name}")
    ui.markdown(f"{remove_extra_spaces(stage.body)}", extras=["cuddled-lists"])
    ui.markdown("Task objects will be selected from the set below:")

    # Get all possible goals and create images for each
    goals = [int(g) for g in possible_goals]
    width = 1.5
    figsize = (len(goals) * width, width)

    with ui.matplotlib(figsize=figsize).figure as fig:
      axs = fig.subplots(1, len(goals))
      for i, goal_idx in enumerate(goals):
        # Get image for this goal
        image = get_goal_image(goal_idx)
        # Plot in matplotlib
        axs[i].imshow(image)
        category = Achievement(goal_idx).name.replace("_", "\n").title()
        axs[i].set_title(f"{category}")
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].axis("off")

      # Adjust layout
      fig.tight_layout()


async def stage_instructions_display_fn(stage, container, new_world=False):
  if new_world:
    ########################################
    # First tell participant that entering a new world
    ########################################
    with container.style("align-items: center;"):
      nicewebrl.clear_element(container)
      ui.markdown("# You are entering a new world.")
      ui.markdown("Please wait 3 seconds before continuing.")
      await asyncio.sleep(3)
      button = ui.button("click to start")
      await nicewebrl.wait_for_button_or_keypress(button, ignore_recent_press=True)

  ########################################
  # Then tell participant the task
  ########################################
  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)

    ui.markdown(f"## {stage.name}")
    if DEBUG:
      debug_info(stage)
    ui.markdown(f"{remove_extra_spaces(stage.body)}", extras=["cuddled-lists"])

    ui.markdown("Task objects will be selected from the set below.")
    if SAY_REUSE:
      ui.markdown("**We note objects relevant to phase 2**")

    # Get all possible goals
    goals = [int(g) for g in possible_goals]

    # Create random display order
    key = nicewebrl.new_rng()
    order = jax.random.permutation(key, jnp.arange(len(goals)))

    test_objects = stage.metadata["block_metadata"]["test_objects"]
    test_objects = [block_type_2_goal[t] for t in test_objects]

    width = 1.5
    figsize = (len(goals) * width, width)
    with ui.matplotlib(figsize=figsize).figure as fig:
      axs = fig.subplots(1, len(goals))
      for i, idx in enumerate(order):
        goal_idx = goals[idx]
        image = get_goal_image(goal_idx)
        category = Achievement(goal_idx).name.replace("_", "\n").title()

        axs[i].imshow(image)
        if SAY_REUSE:
          is_test_object = goal_idx in test_objects
          reward = 1 if is_test_object else 0
          axs[i].set_title(
            f"{category}: {reward}",
            fontsize=10,
            color="green" if is_test_object else "black",
            weight="bold" if is_test_object else "normal",
          )
        else:
          axs[i].set_title(f"{category}")

        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].axis("off")

      fig.tight_layout()

    ui.markdown("Please wait 3 seconds before continuing.")
    await asyncio.sleep(3)


async def env_reset_display_fn(
  stage: EnvStage,
  container: ui.element,
  timestep: nicewebrl.TimeStep,
  pause: int = 0,
):
  goal_object_idx = timestep.state.current_goal
  image = get_goal_image(goal_object_idx)
  image = resize(image, (64, 64, 3), anti_aliasing=True, preserve_range=True).astype(
    np.uint8
  )
  image = base64_npimage(image)

  category = Achievement(goal_object_idx).name.replace("_", " ").title()

  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    ui.markdown(f"#### Goal task: {category}")
    ui.html(make_image_html(src=image))
    button = ui.button("click to start")
    await nicewebrl.wait_for_button_or_keypress(button, ignore_recent_press=True)


async def env_stage_display_fn(
  stage: EnvStage, container: ui.element, timestep: nicewebrl.TimeStep
):
  # Get partial observation image
  partial_obs_image = stage.render_fn(timestep)
  partial_obs_image = base64_npimage(partial_obs_image)

  # Get full map image from metadata
  world_seed = stage.env_params.world_seeds[0]
  full_map_image = FULLMAP_IMAGES[world_seed]
  full_map_image = base64_npimage(full_map_image)

  # Get goal object image
  goal_object_idx = timestep.state.current_goal
  goal_image = get_goal_image(goal_object_idx)
  current_goal_name = Achievement(int(goal_object_idx)).name
  current_goal_name = current_goal_name.replace("_", " ").title()

  stage_state = stage.get_user_data("stage_state")

  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)

    # ui.markdown(f"#### Goal task: {current_goal_name}")
    # Display goal object using matplotlib
    with ui.matplotlib(figsize=(1, 1)).figure as fig:
      ax = fig.subplots(1, 1)
      ax.set_title("Goal")
      ax.imshow(goal_image)
      ax.axis("off")
      fig.tight_layout()

    # Debug info if enabled
    if DEBUG:
      debug_info(stage)

    # Progress tracking row
    with ui.row():
      with ui.element("div").classes("p-2 bg-blue-100"):
        ui.label(
          f"Number of successful episodes: {stage_state.nsuccesses}/{int(stage.min_success)}"
        )
      with ui.element("div").classes("p-2 bg-green-100"):
        ui.label().bind_text_from(
          stage_state, "nepisodes", lambda n: f"Try: {n}/{stage.max_episodes}"
        )
    # Required episodes text
    text = f"You must complete at least {int(stage.min_success)} episodes. You have {stage.max_episodes} tries."
    ui.html(text).style("align-items: center;")
    # ui.html(make_image_html(src=partial_obs_image, id="stateImage"))
    # Side by side images using direct HTML with optimized sizing
    ui.html(f"""
    <div id="stateImageContainer" style="display: flex; width: 100%; gap: 10px; justify-content: center; align-items: center; margin-top: 10px;">
      <div style="flex: 3; max-width: 60%;">
          <div style="text-align: center; margin-bottom: 5px;">Full Map</div>
          <img src="{full_map_image}" id="fullMapImage" style="width: 100%; height: auto; max-height: 60vh; object-fit: contain;">
      </div>
      <div style="flex: 2; max-width: 35%;">
          <div style="text-align: center; margin-bottom: 5px;">Current View</div>
          <img src="{partial_obs_image}" id="stateImage" style="width: 100%; height: auto; max-height: 60vh; object-fit: contain;">
      </div>
    </div>
    """)


def make_env_stage(
  name: str,
  config: WorldConfig,
  metadata: dict,
  min_success: Optional[int] = None,
):
  active_goals = goals_from_blocktypes(config.goals)
  start_position = jnp.zeros((MAX_START_POSITIONS, 2), dtype=jnp.int32)
  start_position = start_position.at[: len(config.start_positions)].set(
    jnp.array(config.start_positions)
  )
  env_params = default_params.replace(
    active_goals=active_goals.astype(jnp.float32),
    world_seeds=(config.world_seed,),
    start_position=start_position,
  )
  min_success = min_success or MIN_SUCCESS_TASK
  return EnvStage(
    name=name,
    web_env=jax_web_env,
    action_keys=action_keys,
    action_to_name=action_to_name,
    env_params=env_params,
    render_fn=render_fn,
    vmap_render_fn=vmap_render_fn,
    reset_display_fn=env_reset_display_fn,
    display_fn=env_stage_display_fn,
    evaluate_success_fn=evaluate_success_fn,
    min_success=min_success * sum(active_goals),
    max_episodes=MAX_STAGE_EPISODES,
    verbosity=VERBOSITY,
    user_save_file_fn=get_user_save_file_fn,
    autoreset_on_done=True,
    msg_display_time=100,
    metadata=metadata,
    precompile=DEBUG == 0,
  )


#########################################################################
# Define stages of experiment
#########################################################################


def train_phase_text():
  phase_1_text = f"""
    Please learn to obtain these objects. You need to succeed {MIN_SUCCESS_TASK} times per object.

    If you retrieve the wrong object, the episode ends early.
    """
  return phase_1_text


if SAY_REUSE:
  instruct_text = """
          This experiment tests how effectively people can learn about goals before direct experience on them.

          It will consist of blocks with two phases each: **one** where you navigate to objects, and **another** where you navigate to other objects that you could have learned about previously.
  """

  def eval_phase_text(time=30):
    threshold = int(time * 2 / 3)
    phase_2_text = f"""
      You will get a <span style="color: green; font-weight: bold;">bonus</span> if you complete the task in less than <span style="color: green; font-weight: bold;">{int(threshold)}</span> seconds. 
      """
    return phase_2_text

else:
  instruct_text = """
    This experiment tests how people learn to navigate maps.

    It will consist of blocks with two phases each: **one** where you navigate to objects, and **another** where you navigate to other objects.
  """

  def eval_phase_text(time=30):
    threshold = int(time * 2 / 3)
    phase_2_text = f"""
      You will get a <span style="color: green; font-weight: bold;">bonus</span> if you complete the task in less than <span style="color: green; font-weight: bold;">{int(threshold)}</span> seconds.
      """
    return phase_2_text


def make_block(
  map: jax.Array,
  train_text: str,
  eval_text: str,
  train_config: WorldConfig,
  eval_config: WorldConfig,
  metadata: dict,
  eval2_config: Optional[WorldConfig] = None,
):
  """
  A block is defined by
  1. training stage instructions
  2. training stage environment
  3. evaluation stage instructions
  4. evaluation stage environment
  5. (optional) second evaluation stage environment
  """
  train_stage_instructions = Stage(
    name="Phase 1 instructions",
    body=train_text,
    display_fn=partial(
      stage_instructions_display_fn, new_world=True),
  )

  train_stage_env = make_env_stage(
    name="Phase 1 training",
    config=train_config,
    metadata=dict(
      world_seed=train_config.world_seed,
      condition=0,
      eval=False,
    ),
  )

  eval_stage_instructions = Stage(
    name="Phase 2 instructions",
    body=eval_text,
    display_fn=stage_instructions_display_fn,
  )

  eval_stage_env = make_env_stage(
    name="Phase 2 eval",
    config=eval_config,
    metadata=dict(
      world_seed=eval_config.world_seed,
      condition=1,
      eval=True,
    ),
  )

  stages = [
    train_stage_instructions,
    train_stage_env,
    eval_stage_instructions,
    eval_stage_env,
  ]
  randomize = []
  if eval2_config is not None:
    # If have 2 evals, randomize them
    randomize = [False, False, False, True, True]
    eval2_stage = make_env_stage(
      name="Phase 2 eval 2",
      config=eval2_config,
      metadata=dict(world_seed=eval2_config.world_seed, condition=2, eval=True),
    )
    stages.append(eval2_stage)

  block = nicewebrl.Block(
    metadata=metadata,
    stages=stages,
    randomize=randomize,
  )
  return block


####################
# practice block
####################
def make_practice_block():
  world_seed = 1
  train_config = WorldConfig(
    world_seed=world_seed,
    start_positions=[(24, 24)],
    goals=[
      BlockType.DIAMOND.value,
      BlockType.CRAFTING_TABLE.value,
      # BlockType.STONE.value
    ],
  )
  eval_config = WorldConfig(
    world_seed=world_seed,
    start_positions=[(24, 24)],
    goals=[BlockType.STONE.value],
  )
  return make_block(
    map=FULLMAP_IMAGES[world_seed],
    train_config=train_config,
    eval_config=eval_config,
    train_text=train_phase_text(),
    eval_text=eval_phase_text(10),
    metadata=dict(
      manipulation="practice",
      world_seed=world_seed,
      desc="practice",
      long="practice",
    ),
  )


def make_manipulation_block(
  world_seed: int,
  train_text: str,
  eval_text: str,
  start_train_positions: List[Tuple[int, int]],
  start_eval_positions: List[Tuple[int, int]],
  train_objects: List[int],
  test_objects: List[int],
  manipulation: str,
  desc: str,
  long: str,
  start_eval2_positions: Optional[List[Tuple[int, int]]] = None,
):
  """Creates a manipulation block for either paths or juncture experiments."""
  train_config = WorldConfig(
    world_seed=world_seed,
    start_positions=start_train_positions + start_eval_positions,
    goals=train_objects,
  )

  eval_config = WorldConfig(
    world_seed=world_seed,
    start_positions=start_eval_positions,
    goals=test_objects,
  )

  eval2_config = None
  if start_eval2_positions is not None:
    eval2_config = WorldConfig(
      world_seed=world_seed,
      start_positions=start_eval2_positions,
      goals=test_objects,
    )

  metadata = dict(
    manipulation=manipulation,
    world_seed=world_seed,
    train_objects=train_objects,
    test_objects=test_objects,
    desc=desc,
    long=long,
  )

  return make_block(
    map=FULLMAP_IMAGES[world_seed],
    train_config=train_config,
    eval_config=eval_config,
    eval2_config=eval2_config,
    train_text=train_text,
    eval_text=eval_text,
    metadata=metadata,
  )


# Create experiment blocks with descriptions inline
if MANIPULATION == "paths":
  experiment_blocks = [
    make_manipulation_block(
      world_seed=config.world_seed,
      start_train_positions=config.start_train_positions,
      start_eval_positions=config.start_eval_positions,
      train_objects=config.train_objects,
      test_objects=config.test_objects,
      train_text=train_phase_text(),
      eval_text=eval_phase_text(),
      manipulation=MANIPULATION,
      desc="reusing longer of two paths which matches training path",
      long="Here there are two paths to the test object. We predict that people will take the path that was used to get to the training object.",
    )
    for config in PATHS_CONFIGS
  ]
elif MANIPULATION == "juncture":
  experiment_blocks = [
    make_manipulation_block(
      world_seed=config.world_seed,
      start_train_positions=config.start_train_positions,
      start_eval_positions=config.start_eval_positions,
      start_eval2_position=config.start_eval2_positions,
      train_objects=config.train_objects,
      test_objects=config.test_objects,
      train_text=train_phase_text(),
      eval_text=eval_phase_text(10),
      manipulation=MANIPULATION,
      desc="probe behavior at juncture",
      long="Here there is a juncture to the test object. We predict that people will be faster at a juncture than at another point on the map.",
      start_eval2_positions=config.start_eval2_positions,
    )
    for config in JUNCTURE_CONFIGS
  ]

instruct_block = nicewebrl.Block(
  [
    Stage(
      name="Experiment instructions",
      body=instruct_text,
      display_fn=experiment_instructions_display_fn,
    ),
  ],
  metadata=dict(desc="instructions", long="instructions"),
)

all_blocks = []
if GIVE_INSTRUCTIONS:
  all_blocks.extend([instruct_block, make_practice_block()])

all_blocks.extend(experiment_blocks)
all_stages = nicewebrl.prepare_blocks(all_blocks)

##########################
# generating stage order
##########################


def generate_block_stage_order(rng_key):
  """Take blocks defined above, flatten all their stages, and generate an order where the (1) blocks are randomized, and (2) stages within blocks are randomized if they're consecutive eval stages."""
  fixed_blocks = []
  offset = 0
  if GIVE_INSTRUCTIONS:
    offset = 2
  # fix ordering of instruct_block, practice_block
  fixed_blocks.extend(list(range(offset)))
  fixed_blocks = jnp.array(fixed_blocks)

  # blocks afterward are randomized
  randomized_blocks = list(all_blocks[offset:])
  random_order = jax.random.permutation(rng_key, len(randomized_blocks)) + offset

  block_order = jnp.concatenate(
    [
      fixed_blocks,  # instruction blocks
      random_order,  # experiment blocks
    ]
  ).astype(jnp.int32)
  block_order = block_order.tolist()
  stage_order = nicewebrl.generate_stage_order(all_blocks, block_order, rng_key)
  stage_order = [int(i) for i in stage_order]
  return block_order, stage_order
