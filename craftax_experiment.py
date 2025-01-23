from typing import Callable
from dotenv import load_dotenv
from flax import struct
import jax
import distrax
from jax.experimental import io_callback
import jax.numpy as jnp
import os

from nicegui import ui, app
import nicewebrl
from nicewebrl import JaxWebEnv, base64_npimage, TimestepWrapper
from nicewebrl import Stage, EnvStage
from nicewebrl import get_logger

load_dotenv()

logger = get_logger(__name__)
VERBOSITY = int(os.environ.get("VERBOSITY", 1))
DEBUG = int(os.environ.get("DEBUG", 1))
WORLD_SEED = int(os.environ.get("WORLD_SEED", 3))
DUMMY_ENV = int(os.environ.get("DUMMY_ENV", 1))
MONSTERS = int(os.environ.get("MONSTERS", 1))
NAME = os.environ.get("NAME", "exp")
DATA_DIR = os.environ.get("DATA_DIR", "data")

MAX_STAGE_EPISODES = 100
MIN_SUCCESS_EPISODES = 100


if DUMMY_ENV:
  from craftax_dummy_env import CraftaxSymbolicWebEnvNoAutoReset, Action, Achievement

  BLOCK_PIXEL_SIZE_HUMAN = 64
else:
  from craftax.craftax.renderer import render_craftax_pixels
  from craftax_fullmap_renderer import (
    render_craftax_pixels as full_obs_render_craftax_pixels,
  )
  from simulations.craftax_web_env import CraftaxSymbolicWebEnvNoAutoReset
  from craftax.craftax.constants import Action, BLOCK_PIXEL_SIZE_HUMAN, Achievement


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
  default = {g: n for g, n in zip(possible_goals, num_success)}
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
    autoreset: bool = True,
  ):
    super().__init__(env, autoreset=autoreset)
    self.num_success = num_success

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
      jax.ShapeDtypeStruct(shape=(len(params.possible_goals),), dtype=jnp.int32),
      params.possible_goals,
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


class CraftaxEnvStage(EnvStage):
  """This env stage optionally displays both the fully observable and partially observable images."""

  async def change_world(self, container: ui.element, number: int = None):
    self.env_params = self.env_params.replace(world_seeds=(number,))
    rng = nicewebrl.new_rng()
    timestep = self.web_env.reset(rng, self.env_params)
    stage_state = self.get_user_data("stage_state")
    stage_state = stage_state.replace(timestep=timestep)
    await self.set_user_data(stage_state=stage_state)
    if self.reset_display_fn is not None:
      await self.reset_display_fn(stage=self, container=container, timestep=timestep)
    await self.step_and_send_timestep(container, timestep)


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
    Achievement.MAKE_WOOD_SWORD.value,
    Achievement.EAT_COW.value,
    Achievement.COLLECT_WOOD.value,
    Achievement.COLLECT_DRINK.value,
    # Achievement.COLLECT_STONE.value,
  )
)

all_goals_active = jnp.ones(len(possible_goals), dtype=jnp.float32)
train_goals_active = jnp.array((1, 1, 0, 0), dtype=jnp.float32)
test_goals_active = jnp.array((0, 0, 1, 1), dtype=jnp.float32)

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


default_params = jax_env.default_params.replace(
  day_length=100000,
  mob_despawn_distance=100000,
  world_seeds=(WORLD_SEED,),
  possible_goals=possible_goals,
  active_goals=all_goals_active,
)

jax_env = GoalSettingSuccessTrackingEnvWrapper(
  env=jax_env,
  num_success=MIN_SUCCESS_EPISODES,
  # evaluate_success_fn=evaluate_success_fn,
  autoreset=False,
)

# create web environment wrapper
jax_web_env = JaxWebEnv(env=jax_env, actions=action_array)


# Call this function to pre-compile jax functions before experiment starts.
jax_web_env.precompile(dummy_env_params=default_params)


# Define rendering function
if DUMMY_ENV:

  def render_fn(timestep: nicewebrl.TimeStep):
    return timestep.observation.astype(jnp.uint8)

  def fullmap_render_fn(timestep: nicewebrl.TimeStep):
    return timestep.observation.astype(jnp.uint8)
else:

  def render_fn(timestep: nicewebrl.TimeStep):
    return render_craftax_pixels(
      timestep.state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN
    ).astype(jnp.uint8)

  def fullmap_render_fn(timestep: nicewebrl.TimeStep):
    return full_obs_render_craftax_pixels(
      timestep.state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN
    ).astype(jnp.uint8)


# precompile vmapped render fn that will vmap over all actions
vmap_render_fn = jax_web_env.precompile_vmap_render_fn(render_fn, default_params)

# compile it so fast
render_fn = (
  jax.jit(render_fn)
  .lower(jax_web_env.reset(jax.random.PRNGKey(0), default_params))
  .compile()
)


########################################
# Define Stages of experiment
########################################
all_stages = []


# ------------------
# Instruction stage
# ------------------
async def instruction_display_fn(stage, container):
  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    ui.markdown(f"## {stage.name}")
    ui.markdown("These are instructions")


instruction_stage = Stage(name="Instuctions", display_fn=instruction_display_fn)


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


async def reset_display_fn(
  stage: EnvStage,
  container: ui.element,
  timestep: nicewebrl.TimeStep,
):
  del stage
  with jax.disable_jit():
    full_obs_image = fullmap_render_fn(timestep)
  full_obs_image = base64_npimage(full_obs_image)

  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    ui.html(make_image_html(src=full_obs_image, id="stateImage", percent=100))
    button = ui.button("click to start")
    await nicewebrl.wait_for_button_or_keypress(button, ignore_recent_press=True)


async def env_stage_display_fn(
  stage: EnvStage, container: ui.element, timestep: nicewebrl.TimeStep
):
  partial_obs_image = stage.render_fn(timestep)
  print("partial_obs_image", partial_obs_image.max())
  partial_obs_image = base64_npimage(partial_obs_image)
  stage_state = stage.get_user_data("stage_state")
  current_goal = timestep.state.current_goal
  current_goal_name = Achievement(int(current_goal)).name
  current_goal_name = current_goal_name.replace("_", " ").title()

  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    # --------------------------------
    # tell person how many episodes completed and how many successful
    # --------------------------------
    with ui.row():
      with ui.element("div").classes("p-2 bg-blue-100"):
        ui.label(
          f"Number of successful episodes: {stage_state.nsuccesses}/{int(stage.min_success)}"
        )
      with ui.element("div").classes("p-2 bg-green-100"):
        ui.label().bind_text_from(
          stage_state, "nepisodes", lambda n: f"Try: {n}/{stage.max_episodes}"
        )

    if DEBUG:
      ui.label("World seed: " + str(stage.env_params.world_seeds[0]))
      stage_idx = app.storage.user["stage_idx"]

      def print_remaining(rem):
        return {Achievement(int(g)).name: r for g, r in rem.items()}

      key = f"{stage_idx}_remaining"
      ui.label().bind_text_from(app.storage.user, key, print_remaining)
    ui.label(f"Goal: {current_goal_name}")
    ui.html(make_image_html(src=partial_obs_image, id="stateImage", percent=50))


def evaluate_success_fn(timestep: nicewebrl.TimeStep, params: EnvParams):
  success = timestep.reward > 0.5
  update_successes(params.current_goal, success)
  return success


def make_env_stage(active_goals):
  env_params = default_params.replace(
    active_goals=active_goals.astype(jnp.float32),
  )

  return CraftaxEnvStage(
    name="Environment",
    web_env=jax_web_env,
    action_keys=action_keys,
    action_to_name=action_to_name,
    env_params=env_params,
    render_fn=render_fn,
    vmap_render_fn=vmap_render_fn,
    reset_display_fn=reset_display_fn,
    display_fn=env_stage_display_fn,
    evaluate_success_fn=evaluate_success_fn,
    min_success=MIN_SUCCESS_EPISODES * sum(active_goals),
    max_episodes=MAX_STAGE_EPISODES,
    verbosity=VERBOSITY,
    user_save_file_fn=get_user_save_file_fn,
    autoreset_on_done=True,
    msg_display_time=100,
    metadata=dict(
      # nothing required, just for bookkeeping
      desc="some description",
      key1="value1",
      key2="value2",
    ),
  )


# all_stages.append(instruction_stage)
all_stages.append(make_env_stage(all_goals_active))
