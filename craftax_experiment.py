from typing import Callable
from dotenv import load_dotenv
from flax import struct
import jax
import distrax
from jax.experimental import io_callback
import jax.numpy as jnp
import os

from craftax.craftax.renderer import render_craftax_pixels
from craftax.craftax.constants import Action, BLOCK_PIXEL_SIZE_HUMAN, Achievement

from simulations.craftax_web_env import CraftaxSymbolicWebEnvNoAutoReset

from nicegui import ui, app
import nicewebrl
from nicewebrl import JaxWebEnv, base64_npimage, TimestepWrapper
from nicewebrl import Stage, EnvStage
from nicewebrl import get_logger

load_dotenv()

logger = get_logger(__name__)
VERBOSITY = int(os.environ.get('VERBOSITY', 1))
DEBUG = int(os.environ.get('DEBUG', 0))
WORLD_SEED = int(os.environ.get('WORLD_SEED', 2))
MONSTERS = int(os.environ.get('MONSTERS', 1))
NAME = os.environ.get('NAME', 'exp')
DATA_DIR = os.environ.get('DATA_DIR', 'data')

MAX_STAGE_EPISODES = 100
MIN_SUCCESS_EPISODES = 1


def get_user_save_file_fn():
    return f'{DATA_DIR}/user={app.storage.user.get("seed")}_name={NAME}_debug={DEBUG}.json'

########################################
# Utility functions
########################################
EnvParams = struct.PyTreeNode
EvaluateSuccessFn = Callable[[nicewebrl.TimeStep, EnvParams], jnp.bool_]

def get_successes(goals):
   import pdb; pdb.set_trace()
   stage_idx = app.storage.user['stage_idx']
   key = f'{stage_idx}_successes'
   successes = app.storage.user.get(key, {g:0 for g in goals})
   app.storage.user[key] = successes
   return successes

def update_successes(goal, success):
   stage_idx = app.storage.user['stage_idx']
   key = f'{stage_idx}_successes'
   successes = app.storage.user.get(key)
   successes[goal] += success
   app.storage.user[key] = successes

class GoalSettingSuccessTrackingEnvWrapper:
  """
  Wraps an environment to (1) sample new goals and (2) track the number of successful episodes.

  This is in tracker because goals are sampled until enough successful episodes are completed.
  """
  def __init__(
        self,
        env,
        num_success: int = 5, 
        evaluate_success_fn: EvaluateSuccessFn = None,
        ):
    self._env = env
    self.num_success = num_success
    self.evaluate_success_fn = evaluate_success_fn

  # provide proxy access to regular attributes of wrapped object
  def __getattr__(self, name):
      return getattr(self._env, name)

  def reset(self, key, params):
      """Sample goals according to user successes"""
      #########################################
      # Compute how many successful episodes must be completed
      # for each goal
      #########################################
      import pdb; pdb.set_trace()
      # TODO: fix code to use both goal_vector and goals
      successes = io_callback(get_successes, params.goals)
      start = jnp.ones_like(successes)*self.num_success
      successes = jnp.minimum(successes, start)
      remaining = jax.nn.relu(self.num_success - successes)

      #########################################
      # Sample goal in proportion to remaining successes
      # more success = sample less often
      #########################################
      # e.g. [5, 5] --> [0.5, 0.5]
      # e.g. [1, 5] --> [0.2, 0.8]
      goal_probs = remaining/(remaining.sum())
      goal_sampler = distrax.Categorical(probs=goal_probs)
      key, key_ = jax.random.split(key)
      goal_idx = goal_sampler.sample(seed=key_)
      goal = jax.lax.dynamic_index_in_dim(params.goals, goal_idx, keepdims=False)
      params = params.replace(goal=goal)

      return self._env.reset(key, params)

  def step(self, key, timestep, action, params):
      """Check if episode is successful and update user-specific global success count"""
      timestep = self._env.step(key, timestep, action, params)
      success = self.evaluate_success_fn(timestep, params)
      io_callback(update_successes, params.goal, success)
      return timestep

########################################
# Define actions and corresponding keys
########################################
actions = [Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP, Action.DO]
action_array = jnp.array([a.value for a in actions])
action_keys = ["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", " "]
action_to_name = [a.name for a in actions[:len(action_keys)]]

########################################
# Define Craftax environment
########################################
# make environment

static_env_params = CraftaxSymbolicWebEnvNoAutoReset.default_static_params()
static_env_params = static_env_params.replace(
  max_melee_mobs=MONSTERS,
  max_ranged_mobs=MONSTERS,
  max_passive_mobs=10,  # cows
  initial_crafting_tables=True,
  initial_strength=20,
  map_size=(24, 24) if DEBUG else (96, 96),
)
jax_env = CraftaxSymbolicWebEnvNoAutoReset(
   static_env_params=static_env_params,
   )

possible_goals = (
    Achievement.EAT_COW.value,
    Achievement.COLLECT_WOOD.value,
    Achievement.COLLECT_DRINK.value,
    Achievement.MAKE_WOOD_SWORD.value,
)
goal_vector = jnp.ones(len(possible_goals))

default_params = jax_env.default_params.replace(
  day_length=100000,
  mob_despawn_distance=100000,
  world_seeds=(WORLD_SEED,),
  goals=possible_goals,
  goal_vector=goal_vector,
)

# NiceWebRL exploits a `TimeStep` object for checking episode conditions
# wrap environment in wrapper if needed
jax_env = TimestepWrapper(jax_env, autoreset=True)

def evaluate_success_fn(timestep: nicewebrl.TimeStep, params: EnvParams):
  """Episode finishes if person gets goal achievement"""
  import pdb; pdb.set_trace()
  achievements = timestep.state.achievements.astype(jnp.float32)
  goal_achieved = jax.lax.dynamic_index_in_dim(
      achievements, params.goal, keepdims=False)
  return goal_achieved

jax_env = GoalSettingSuccessTrackingEnvWrapper(
    env=jax_env,
    num_success=MIN_SUCCESS_EPISODES,
    evaluate_success_fn=evaluate_success_fn,
)

# create web environment wrapper
jax_web_env = JaxWebEnv(
    env=jax_env,
    actions=action_array)


# Call this function to pre-compile jax functions before experiment starts.
jax_web_env.precompile(dummy_env_params=default_params)

# Define rendering function
def render_fn(timestep: nicewebrl.TimeStep):
    image = render_craftax_pixels(
        timestep.state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
    return image.astype(jnp.uint8)


# precompile vmapped render fn that will vmap over all actions
vmap_render_fn = jax_web_env.precompile_vmap_render_fn(
    render_fn, default_params)

# compile it so fast
render_fn = jax.jit(render_fn).lower(
    jax_web_env.reset(jax.random.PRNGKey(0), default_params)).compile()

########################################
# Define Stages of experiment
########################################
all_stages = []

# ------------------
# Instruction stage
# ------------------
async def instruction_display_fn(stage, container):
    with container.style('align-items: center;'):
        nicewebrl.clear_element(container)
        ui.markdown(f"## {stage.name}")
        ui.markdown("These are instructions")

instruction_stage = Stage(
    name="Instuctions",
    display_fn=instruction_display_fn)
all_stages.append(instruction_stage)


# ------------------
# Environment stage
# ------------------
env_params = default_params


def make_image_html(src):
  html = f'''
  <div id="stateImageContainer" style="display: flex; justify-content: center; align-items: center;">
      <img id="stateImage" src="{src}" style="width: 50%; height: 50%; object-fit: contain;">
  </div>
  '''
  return html


async def env_stage_display_fn(
        stage: EnvStage,
        container: ui.element,
        timestep: nicewebrl.TimeStep):
  state_image = stage.render_fn(timestep)
  state_image = base64_npimage(state_image)
  stage_state = stage.get_user_data('stage_state')

  with container.style('align-items: center;'):
    nicewebrl.clear_element(container)
    # --------------------------------
    # tell person how many episodes completed and how many successful
    # --------------------------------
    with ui.row():
      with ui.element('div').classes('p-2 bg-blue-100'):
        ui.label(
            f"Number of successful episodes: {stage_state.nsuccesses}/{stage.min_success}")
      with ui.element('div').classes('p-2 bg-green-100'):
          ui.label().bind_text_from(
              stage_state, 'nepisodes', lambda n: f"Try: {n}/{stage.max_episodes}")

    # --------------------------------
    # display environment
    # --------------------------------
    ui.html(make_image_html(src=state_image))


def evaluate_success_fn(timestep: nicewebrl.TimeStep):
  """Episode finishes if person every gets 1 achievement"""
  achievements = timestep.state.achievements.astype(jnp.float32)
  success = achievements.sum() >= 100
  return success


def make_env_stage(goals):

  env_params = default_params.replace(
    goals=goals,
  )

  return EnvStage(
      name="Environment",
      web_env=jax_web_env,
      action_keys=action_keys,
      action_to_name=action_to_name,
      env_params=env_params,
      render_fn=render_fn,
      vmap_render_fn=vmap_render_fn,
      display_fn=env_stage_display_fn,
      evaluate_success_fn=evaluate_success_fn,
      min_success=MIN_SUCCESS_EPISODES*len(goals),
      max_episodes=MAX_STAGE_EPISODES,
      verbosity=VERBOSITY,
      user_save_file_fn=get_user_save_file_fn,
      metadata=dict(
          # nothing required, just for bookkeeping
          desc="some description",
          key1="value1",
          key2="value2",
      ),
  )

#def make_block_stage():

#  train_goal_idxs = jax.random.choice(possible_goals, size=2, replace=False)
#  test_goal_idxs = jnp.setdiff1d(possible_goals, train_goal_idxs)
#  world_seed = jax.random.randint(1, 100000)
#  current_env_params = default_params.replace(
#    world_seeds=(world_seed,),
#    )

#  return EnvStage(
#      name="Environment",
#      web_env=jax_web_env,
#      action_keys=action_keys,
#      action_to_name=action_to_name,
#      env_params=current_env_params,
#      render_fn=render_fn,
#      vmap_render_fn=vmap_render_fn,
#      display_fn=env_stage_display_fn,
#      evaluate_success_fn=evaluate_success_fn,
#      min_success=MIN_SUCCESS_EPISODES,
#      max_episodes=MAX_STAGE_EPISODES,
#      verbosity=VERBOSITY,
#      user_save_file_fn=get_user_save_file_fn,
#      metadata=dict(
#          # nothing required, just for bookkeeping
#          desc="some description",
#          key1="value1",
#          key2="value2",
#      ),
#  )

##


all_stages.append(make_env_stage(possible_goals))
