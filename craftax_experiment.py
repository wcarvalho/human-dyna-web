from dotenv import load_dotenv
from flax import struct
import jax
import jax.numpy as jnp
import os

from craftax.craftax.renderer import render_craftax_pixels
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.constants import Action, BLOCK_PIXEL_SIZE_HUMAN
from craftax.craftax import craftax_state


from simulations.craftax_env import CraftaxSymbolicEnvNoAutoReset
from nicegui import ui, app
import nicewebrl
from nicewebrl import JaxWebEnv, base64_npimage, TimestepWrapper
from nicewebrl import Stage, EnvStage, EnvStageState
from nicewebrl import get_logger

load_dotenv()

logger = get_logger(__name__)
VERBOSITY = int(os.environ.get('VERBOSITY', 0))
DEBUG = int(os.environ.get('DEBUG', 0))
WORLD_SEED = int(os.environ.get('WORLD_SEED', 1))
NAME = os.environ.get('NAME', 'exp')
DATA_DIR = os.environ.get('DATA_DIR', 'data')

MAX_STAGE_EPISODES = 1
MAX_EPISODE_TIMESTEPS = 10
MIN_SUCCESS_EPISODES = 1


def get_user_save_file_fn():
    return f'{DATA_DIR}/user={app.storage.user.get("seed")}_name={NAME}_debug={DEBUG}.json'


########################################
# Define actions and corresponding keys
########################################
actions = [Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP, Action.DO]
action_array = jnp.array([a.value for a in actions])
action_keys = ["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", " "]
action_to_name = [a.name for a in actions]

########################################
# Define Craftax environment
########################################
# make environment
jax_env = CraftaxSymbolicEnvNoAutoReset()
default_params = jax_env.default_params.replace(
   world_seeds=(WORLD_SEED,))

# NiceWebRL exploits a `TimeStep` object for checking episode conditions
# wrap environment in wrapper if needed
jax_env = TimestepWrapper(jax_env, autoreset=True)

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
    success = achievements.sum() >= 1
    return success


environment_stage = EnvStage(
    name="Environment",
    web_env=jax_web_env,
    action_keys=action_keys,
    action_to_name=action_to_name,
    env_params=env_params,
    render_fn=render_fn,
    vmap_render_fn=vmap_render_fn,
    display_fn=env_stage_display_fn,
    evaluate_success_fn=evaluate_success_fn,
    min_success=MIN_SUCCESS_EPISODES,
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
all_stages.append(environment_stage)
