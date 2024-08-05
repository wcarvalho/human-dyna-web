from housemaze import renderer
from housemaze.maze import KeyboardActions
from housemaze.human_dyna import env
from housemaze.human_dyna import utils
from housemaze.human_dyna import env as maze
from housemaze.human_dyna import mazes
from housemaze.human_dyna import experiments

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from nicegui import app, ui
from nicewebrl.stages import Stage, EnvStage
from nicewebrl.nicejax import JaxWebEnv

char2key, group_set, task_objects = mazes.get_group_set(3)
image_data = utils.load_image_dict()

task_runner = env.TaskRunner(task_objects=task_objects)
keys = image_data['keys']

jax_env = env.HouseMaze(
    task_runner=task_runner,
    num_categories=len(keys),
    use_done=True,
)
jax_env = utils.AutoResetWrapper(jax_env)

maze3_train_params = mazes.get_maze_reset_params(
    group_set=group_set,
    char2key=char2key,
    maze_str=mazes.maze3,
    label=jnp.array(0),
    make_env_params=True,
)


def housemaze_render_fn(timestep: maze.TimeStep) -> jnp.ndarray:
    image = renderer.create_image_from_grid(
        timestep.state.grid,
        timestep.state.agent_pos,
        timestep.state.agent_dir,
        image_data)
    return image

def task_desc_fn(timestep):
  category = keys[timestep.state.task_object]
  return ui.markdown(f"### GOAL: {category}")

action_to_key = {
  int(KeyboardActions.right): "ArrowRight",
  int(KeyboardActions.down): "ArrowDown",
  int(KeyboardActions.left): "ArrowLeft",
  int(KeyboardActions.up) : "ArrowUp",
  int(KeyboardActions.done): "d",
}

web_env = JaxWebEnv(jax_env)

def evaluate_success(timestep):
    return int(timestep.reward > .5)

@struct.dataclass
class StageState:
    finished: bool = False


@struct.dataclass
class EnvStageState(StageState):
    timestep: maze.TimeStep = None
    nsteps: int = 0
    nepisodes: int = 0
    nsuccesses: int = 0

stages = [
  EnvStage(
    name='Training',
    instruction='Please get the object of interest',
    web_env=web_env,
    action_to_key=action_to_key,
    env_params=maze3_train_params.replace(training=False),
    render_fn=jax.jit(housemaze_render_fn),
    multi_render_fn=jax.jit(jax.vmap(housemaze_render_fn)),
    task_desc_fn=task_desc_fn,
    evaluate_success_fn=evaluate_success,
    state_cls=EnvStageState,
  )
]