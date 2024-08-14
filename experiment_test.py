from functools import partial

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
from nicewebrl.stages import Stage, EnvStage, make_image_html
from nicewebrl.nicejax import JaxWebEnv, base64_npimage

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

def make_train_params(maze_str):
  return mazes.get_maze_reset_params(
      group_set=group_set,
      char2key=char2key,
      maze_str=maze_str,
      label=jnp.array(0),
      make_env_params=True,
  ).replace(training=False)


def housemaze_render_fn(timestep: maze.TimeStep) -> jnp.ndarray:
    image = renderer.create_image_from_grid(
        timestep.state.grid,
        timestep.state.agent_pos,
        timestep.state.agent_dir,
        image_data)
    return image

render_fn = jax.jit(housemaze_render_fn)

action_to_key = {
  int(KeyboardActions.right): "ArrowRight",
  int(KeyboardActions.down): "ArrowDown",
  int(KeyboardActions.left): "ArrowLeft",
  int(KeyboardActions.up) : "ArrowUp",
  int(KeyboardActions.done): "d",
}

action_to_name = {
    int(KeyboardActions.right): "right",
    int(KeyboardActions.down): "down",
    int(KeyboardActions.left): "left",
    int(KeyboardActions.up): "up",
    int(KeyboardActions.done): "done",
}

web_env = JaxWebEnv(jax_env)
# Call this function to pre-compile jax functions before experiment starsts.
dummy_env_params = make_train_params(mazes.maze0)
web_env.precompile(dummy_env_params=dummy_env_params)

def evaluate_success_fn(timestep):
    return int(timestep.reward > .5)


@struct.dataclass
class EnvStageState:
    timestep: maze.TimeStep = None
    nsteps: int = 0
    nepisodes: int = 0
    nsuccesses: int = 0

def stage_display_fn(stage, container):
    with container.style('align-items: center;'):
        container.clear()
        ui.markdown(f"## {stage.name}")
        ui.markdown(f"{stage.body}")


def env_stage_display_fn(stage, container, timestep):
    image = housemaze_render_fn(timestep)
    image = base64_npimage(image)
    category = keys[timestep.state.task_object]

    stage_state = stage.get_user_data('stage_state')
    with container.style('align-items: center;'):
        container.clear()
        ui.markdown(f"## {stage.name}")
        ui.markdown(f"#### {stage.instruction}")
        ui.markdown(f"#### GOAL: {category}")
        text = f"Episodes completed: {stage_state.nepisodes}/{stage.max_episodes}"
        text += f" | # of Successful episodes: {stage_state.nsuccesses}"
        ui.markdown(text)
        ui.html(make_image_html(src=image))

def make_env_stage(maze_name):
    return EnvStage(
        name=f'Training: {maze_name}',
        instruction='Please get the object of interest',
        web_env=web_env,
        action_to_key=action_to_key,
        action_to_name=action_to_name,
        env_params=make_train_params(getattr(mazes, maze_name)),
        render_fn=render_fn,
        display_fn=env_stage_display_fn,
        evaluate_success_fn=evaluate_success_fn,
        state_cls=EnvStageState,
        max_episodes=1,
        min_success=1,
    )
stages = [
    Stage(
        name='Instructions',
        body="""
        You will practice learning how to interact with the environment.
        <br><br>
        You can control the red triangle with the arrow keys on your keyboard.
        <br><br>
        Your goal is to move it to the goal object.
        """,
        display_fn=stage_display_fn,
    ),
    make_env_stage('maze0'),
    make_env_stage('maze1'),
    #make_env_stage('maze2'),
]