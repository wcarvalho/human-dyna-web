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
from flax import struct
from dotenv import load_dotenv
import os
load_dotenv()

DEBUG = int(os.environ.get('DEBUG', 0))

from nicegui import ui
from nicewebrl.stages import Stage, EnvStage, make_image_html
from nicewebrl.nicejax import JaxWebEnv, base64_npimage

num_rooms = 3
num_pairs_for_exp = 1
char2key, group_set, task_objects = mazes.get_group_set(num_rooms)
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
      group_set=group_set[:num_pairs_for_exp],
      char2key=char2key,
      maze_str=maze_str,
      label=jnp.array(0),
      make_env_params=True,
  ).replace(
        terminate_with_done=jnp.array(2),
  )


def housemaze_render_fn(timestep: maze.TimeStep) -> jnp.ndarray:
    image = renderer.create_image_from_grid(
        timestep.state.grid,
        timestep.state.agent_pos,
        timestep.state.agent_dir,
        image_data)
    return image

housemaze_render_fn = jax.jit(housemaze_render_fn)

action_to_key = {
    int(KeyboardActions.right): "ArrowRight",
    int(KeyboardActions.down): "ArrowDown",
    int(KeyboardActions.left): "ArrowLeft",
    int(KeyboardActions.up): "ArrowUp",
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


def markdown_stage_display_fn(stage, container):
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
        #ui.markdown(f"#### {stage.instruction}")
        ui.markdown(f"#### Please obtain the {category}")
        with ui.row():
            with ui.element('div').classes('p-2 bg-blue-100'):
                ui.label(f"Number of successful episodes: {stage_state.nsuccesses}/{stage.min_success}") 
            with ui.element('div').classes('p-2 bg-green-100'):
                ui.label(
                    f"Try: {stage_state.nepisodes}/{stage.max_episodes}")
        text = f"You must complete at least {stage.min_success} episodes. You have {stage.max_episodes} tries."
        ui.html(text).style('align-items: center;')
        ui.html(make_image_html(src=image))


def make_env_stage(
        name,
        maze_str,
        max_episodes=1,
        min_success=1,
        training=True):
    return EnvStage(
        name=name,
        instruction='Please obtain the goal object',
        web_env=web_env,
        action_to_key=action_to_key,
        action_to_name=action_to_name,
        env_params=make_train_params(maze_str).replace(
          training=False,
          # if training sample_train=1.0
          # if testing, sample_train=0.0
          p_test_sample_train=float(training),
          ),
        render_fn=housemaze_render_fn,
        display_fn=env_stage_display_fn,
        evaluate_success_fn=evaluate_success_fn,
        state_cls=EnvStageState,
        max_episodes=1 if DEBUG else max_episodes,
        min_success=1 if DEBUG else min_success,
    )

stages = [
    Stage(
        name='Experiment instructions',
        body="""
        This is an experiment to test a person's ability to remember how to do a task that they could have done but didn't do.

        In the experiment, you'll control a red triangle in a maze. 

        During training you'll need to get an object. At evaluation, we'll test your ability to get a different object.

        Press the button when you're ready to begin.
        """,
        display_fn=stage_display_fn,
    ),
]
#############
# Practice
#############
stages.extend([
    Stage(
        name='Practice training',
        body="""
        Here you'll practice learning a task in the environment.

        You can control the red triangle with the arrow keys on your keyboard.
        Your goal is to move it to the goal object. 

        Press the button when you're ready to continue.
        """,
        display_fn=stage_display_fn,
    ),
    make_env_stage(
        'Practice',
        maze_str=getattr(mazes, 'maze1'),
        min_success=2,
        max_episodes=5,
        training=True),
    Stage(
        name='Practice evaluation',
        body="""
        Here you'll experience the task of getting an object that was nearby the other object you could have achieved.

        During the evaluation phase, you will only have 1 try. If you get this stage correct, you will double your earnings.

        Press the button when you're ready to continue.
        """,
        display_fn=stage_display_fn,
    ),
    make_env_stage(
        'Practice',
        maze_str=getattr(mazes, 'maze1'),
        min_success=1,
        max_episodes=1,
        training=False),
])
"""
Experiments Will be
1. Shortcut introduction (need to higlight open sections somehow)
  - train: maze3
  - test1: maze3_open2
2. Change starting location (need to highlight map changes)
  - test: maze3_onpath
  - test: maze3_onpath_shortcut
  - test: maze3_offpath_shortcut

3. closer/further object:
  - train: maze5
  - test: maze5
"""
stages.extend([
    Stage(
        name='Training on Maze 1',
        body="""
        Please learn to obtain the objects. You need to succeed 20 times.
        """,
        display_fn=stage_display_fn,
    ),
    make_env_stage(
        'Maze 1', maze_str=getattr(mazes, 'maze3'),
        min_success=20, max_episodes=30, training=True),
    Stage(
        name='Evaluation on Maze 1',
        body="""
        The following are evaluaton tasks. You will get 1 chance each time.

        **Note that some parts of the maze may have changed**.
        """,
        display_fn=stage_display_fn,
    ),
    #make_env_stage(
    #    'Maze 1', 'maze3_open',
    #    min_success=1, max_episodes=1, training=False),
    make_env_stage(
        'Maze 1', maze_str=getattr(mazes, 'maze3_open2'),
        min_success=1, max_episodes=1, training=False),
    make_env_stage(
        'Maze 1', maze_str=getattr(mazes, 'maze3_onpath'),
        min_success=1, max_episodes=1, training=False),
    make_env_stage(
        'Maze 1', maze_str=getattr(mazes, 'maze3_onpath_shortcut'),
        min_success=1, max_episodes=1, training=False),
    make_env_stage(
        'Maze 1', maze_str=getattr(mazes, 'maze3_offpath_shortcut'),
        min_success=1, max_episodes=1, training=False),
])

stages.extend([
    Stage(
        name='Training on Maze 2',
        body="""
        Please learn to obtain the objects. You need to succeed 20 times.
        """,
        display_fn=stage_display_fn,
    ),
    make_env_stage(
        'Maze 1', maze_str=getattr(mazes, 'maze5'),
        min_success=20, max_episodes=30, training=True),
    Stage(
        name='Evaluation on Maze 2',
        body="""
        The following are evaluaton tasks. You will get 1 chance each time.
        """,
        display_fn=stage_display_fn,
    ),
    make_env_stage(
        'Maze 1', maze_str=getattr(mazes, 'maze5'),
        min_success=1, max_episodes=1, training=False),
])
