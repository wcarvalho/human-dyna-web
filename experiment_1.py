import dataclasses
from typing import List
from functools import partial

from housemaze import renderer
from housemaze.maze import KeyboardActions
from housemaze.human_dyna import env
from housemaze.human_dyna import utils
from housemaze.human_dyna import env as maze
from housemaze.human_dyna import mazes

import jax
import jax.numpy as jnp
from flax import struct
from dotenv import load_dotenv
import os
load_dotenv()

DEBUG = int(os.environ.get('DEBUG', 0))

from nicegui import ui
from nicewebrl import stages
from nicewebrl.stages import Stage, EnvStage, make_image_html, Block
from nicewebrl.nicejax import JaxWebEnv, base64_npimage

min_success_train = 10
num_rooms = 3          # number of pairs to recognize interaction for
num_pairs_train = 1  # number of objects to be train
num_pairs_test = 1  # number of objects to be train
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


def make_params(maze_str, train_objects: bool = True):
  num_pairs = num_pairs_train if train_objects else num_pairs_test
  return mazes.get_maze_reset_params(
      group_set=group_set[:num_pairs],
      char2key=char2key,
      maze_str=maze_str,
      label=jnp.array(0),
      make_env_params=True,
  ).replace(
        terminate_with_done=jnp.array(2),
  )


def render_fn(timestep: maze.TimeStep) -> jnp.ndarray:
    image = renderer.create_image_from_grid(
        timestep.state.grid,
        timestep.state.agent_pos,
        timestep.state.agent_dir,
        image_data)
    return image

render_fn = jax.jit(render_fn)

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
dummy_env_params = make_params(mazes.maze0)
web_env.precompile(dummy_env_params=dummy_env_params)
vmap_render_fn = web_env.precompile_vmap_render_fn(
    render_fn, dummy_env_params)


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

def env_stage_display_fn(
        stage,
        container,
        timestep):
    image = render_fn(timestep)
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
                ui.label().bind_text_from(
                    stage_state, 'nsuccesses', lambda n: f"Number of successful episodes: {n}/{stage.min_success}")
            with ui.element('div').classes('p-2 bg-green-100'):
                ui.label().bind_text_from(
                    stage_state, 'nepisodes', lambda n: f"Try: {n}/{stage.max_episodes}")

        text = f"You must complete at least {stage.min_success} episodes. You have {stage.max_episodes} tries."
        ui.html(text).style('align-items: center;')
        ui.html(make_image_html(src=image))


def make_env_stage(
        name,
        maze_str,
        max_episodes=1,
        min_success=1,
        train_objects=True,
        metadata=dict(),
        ):
    metadata.update(eval=not train_objects)
    return EnvStage(
        name=name,
        instruction='Please obtain the goal object',
        web_env=web_env,
        action_to_key=action_to_key,
        action_to_name=action_to_name,
        env_params=make_params(maze_str, train_objects=train_objects).replace(
          training=False,
          # if training sample train objects w prob =1.0
          # if testing, sample train objects w prob=0.0
          p_test_sample_train=float(train_objects),
          ),
        render_fn=render_fn,
        vmap_render_fn=vmap_render_fn,
        display_fn=env_stage_display_fn,
        evaluate_success_fn=evaluate_success_fn,
        state_cls=EnvStageState,
        max_episodes=1 if DEBUG else max_episodes,
        min_success=1 if DEBUG else min_success,
        metadata=metadata,
    )


all_blocks = []

##########################
# Instructions
##########################
instruct_block = Block([
    Stage(
        name='Experiment instructions',
        body="""
        This experiment tests how people solve new tasks.
        Br  
        In the experiment, you'll control a red triangle in a maze.
        <br>
        <br>
        We will first give you some training scenarios, where we ask to retrieve an object from the maze. After that, we will ask you to retrieve different objects.
        <br>
        Press the button when you're ready to begin.
        """,
        display_fn=stage_display_fn,
    ),
])
all_blocks.append(instruct_block)

##########################
# Practice
##########################
maze1 = """
.#.C...##....
.#..D...####.
.######......
......######.
.#.#..#......
.#.#.##..#...
##.#.#>.###.#
A..#.##..#...
.B.#.........
#####.#..####
......####.#.
.######E.#.#.
........F#...
""".strip()

practice_block = Block(stages=[
    Stage(
        name='Practice training',
        body="""
        Here you'll get some experience in a practice maze.
        You can control the red triangle to move around the maze with the arrow keys on your keyboard. Your goal is to move to the goal object shown on screen.
        <br>
        <br>
        Press the button when you're ready to continue.
        """,
        display_fn=stage_display_fn,
    ),
    make_env_stage(
        'Practice',
        maze_str=maze1,
        min_success=2,
        max_episodes=5,
        train_objects=True),
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
        maze_str=mazes.maze1,
        min_success=1,
        max_episodes=1,
        train_objects=False),
], metadata=dict(desc="practice"))
all_blocks.append(practice_block)

##########################
# Manipulation 1: Shortcut
##########################
"""
Manipulation 1: shortcut
---
Shortcut introduction (need to higlight open sections somehow)
  - train: maze3
  - test1: maze3_open2
"""

block0 = Block(stages=[
    Stage(
        name='Training on Maze 1',
        body="""
    Please learn to obtain the objects. You need to succeed 10 times.
    """,
        display_fn=stage_display_fn,
    ),
    make_env_stage(
        'Maze 1', maze_str=mazes.maze3,
        metadata=dict(desc="training"),
        min_success=min_success_train, max_episodes=30, train_objects=True),
    Stage(
        name='Evaluation on Maze 1',
        body="""
    The following are evaluaton tasks. You will get 1 chance each time.

    **Note that some parts of the maze may have changed**.
    """,
        display_fn=stage_display_fn,
    ),
    make_env_stage(
        'Maze 1', maze_str=mazes.maze3_open2,
        metadata=dict(desc="'hard' shortcut"),
        min_success=1, max_episodes=1, train_objects=False),
], metadata=dict(manipulation=1, desc="shortcut"))
all_blocks.append(block0)

##########################
# Manipulation 2: Faster when on-path but further than off-path but closer
##########################
"""
Manpulation 2: Faster when on-path but further than off-path but closer
---
Change starting location (maybe need to highlight map changes?)
  - test: maze3_onpath
  - test: maze3_onpath_shortcut
  - test: maze3_offpath_shortcut
"""

block1 = Block(stages=[
    Stage(
        name='Training on Maze 1',
        body="""
        Please learn to obtain the objects. You need to succeed 10 times.
        """,
        display_fn=stage_display_fn,
    ),
    make_env_stage(
        'Maze 1', maze_str=mazes.maze3,
        metadata=dict(desc="training"),
        min_success=min_success_train, max_episodes=30, train_objects=True),
    Stage(
        name='Evaluation on Maze 1',
        body="""
        The following are evaluaton tasks. You will get 1 chance each time.

        **Note that some parts of the maze may have changed**.
        """,
        display_fn=stage_display_fn,
    ),
    make_env_stage(
        'Maze 1', maze_str=mazes.maze3_onpath_shortcut,
        metadata=dict(desc="Map changed, new location, on path"),
        min_success=1, max_episodes=1, train_objects=False),
    make_env_stage(
        'Maze 1', maze_str=mazes.maze3_offpath_shortcut,
        metadata=dict(desc="Map changed, new location, off-path"),
        min_success=1, max_episodes=1, train_objects=False),
], metadata=dict(
    manipulation=2, desc="faster when on-path but further than off-path but closer"))
all_blocks.append(block1)



"""
Manipulation 3: Reusing longer of two paths if training path
---
Shortcut introduction (need to higlight open sections somehow)
  - train: maze5
  - test: maze5
"""
##########################
# Manipulation 3: reusing longer of two paths matching training path
##########################
block2 = Block([
    Stage(
        name='Training on Maze 2',
        body="""
        Please learn to obtain the objects. You need to succeed 10 times.
        """,
        display_fn=stage_display_fn,
    ),
    make_env_stage(
        'Maze 2', maze_str=mazes.maze5,
        min_success=min_success_train, max_episodes=30, train_objects=True),
    Stage(
        name='Evaluation on Maze 2',
        body="""
        The following are evaluaton tasks. You will get 1 chance each time.
        """,
        display_fn=stage_display_fn,
    ),
    make_env_stage(
        'Maze 2', maze_str=mazes.maze5,
        min_success=1, max_episodes=1, train_objects=False),
], metadata=dict(manipulation=3, desc="reusing longer of two paths matching training path"))
all_blocks.append(block2)

"""
Manipulation 4: probing for planning near goal
---
At test time, change the location of the off-task object so it's equidistant from path during training.
  - train: maze6
  - test: maze6_flipped_offtask
"""
##########################
# Manipulation 4: Planning Near Goal
##########################
block3 = Block([
    Stage(
        name='Training on Maze 3',
        body="""
        Please learn to obtain the objects. You need to succeed 10 times.
        """,
        display_fn=stage_display_fn,
    ),
    make_env_stage(
        'Maze 3', maze_str=mazes.maze6,
        metadata=dict(desc="training"),
        min_success=min_success_train, max_episodes=30, train_objects=True),
    Stage(
        name='Evaluation on Maze 3',
        body="""
        The following are evaluaton tasks. You will get 1 chance each time.
        """,
        display_fn=stage_display_fn,
    ),
    make_env_stage(
        'Maze 3', maze_str=mazes.maze6,
        metadata=dict(desc="off-task object regular"),
        min_success=1, max_episodes=1, train_objects=False),
    make_env_stage(
        'Maze 3', maze_str=mazes.maze6_flipped_offtask,
        metadata=dict(desc="off-task object flipped"),
        min_success=1, max_episodes=1, train_objects=False)],
    metadata=dict(manipulation=4, desc="probing for planning near goal"))
all_blocks.append(block3)

all_stages = stages.prepare_blocks(all_blocks)



def generate_stage_order(rng_key):
    """Take blocks defined above, flatten all their stages, and generate an order where the (1) blocks are randomized, and (2) stages within blocks are randomized if they're consecutive eval stages."""
    randomized_blocks = list(all_blocks[2:])
    fixed_blocks = jnp.array([0, 1])  # instruct_block, practice_block
    random_order = jax.random.permutation(rng_key, len(randomized_blocks)) + 2
    block_order = jnp.concatenate([fixed_blocks, random_order])

    stage_order = stages.generate_stage_order(all_blocks, block_order, rng_key)
    return stage_order
