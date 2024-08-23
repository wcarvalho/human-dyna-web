import dataclasses
import random
from typing import List
from functools import partial

from housemaze import renderer
from housemaze.maze import KeyboardActions
from housemaze.human_dyna import env
from housemaze.human_dyna import utils
from housemaze.human_dyna import env as maze
from housemaze.human_dyna import mazes
import numpy as np

import jax
import jax.numpy as jnp
from flax import struct
from dotenv import load_dotenv
import os
load_dotenv()

DEBUG = int(os.environ.get('DEBUG', 0))
USE_DONE = DEBUG > 0

from nicegui import ui
from nicewebrl import stages
from nicewebrl.stages import Stage, EnvStage, make_image_html, Block
from nicewebrl.nicejax import JaxWebEnv, base64_npimage

# number of rooms to user for tasks (1st n)
num_rooms = 2

min_success_train = 10*num_rooms
max_episodes_train = 30*num_rooms if DEBUG == 0 else min_success_train

image_data = utils.load_image_dict()
image_keys = image_data['keys']
groups = [
    # room 1
    [image_keys.index('orange'), image_keys.index('potato')],
    # room 2
    [image_keys.index('knife'), image_keys.index('spoon')],
    # room 3
    [image_keys.index('tomato'), image_keys.index('lettuce')],
]
groups = np.array(groups, dtype=np.int32)
task_objects = groups.reshape(-1)

# can auto-generate this from group_set
# with mazes.groups_to_char2key(groups)
char2idx = {
    # room 1
    'A': np.int32(image_keys.index('orange')),
    'B': np.int32(image_keys.index('potato')),
    # room 2
    'C': np.int32(image_keys.index('knife')),
    'D': np.int32(image_keys.index('spoon')),
    # room 3
    'E': np.int32(image_keys.index('tomato')),
    'F': np.int32(image_keys.index('lettuce')),
}
# same thing
char2idx = mazes.groups_to_char2key(groups)


# shared across all tasks
task_runner = env.TaskRunner(task_objects=task_objects)
keys = image_data['keys']

jax_env = env.HouseMaze(
    task_runner=task_runner,
    num_categories=len(keys),
    use_done=USE_DONE,
)
jax_env = utils.AutoResetWrapper(jax_env)


def permute_groups(groups):
    # Flatten the groups
    flattened = groups.flatten()

    # Create a random permutation
    permutation = np.random.permutation(len(flattened))

    # Apply the permutation
    permuted_flat = flattened[permutation]

    # Reshape back to the original shape
    new_groups = permuted_flat.reshape(groups.shape)

    # Create a new char2idx mapping
    new_char2idx = mazes.groups_to_char2key(new_groups)

    return new_groups, new_char2idx

def make_params(maze_str, groups, char2idx):
  if num_rooms < 3:
      if num_rooms == 1:
          # use room 0 (manipulation room only)
          groups = groups[:1]
      elif num_rooms == 2:
          # use room 0 (maipulation room) and 1st or 2nd room
          groups = np.array([groups[0], groups[np.random.randint(1, 3)]])
      # else: groups remains unchanged (all rooms)
  return mazes.get_maze_reset_params(
      groups=groups,
      char2key=char2idx,
      maze_str=maze_str,
      label=jnp.array(0),
      make_env_params=True,
  ).replace(
      terminate_with_done=jnp.array(2) if USE_DONE else jnp.array(0))

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
dummy_env_params = make_params(mazes.maze0, groups, char2idx)
dummy_timestep = web_env.reset(jax.random.PRNGKey(42), dummy_env_params)
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
        ui.markdown(f"#### Please retrieve the {category}")
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
        groups,
        char2idx,
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
        env_params=make_params(
            maze_str, 
            groups=groups,
            char2idx=char2idx).replace(
          training=False,
          # if eval, always force to room 0 (where we target manipualtion)
          force_room=jnp.array(not train_objects),
          default_room=jnp.array(0),
          # if training sample train objects w prob =1.0
          # if testing, sample train objects w prob=0.0
          p_test_sample_train=float(train_objects),
          ),
        render_fn=render_fn,
        vmap_render_fn=vmap_render_fn,
        display_fn=env_stage_display_fn,
        evaluate_success_fn=evaluate_success_fn,
        state_cls=EnvStageState,
        max_episodes=1 if DEBUG == 1 else max_episodes,
        min_success=1 if DEBUG == 1 else min_success,
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
if not DEBUG:
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
        groups=groups,
        char2idx=char2idx,
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
        groups=groups,
        char2idx=char2idx,
        train_objects=False),
], metadata=dict(desc="practice"))
if not DEBUG:
    all_blocks.append(practice_block)

##########################
# Manipulation 1: Shortcut
##########################
reversals = [(False, False), (True, False), (False, True), (True, True)]

for reversal in reversals[:2]:
    block_groups, block_char2idx = permute_groups(groups)
    block0 = Block([
        Stage(
            name='Training on Maze 1',
            body="""
        Please learn to obtain the objects. You need to succeed 10 times.
        """,
            display_fn=stage_display_fn,
        ),
        make_env_stage(
            'Maze 1', 
            maze_str=mazes.reverse(mazes.maze3, *reversal),
            metadata=dict(desc="training"),
            min_success=min_success_train,
            max_episodes=max_episodes_train,
            groups=block_groups,
            char2idx=block_char2idx,
            train_objects=True),
        Stage(
            name='Evaluation on Maze 1',
            body="""
        The following are evaluaton tasks. You will get 1 chance each time.

        <p style="color: red;"><strong>Note that some parts of the maze may have changed</strong>.</p>
        """,
            display_fn=stage_display_fn,
        ),
        make_env_stage(
            'Maze 1', 
            maze_str=mazes.reverse(mazes.maze3_open2, *reversal),
            metadata=dict(desc="'not obvious' shortcut"),
            min_success=1, max_episodes=1,
            groups=block_groups,
            char2idx=block_char2idx, train_objects=False),
    ],
    metadata=dict(
        manipulation=1,
        desc="shortcut",
        long=f"A shortcut is introduced. reverse h = {reversal[0]}, v = {reversal[1]}")
    )
    all_blocks.append(block0)
    if DEBUG: break

##########################
# Manipulation 2: Faster when on-path but further than off-path but closer
##########################

for reversal in reversals[2:]:
    block_groups, block_char2idx = permute_groups(groups)
    block1 = Block(stages=[
        Stage(
            name='Training on Maze 1',
            body="""
            Please learn to obtain the objects. You need to succeed 10 times.
            """,
            display_fn=stage_display_fn,
        ),
        make_env_stage(
            'Maze 1', maze_str=mazes.reverse(mazes.maze3, *reversal),
            metadata=dict(desc="training"),
            min_success=min_success_train, 
            max_episodes=max_episodes_train,
            groups=block_groups,
            char2idx=block_char2idx, train_objects=True),
        Stage(
            name='Evaluation on Maze 1',
            body="""
            The following are evaluaton tasks. You will get 1 chance each time.

            <p style="color: red;"><strong>Note that some parts of the maze may have changed</strong>.</p>
            """,
            display_fn=stage_display_fn,
        ),
        make_env_stage(
            'Maze 1', maze_str=mazes.reverse(mazes.maze3_onpath_shortcut, *reversal),
            metadata=dict(desc="Map changed, new location, on path"),
            min_success=1, max_episodes=1,
            groups=block_groups,
            char2idx=block_char2idx, train_objects=False),
        make_env_stage(
            'Maze 1', maze_str=mazes.reverse(mazes.maze3_offpath_shortcut, *reversal),
            metadata=dict(desc="Map changed, new location, off-path"),
            min_success=1, max_episodes=1,
            groups=block_groups,
            char2idx=block_char2idx, train_objects=False),
    ], metadata=dict(
        manipulation=2,
        desc="faster when on-path but further than off-path but closer",
        long=f"""
        In both tests, a shortcut is introduced. In the first, the agent is tested on the same path it trained on. In the second, the agent is tested on a different path.

        reverse h = {reversal[0]}, v = {reversal[1]}
        """))
    all_blocks.append(block1)
    if DEBUG: break


##########################
# Manipulation 3: reusing longer of two paths matching training path
##########################
for reversal in reversals:
    block_groups, block_char2idx = permute_groups(groups)
    block2 = Block([
        Stage(
            name='Training on Maze 2',
            body="""
            Please learn to obtain the objects. You need to succeed 10 times.
            """,
            display_fn=stage_display_fn,
        ),
        make_env_stage(
            'Maze 2', maze_str=mazes.reverse(mazes.maze5, *reversal),
            min_success=min_success_train,
            max_episodes=max_episodes_train,
            groups=block_groups,
            char2idx=block_char2idx,
            train_objects=True),
        Stage(
            name='Evaluation on Maze 2',
            body="""
            The following are evaluaton tasks. You will get 1 chance each time.
            """,
            display_fn=stage_display_fn,
        ),
        make_env_stage(
            'Maze 2', maze_str=mazes.reverse(mazes.maze5, *reversal),
            min_success=1, max_episodes=1,
            groups=block_groups,
            char2idx=block_char2idx, train_objects=False),
    ], metadata=dict(
        manipulation=3,
        desc="reusing longer of two paths matching training path",
        long=f"""
        Here there are two paths to the test object. We predict that people will take the path that was used to get to the training object.
        reverse h = {reversal[0]}, v = {reversal[1]}
        """))
    all_blocks.append(block2)
    if DEBUG: break


##########################
# Manipulation 4: Planning Near Goal
##########################
# last option, is full reverse.
# we built maze6 by doing a full reverse of maze 3. so don't want to re-use it.
for reversal in reversals[:-1]:
    block_groups, block_char2idx = permute_groups(groups)
    block3 = Block([
        Stage(
            name='Training on Maze 3',
            body="""
            Please learn to obtain the objects. You need to succeed 10 times.
            """,
            display_fn=stage_display_fn,
        ),
        make_env_stage(
            'Maze 3', maze_str=mazes.reverse(mazes.maze6, *reversal),
            metadata=dict(desc="training"),
            min_success=min_success_train,
            max_episodes=max_episodes_train,
            groups=block_groups,
            char2idx=block_char2idx,
            train_objects=True),
        Stage(
            name='Evaluation on Maze 3',
            body="""
            The following are evaluaton tasks. You will get 1 chance each time.
            """,
            display_fn=stage_display_fn,
        ),
        make_env_stage(
            'Maze 3', maze_str=mazes.reverse(mazes.maze6, *reversal),
            metadata=dict(desc="off-task object regular"),
            min_success=1, max_episodes=1,
            groups=block_groups,
            char2idx=block_char2idx,
            train_objects=False),
        make_env_stage(
            'Maze 3', maze_str=mazes.reverse(mazes.maze6_flipped_offtask, *reversal),
            metadata=dict(desc="off-task object flipped"),
            min_success=1, max_episodes=1,
            groups=block_groups,
            char2idx=block_char2idx,
            train_objects=False)],
        metadata=dict(
            manipulation=4,
            desc="probing for planning near goal",
            long=f"""
            At test time, we'll change the location of the off-task object so it's equidistant from path during training.
            We'll first query when the off-task object is in the same location as during training. We'll then query again with it being in a different locaiton.
            reverse h = {reversal[0]}, v = {reversal[1]}
            """
            ))
    all_blocks.append(block3)
    if DEBUG: break

all_stages = stages.prepare_blocks(all_blocks)



def generate_stage_order(rng_key):
    """Take blocks defined above, flatten all their stages, and generate an order where the (1) blocks are randomized, and (2) stages within blocks are randomized if they're consecutive eval stages."""
    randomized_blocks = list(all_blocks[2:])
    fixed_blocks = jnp.array([0, 1])  # instruct_block, practice_block
    random_order = jax.random.permutation(rng_key, len(randomized_blocks)) + 2
    block_order = jnp.concatenate([fixed_blocks, random_order])

    stage_order = stages.generate_stage_order(all_blocks, block_order, rng_key)
    return stage_order
