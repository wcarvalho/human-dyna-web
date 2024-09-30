from typing import List
from functools import partial

from skimage.transform import resize
import matplotlib.pyplot as plt
import asyncio
from housemaze import renderer
from housemaze.env import KeyboardActions
from housemaze.human_dyna import utils
from housemaze.human_dyna import multitask_env as maze
from housemaze.human_dyna import multitask_env
from housemaze.human_dyna import web_env
from housemaze.human_dyna import mazes
import numpy as np

import jax
import jax.numpy as jnp
from flax import struct
from dotenv import load_dotenv
import os
from experiment_utils import SuccessTrackingAutoResetWrapper


from nicegui import ui, app
from nicewebrl import stages
from nicewebrl.stages import Stage, EnvStage, Block
from nicewebrl.nicejax import JaxWebEnv, base64_npimage, make_serializable
from nicewebrl.utils import wait_for_button_or_keypress


load_dotenv()

GIVE_INSTRUCTIONS = int(os.environ.get('INST', 1))
DEBUG = int(os.environ.get('DEBUG', 0))
NMAN = int(os.environ.get('NMAN', 3))  # number of manipulations to keep

USE_REVERSALS = int(os.environ.get('REV', 0))
EVAL_OBJECTS = int(os.environ.get('EVAL_OBJECTS', 1))
TIMER = int(os.environ.get('TIMER', 60))
USE_DONE = DEBUG > 0

def remove_extra_spaces(text):
    """For each line, remove extra space."""
    return "\n".join([i.strip() for i in text.strip().split("\n")])

timer_text = ""
eval_objects_text = ""
if TIMER > 0:
    timer_text = f"* You will have a time-limit of {TIMER} seconds"
if EVAL_OBJECTS == 0:
    eval_objects_text = "* The object identities will not be visible on the map, so you'll need to learn where they are in phase 1."

if timer_text or eval_objects_text:
    phase_2_text = f"""
**Note that in phase 2:**

* You will only have 1 try.
{eval_objects_text}
{timer_text}
""".strip()
else:
    phase_2_text = "Note that in phase 2, you will only have 1 try"

# number of rooms to user for tasks (1st n)
num_rooms = 2


min_success_task = 8
min_success_train = min_success_task*num_rooms
max_episodes_train = 30*num_rooms
if DEBUG == 0:
    pass
elif DEBUG == 1:
    min_success_task = 1
    min_success_train = 1


image_data = utils.load_image_dict()
image_keys = image_data['keys']
groups = [
    # room 1
    [image_keys.index('orange'), image_keys.index('potato')],
    # room 2
    [image_keys.index('knife'), image_keys.index('spoon')],
    ## room 3
    #[image_keys.index('tomato'), image_keys.index('lettuce')],
]
groups = np.array(groups, dtype=np.int32)
task_objects = groups.reshape(-1)

# can auto-generate this from group_set
char2idx = mazes.groups_to_char2key(groups)

# shared across all tasks
task_runner = multitask_env.TaskRunner(task_objects=task_objects)
keys = image_data['keys']

jax_env = web_env.HouseMaze(
    task_runner=task_runner,
    num_categories=len(keys),
    use_done=USE_DONE,
)
jax_env = SuccessTrackingAutoResetWrapper(
    jax_env,
    num_success=min_success_task)


def permute_groups(groups):
    if DEBUG:
        return groups, char2idx
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


def make_params(maze_str, groups, char2idx, randomize_agent: bool = False):
  return mazes.get_maze_reset_params(
      groups=groups,
      char2key=char2idx,
      maze_str=maze_str,
      label=jnp.array(0),
      make_env_params=True,
      randomize_agent=randomize_agent,
  ).replace(
      terminate_with_done=jnp.array(2) if USE_DONE else jnp.array(0))


def render_fn(timestep: maze.TimeStep, include_objects: bool = True) -> jnp.ndarray:
    image = renderer.create_image_from_grid(
        timestep.state.grid,
        timestep.state.agent_pos,
        timestep.state.agent_dir,
        image_data,
        include_objects=include_objects)
    return image



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
# Call this function to pre-compile jax functions before experiment starts.
dummy_env_params = make_params(mazes.big_practice_maze, groups, char2idx)
dummy_timestep = web_env.reset(jax.random.PRNGKey(42), dummy_env_params)
web_env.precompile(dummy_env_params=dummy_env_params)


train_render_fn = jax.jit(render_fn)
if EVAL_OBJECTS:
    eval_render_fn = train_render_fn
else:
    eval_render_fn = jax.jit(partial(render_fn, include_objects=False))
train_vmap_render_fn = web_env.precompile_vmap_render_fn(
    train_render_fn, dummy_env_params)
eval_vmap_render_fn = web_env.precompile_vmap_render_fn(
    eval_render_fn, dummy_env_params)


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
        ui.markdown(f"{remove_extra_spaces(stage.body)}",
                    extras=['cuddled-lists'])
        
        ui.markdown(f"##### Task objects will be selected from the following set")
        idxs = char2idx.values()
        width = 1
        figsize = (len(idxs)*width, width)
        with ui.matplotlib(figsize=figsize).figure as fig:
            axs = fig.subplots(1, len(idxs))
            for i, idx in enumerate(idxs):
                axs[i].imshow(image_data['images'][idx])
                axs[i].set_title(keys[idx])
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                axs[i].axis("off")
            # Adjust layout
            fig.tight_layout()

def make_image_html(src):
    html = '''
    <div id="stateImageContainer" style="display: flex; justify-content: center; align-items: center; height: 100%;">
        <img id="stateImage" src="{src}" style="max-width: 800px; max-height: 400px; object-fit: contain;">
    </div>
    '''.format(src=src)
    return html

def debug_info(stage):
    stage_state = stage.get_user_data('stage_state')
    debug_info = f"**Manipulation**: {stage.metadata['block_metadata'].get('manipulation')}. "
    debug_info += f"**Episode** idx: {stage_state.nepisodes}. "
    debug_info += f"**Eval**: {stage.metadata['eval']}. "
    return debug_info

async def env_reset_display_fn(
        stage,
        container,
        timestep):
    category = keys[timestep.state.task_object]
    image = image_data['images'][timestep.state.task_object]
    image = resize(image, (64, 64, 3),
                   anti_aliasing=True, preserve_range=True)
    image = base64_npimage(image)

    with container.style('align-items: center;'):
        container.clear()
        ui.markdown(f"#### Goal object: {category}")
        if DEBUG:
            ui.markdown(debug_info(stage))
        ui.html(make_image_html(src=image))
        button = ui.button("click to start")
        await wait_for_button_or_keypress(button)

def env_stage_display_fn(
        stage,
        container,
        timestep):
    state_image = stage.render_fn(timestep)
    state_image = base64_npimage(state_image)
    #category = keys[timestep.state.task_object]

    object_image = image_data['images'][timestep.state.task_object]

    stage_state = stage.get_user_data('stage_state')
    with container.style('align-items: center;'):
        container.clear()
        #ui.markdown(f"#### Goal object: {category}")
        with ui.matplotlib(figsize=(1,1)).figure as fig:
            ax = fig.subplots(1, 1)
            ax.set_title(f"Goal")
            ax.imshow(object_image)
            ax.axis('off')
            fig.tight_layout()
        if DEBUG:
            ui.markdown(debug_info(stage))
        with ui.row():
            with ui.element('div').classes('p-2 bg-blue-100'):
                n = timestep.state.successes.sum()
                ui.label(
                    f"Number of successful episodes: {n}/{stage.min_success}")
            with ui.element('div').classes('p-2 bg-green-100'):
                ui.label().bind_text_from(
                    stage_state, 'nepisodes', lambda n: f"Try: {n}/{stage.max_episodes}")

        text = f"You must complete at least {stage.min_success} episodes. You have {stage.max_episodes} tries."
        ui.html(text).style('align-items: center;')
        ui.html(make_image_html(src=state_image))


def make_env_stage(
        maze_str,
        groups,
        char2idx,
        max_episodes=1,
        min_success=1,
        training=True,
        force_room=False,
        metadata=None,
        randomize_agent: bool = True,
        render_fn=None,
        vmap_render_fn=None,
        ):
    metadata = metadata or {}
    eval = not training
    metadata.update(eval=eval)

    randomize_agent = randomize_agent and training
    
    # Use the provided render_fn if given, otherwise use the default
    if render_fn is None:
        render_fn = train_render_fn if training else eval_render_fn
    
    # Use the provided vmap_render_fn if given, otherwise use the default
    if vmap_render_fn is None:
        vmap_render_fn = train_vmap_render_fn if training else eval_vmap_render_fn
    
    return EnvStage(
        web_env=web_env,
        action_to_key=action_to_key,
        action_to_name=action_to_name,
        env_params=make_params(
            maze_str, 
            groups=groups,
            char2idx=char2idx,
            randomize_agent=randomize_agent).replace(
                randomize_agent=randomize_agent,
                training=training,
                # if eval, always force to room 0 (where we target manipualtion)
                force_room=jnp.array(force_room or not training),
                default_room=jnp.array(0),
                # if training sample train objects w prob =1.0
                # if testing, sample train objects w prob=0.0
                p_test_sample_train=1.0,
          ),
        render_fn=render_fn,
        vmap_render_fn=vmap_render_fn,
        #reset_display_fn=env_reset_display_fn,
        display_fn=env_stage_display_fn,
        evaluate_success_fn=lambda t: int(t.reward > .5),
        check_finished=lambda t: t.finished,
        state_cls=EnvStageState,
        max_episodes=max_episodes,
        min_success=min_success,
        metadata=metadata,
        duration=TIMER if eval else None,
        notify_success=True if training else False,
    )


all_blocks = []

##########################
# Instructions
##########################
instruct_block = Block([
    Stage(
        name='Experiment instructions',
        body=f"""
This experiment tests how people solve new tasks. You'll control a red triangle in a maze. Individual tasks will consist of navigating to objects.

The experiment will consist blocks, each with two phases. In the first phase, you'll learn how to navigate to objects. In the second hase, we will will ask you to navigate to different objects.

{phase_2_text}
""",
        display_fn=stage_display_fn,
    ),
])
if GIVE_INSTRUCTIONS:
    all_blocks.append(instruct_block)

##########################
# Practice
##########################
practice_block = Block(stages=[
    Stage(
        name='Practice phase 1',
        body="""
Here you'll get some experience in a practice maze.

You can control the red triangle to move around the maze with the arrow keys on your keyboard.

Press the button when you're ready to continue.
        """,
        display_fn=stage_display_fn,
    ),
    make_env_stage(
        maze_str=mazes.big_practice_maze,
        min_success=num_rooms,
        max_episodes=5*num_rooms,
        groups=groups,
        char2idx=char2idx,
        force_room=True,
        training=True,
        metadata={'maze': 'big_practice_maze'}),
    Stage(
        name='Practice phase 2',
        body=f"""
Here you'll experience the task of getting an object that was nearby one of the training objects.

{phase_2_text}

Press the button when you're ready to continue.
        """,
        display_fn=stage_display_fn,
    ),
    make_env_stage(
        maze_str=mazes.big_practice_maze,
        min_success=num_rooms,
        max_episodes=1,
        groups=groups,
        char2idx=char2idx,
        force_room=True,
        training=False,
        metadata={'maze': 'big_practice_maze'}),
], metadata=dict(desc="practice"))
if GIVE_INSTRUCTIONS:
    all_blocks.append(practice_block)

###############################################################
# Manipulations
###############################################################
manipulation_groups = []
reversals = [(False, False), (True, False), (False, True), (True, True)]

##########################
# Manipulation 1: Shortcut
##########################
manipulation1_blocks = []
for reversal in reversals:
    block_groups, block_char2idx = permute_groups(groups)
    block0 = Block([
        Stage(
            name='Phase 1',
            body=f"""
            Please learn to obtain these objects. You need to succeed {min_success_task} times per object.

            If you retrieve the wrong object, the episode terminates early.
            """,
            display_fn=stage_display_fn,
        ),
        make_env_stage(
            maze_str=mazes.reverse(mazes.big_m1_maze3, *reversal),
            metadata=dict(desc="training", maze="big_m1_maze3"),
            min_success=min_success_train,
            max_episodes=max_episodes_train,
            groups=block_groups,
            char2idx=block_char2idx,
            training=True),
        Stage(
            name='Phase 2',
            body=f"""
        Your performance here will count towards your bonus payment. Go as fast as you can.

        {phase_2_text}

        <p style="color: red;"><strong>Note that some parts of the maze may have changed</strong>.</p>
        """,
            display_fn=stage_display_fn,
        ),
        make_env_stage(
            maze_str=mazes.reverse(mazes.big_m1_maze3_shortcut, *reversal),
            metadata=dict(desc="shortcut", maze="big_m1_maze3_shortcut"),
            min_success=1, max_episodes=1,
            groups=block_groups,
            char2idx=block_char2idx, training=False),
    ],
    metadata=dict(
        manipulation=1,
        desc="shortcut",
        long=f"A shortcut is introduced",
        groups=make_serializable(block_groups),
        char2idx=jax.tree_map(int, block_char2idx)
    ))
    manipulation1_blocks.append(block0)
    if not USE_REVERSALS: break
manipulation_groups.append(manipulation1_blocks)

##########################
# Manipulation 2: Faster when on-path but further than off-path but closer
##########################
manipulation2_blocks = []
for reversal in reversals:
    block_groups, block_char2idx = permute_groups(groups)

    block1 = Block(stages=[
        Stage(
            name='Phase 1',
            body=f"""
            Please learn to obtain these objects. You need to succeed {min_success_task} times per object.

            If you retrieve the wrong object, the episode terminates early.
            """,
            display_fn=stage_display_fn,
        ),
        make_env_stage(
            maze_str=mazes.reverse(mazes.big_m2_maze2, *reversal),
            metadata=dict(desc="training", maze="big_m2_maze2"),
            min_success=min_success_train, 
            max_episodes=max_episodes_train,
            groups=block_groups,
            char2idx=block_char2idx, training=True),
        Stage(
            name='Phase 2',
            body=f"""
            Your performance here will count towards your bonus payment. Try to reuse what you learned as best you can.

            {phase_2_text}
            """,
            display_fn=stage_display_fn,
        ),
        make_env_stage(
            maze_str=mazes.reverse(mazes.big_m2_maze2_onpath, *reversal),
            metadata=dict(desc="new location, on-path",
                          maze="big_m2_maze2_offpath"),
            min_success=1, max_episodes=1,
            groups=block_groups,
            char2idx=block_char2idx, training=False),
        make_env_stage(
            maze_str=mazes.reverse(mazes.big_m2_maze2_offpath, *reversal),
            metadata=dict(desc="new location, off-path",
                          maze="big_m2_maze2_offpath"),
            min_success=1, max_episodes=1,
            groups=block_groups,
            char2idx=block_char2idx,
            training=False,
            render_fn=train_render_fn,
            vmap_render_fn=train_vmap_render_fn,
            ),
    ], metadata=dict(
        manipulation=2,
        desc="faster when on-path but further than off-path but closer",
        long=f"""
        In both tests, a shortcut is introduced. In the first, the agent is tested on the same path it trained on. In the second, the agent is tested on a different path.
        """,
        groups=make_serializable(block_groups),
        char2idx=jax.tree_map(int, block_char2idx)
    ))
    manipulation2_blocks.append(block1)
    if not USE_REVERSALS: break
manipulation_groups.append(manipulation2_blocks)

##########################
# Manipulation 3: reusing longer of two paths matching training path
##########################
manipulation3_blocks = []
for reversal in reversals:
    block_groups, block_char2idx = permute_groups(groups)
    block2 = Block([
        Stage(
            name='Phase 1',
            body=f"""
            Please learn to obtain these objects. You need to succeed {min_success_task} times per object.

            If you retrieve the wrong object, the episode terminates early.
            """,
            display_fn=stage_display_fn,
        ),
        make_env_stage(
            maze_str=mazes.reverse(mazes.big_m3_maze1, *reversal),
            min_success=min_success_train,
            max_episodes=max_episodes_train,
            groups=block_groups,
            char2idx=block_char2idx,
            training=True,
            metadata={'maze': 'big_m3_maze1'}),
        Stage(
            name='Phase 2',
            body=f"""
            Your performance here will count towards your bonus payment. Try to reuse what you learned as best you can.

            {phase_2_text}
            """,
            display_fn=stage_display_fn,
        ),
        make_env_stage(
            maze_str=mazes.reverse(mazes.big_m3_maze1, *reversal),
            min_success=1,
            max_episodes=1,
            groups=block_groups,
            char2idx=block_char2idx,
            training=False,
            metadata={'maze': 'big_m3_maze1'}),
    ], metadata=dict(
        manipulation=3,
        desc="reusing longer of two paths which matches training path",
        long=f"""
        Here there are two paths to the test object. We predict that people will take the path that was used to get to the training object.
        """,
        groups=make_serializable(block_groups),
        char2idx=jax.tree_map(int, block_char2idx)
    ))
    manipulation3_blocks.append(block2)
    if not USE_REVERSALS: break
manipulation_groups.append(manipulation3_blocks)


# Select the specified number of manipulation groups and flatten
manipulations = manipulation_groups[:NMAN]
for manipulation_blocks in manipulations:
    all_blocks.extend(manipulation_blocks)

all_stages = stages.prepare_blocks(all_blocks)



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
    block_order = jnp.concatenate([fixed_blocks, random_order]).astype(jnp.int32)
    block_order = block_order.tolist()

    stage_order = stages.generate_stage_order(all_blocks, block_order, rng_key)

    return block_order, stage_order
