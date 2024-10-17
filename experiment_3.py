from typing import List, Callable
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
from nicewebrl.stages import Stage, EnvStage, Block, FeedbackStage
from nicewebrl.stages import ExperimentData
from nicewebrl.nicejax import JaxWebEnv, base64_npimage, make_serializable
from nicewebrl.utils import wait_for_button_or_keypress
from nicewebrl import nicejax

load_dotenv()

GIVE_INSTRUCTIONS = int(os.environ.get('INST', 1))
DEBUG = int(os.environ.get('DEBUG', 0))
NMAN = int(os.environ.get('NMAN', 0))  # number of manipulations to keep

#USE_REVERSALS = int(os.environ.get('REV', 0))
#EVAL_OBJECTS = int(os.environ.get('EVAL_OBJECTS', 1))
SAY_REUSE = int(os.environ.get('SAY_REUSE', 1))
TIMER = int(os.environ.get('TIMER', 45))
USE_DONE = DEBUG > 0

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

##############################################
# Creating environment stuff
##############################################
image_data = utils.load_image_dict()

def create_env_params(
    maze_str,
    groups,
    char2idx,
    randomize_agent=False,
    use_done=False,
    training=True,
    force_room=False,
    label=0,
    time_limit=10_000_000,
    default_room=0,
    p_test_sample_train=1.0
):
    env_params = mazes.get_maze_reset_params(
        groups=groups,
        char2key=char2idx,
        maze_str=maze_str,
        label=jnp.array(label),
        make_env_params=True,
        randomize_agent=randomize_agent,
    ).replace(
        time_limit=time_limit,
        terminate_with_done=jnp.array(2) if use_done else jnp.array(0),
        randomize_agent=randomize_agent,
        training=training,
        force_room=jnp.array(force_room or not training),
        default_room=jnp.array(default_room),
        p_test_sample_train=p_test_sample_train,
    )
    return env_params

def permute_groups(groups):
    if DEBUG:
        return groups, block_char2idx
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

def housemaze_render_fn(timestep: maze.TimeStep, include_objects: bool = True) -> jnp.ndarray:
    image = renderer.create_image_from_grid(
        timestep.state.grid,
        timestep.state.agent_pos,
        timestep.state.agent_dir,
        image_data,
        include_objects=include_objects)
    return image



@struct.dataclass
class EnvStageState:
    timestep: maze.TimeStep = None
    nsteps: int = 0
    nepisodes: int = 0
    nsuccesses: int = 0



image_keys = image_data['keys']
block_groups = [
    # room 1
    [image_keys.index('orange'), image_keys.index('potato')],
    # room 2
    [image_keys.index('knife'), image_keys.index('spoon')],
    ## room 3
    #[image_keys.index('tomato'), image_keys.index('lettuce')],
]
block_groups = np.array(block_groups, dtype=np.int32)
task_objects = block_groups.reshape(-1)

# can auto-generate this from group_set
block_char2idx = mazes.groups_to_char2key(block_groups)

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

jax_web_env = JaxWebEnv(jax_env)
# Call this function to pre-compile jax functions before experiment starts.
dummy_env_params = create_env_params(
    maze_str=mazes.big_practice_maze,
    groups=block_groups,
    char2idx= block_char2idx)
dummy_timestep = jax_web_env.reset(jax.random.PRNGKey(42), dummy_env_params)
jax_web_env.precompile(dummy_env_params=dummy_env_params)


render_fn = jax.jit(housemaze_render_fn)
vmap_render_fn = jax_web_env.precompile_vmap_render_fn(
    render_fn, dummy_env_params)


def went_to_junction(timestep, junction):
    position = timestep.state.agent_pos
    match = np.array(junction) == position
    match = match.sum(-1) == 2  # both x and y matches
    return match.any()

def manip3_data_fn(timestep):
  return {
      'old_path': went_to_junction(timestep, junction=(14, 25)),
      'new_path': went_to_junction(timestep, junction=(3, 11))}


##############################################
# Block/Stage Utility functions
##############################################
def remove_extra_spaces(text):
    """For each line, remove extra space."""
    return "\n".join([i.strip() for i in text.strip().split("\n")])


def debug_info(stage):
    stage_state = stage.get_user_data('stage_state')
    debug_info = f"**Manipulation**: {stage.metadata['block_metadata'].get('manipulation')}. "
    if stage_state is not None:
        debug_info += f"**Eval**: {stage.metadata['eval']}. "
        debug_info += f"**Episode** idx: {stage_state.nepisodes}. "
        debug_info += f"**Step**: {stage_state.nsteps}/{stage.env_params.time_limit}. "
    return debug_info


def instruct_display_fn(stage, container):
    with container.style('align-items: center;'):
        container.clear()
        ui.markdown(f"## {stage.name}")
        ui.markdown(f"{remove_extra_spaces(stage.body)}",
                    extras=['cuddled-lists'])
        ui.markdown("Task objects will be selected from the set below.")

        cats = [int(i) for i in block_char2idx.values()]
        width = 1
        figsize = (len(cats)*width, width)
        with ui.matplotlib(figsize=figsize).figure as fig:
            axs = fig.subplots(1, len(cats))
            for i, cat in enumerate(cats):
                axs[i].imshow(image_data['images'][cat])
                axs[i].set_title(f'{keys[cat]}')
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                axs[i].axis("off")
            # Adjust layout
            fig.tight_layout()

def stage_display_fn(stage, container):
    with container.style('align-items: center;'):
        container.clear()
        ui.markdown(f"## {stage.name}")
        if DEBUG:
            ui.markdown(debug_info(stage))
        ui.markdown(f"{remove_extra_spaces(stage.body)}",
                    extras=['cuddled-lists'])
        
        ui.markdown("Task objects will be selected from the set below.")
        ui.markdown("**We note objects relevant to phase 2**")
        #phase 2 reward
        #idxs = char2idx.values()
        groups = stage.metadata['block_metadata'].get('groups', None)
        cats = groups[0] + groups[1]
        eval_prices = [0, 1, 0, 0]
        key = nicejax.new_rng()
        order = jax.random.permutation(
            key, jnp.arange(len(cats)))

        width = 1
        figsize = (len(cats)*width, width)
        with ui.matplotlib(figsize=figsize).figure as fig:
            axs = fig.subplots(1, len(order))
            for i, idx in enumerate(order):
                cat = cats[idx]
                axs[i].imshow(image_data['images'][cat])
                axs[i].set_title(
                    f'{keys[cat]}: {eval_prices[idx]}' if eval_prices[idx] == 0 else f'{keys[cat]}: {eval_prices[idx]}',
                    fontsize=10, color='green' if eval_prices[idx] != 0 else 'black',
                    weight='bold' if eval_prices[idx] != 0 else 'normal')
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
        await wait_for_button_or_keypress(
            button, ignore_recent_press=True)

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

async def feedback_display_fn(
        stage,
        container,
        name: str = 'big_m3_maze1_eval'):
    container.clear()
    output = {}
    with container.style('align-items: center;'):
        user_data = await ExperimentData.filter(
            session_id=app.storage.browser['id'],
            name=name,
        )
        used_old_path = [d.user_data.get('old_path', False)
                        for d in user_data]
        used_old_path = np.array(used_old_path).any()

        ########
        # radio
        #######
        groups = user_data[0].metadata['block_metadata'].get('groups', None)
        phase2_train = keys[groups[0][0]]
        phase2_test = keys[groups[0][1]]
        ui.html(f"Did you notice that the phase 2 object ({phase2_test}) was accessible from the path towards the phase 1 object ({phase2_train})?")
        #radio = ui.radio({1: "Yes", 2: "No", 3: "I'm not sure"}, value=3).props('inline')
        #output['noticed_path'] = radio.value

        ########
        # freeform
        ########
        if used_old_path is None: 
            output['feedback'] = None
            return output

        if used_old_path:
            text = f"You used the same path as in Phase 1. Please briefly describe why."
        else:
            text = f"You used a different path as in Phase 1. Please briefly describe why. For example, did you re-plan how to get the object?"
        output['question'] = text
        timestep = user_data[0].data['timestep']
        timestep = nicejax.deserialize_bytes(maze.TimeStep, timestep)
        image = render_fn(timestep)

        # Calculate aspect ratio and set figure size
        height, width = image.shape[:2]
        aspect_ratio = width / height
        fig_width = 6
        fig_height = fig_width / aspect_ratio

        with ui.matplotlib(
            figsize=(int(fig_width), int(fig_height))).figure as fig:
            ax = fig.subplots(1, 1)
            ax.imshow(image)
            ax.axis('off')
        ui.html(f"{text}")
        text = ui.textarea().style('width: 80%;')  # Set width to 80% of the container
        button = ui.button("Submit")
        await button.clicked()
        feedback = text.value
        output['feedback'] = feedback
    return output


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
        custom_data_fn=None,
        duration=None,
        name='stage',
        use_done=False,
        ):
    metadata = metadata or {}
    metadata['eval'] = not training

    randomize_agent = randomize_agent and training

    env_params = create_env_params(
        groups=groups,
        char2idx=char2idx,
        maze_str=maze_str,
        randomize_agent=randomize_agent,
        use_done=use_done,
        training=training,
        force_room=force_room
    )

    return EnvStage(
        name=name,
        web_env=jax_web_env,
        action_to_key=action_to_key,
        action_to_name=action_to_name,
        env_params=env_params,
        render_fn=render_fn,
        vmap_render_fn=vmap_render_fn,
        reset_display_fn=env_reset_display_fn,
        display_fn=env_stage_display_fn,
        evaluate_success_fn=lambda t: int(t.reward > .5),
        check_finished=lambda t: t.finished,
        state_cls=EnvStageState,
        max_episodes=max_episodes,
        min_success=min_success,
        metadata=metadata,
        custom_data_fn=custom_data_fn,
        duration=duration if not training else None,
        notify_success=True,
    )

def make_block(
    phase_1_text: str,
    phase_1_maze_name: str,
    phase_2_text: str,
    phase_2_cond1_maze_name: str,
    block_groups: np.ndarray,
    block_char2idx: dict,
    eval_duration: int,
    metadata: dict,
    min_success: int = None,
    max_episodes: int = None,
    phase_2_cond2_maze_name: str=None,
    phase_2_cond1_name: str = None,
    phase_2_cond2_name: str = None,
    make_env_kwargs: dict = None,
):
    def create_stage(name, body):
        return Stage(name=name, body=body, display_fn=stage_display_fn)
    
    make_env_kwargs = make_env_kwargs or {}
    def create_env_stage(name, maze_name, training, min_success, max_episodes, duration=None, **kwargs):
        return make_env_stage(
            name=name,
            maze_str=getattr(mazes, maze_name),
            min_success=min_success,
            max_episodes=max_episodes,
            duration=duration,
            groups=block_groups,
            char2idx=block_char2idx,
            training=training,
            **make_env_kwargs,
            **kwargs,
        )
    stages=[
            create_stage('Phase 1', phase_1_text),
            create_env_stage(
              name=phase_1_maze_name,
              maze_name=phase_1_maze_name,
              metadata=dict(maze=phase_1_maze_name, condition=0),
              training=True,
              min_success=min_success or min_success_train,
              max_episodes=max_episodes or max_episodes_train),
            create_stage('Phase 2', phase_2_text),
            create_env_stage(
              name=phase_2_cond1_name or phase_2_cond1_maze_name,
              maze_name=phase_2_cond1_maze_name,
              metadata=dict(maze=phase_2_cond1_maze_name, condition=1),
              training=False,
              min_success=1,
              max_episodes=1,
              duration=eval_duration),
      ]
    if phase_2_cond2_maze_name is not None:
        stages.append(
            create_env_stage(
              name=phase_2_cond2_name or phase_2_cond2_maze_name,
              maze_name=phase_2_cond2_maze_name,
              metadata=dict(maze=phase_2_cond2_maze_name, condition=2),
              training=False,
              min_success=1,
              max_episodes=1,
              duration=eval_duration))

    block = Block(
        metadata=dict(
            **metadata,
            groups=make_serializable(block_groups),
            char2idx=jax.tree_map(int, block_char2idx)
        ),
        stages=stages
    )
    return block


##############################################
# Create blocks
##############################################

if SAY_REUSE:
  instruct_text = f"""
          This experiment tests how effectively people can learn about goals before direct experience on them.

          It will consist of blocks with two phases each: **one** where you navigate to objects, and **another** where you navigate to other objects that you could have learned about previously.
  """
  def make_phase_2_text(time=30, include_time=True):
      time_str = f' of <span style="color: red; font-weight: bold;">{time}</span> seconds' if include_time else ""
      threshold = int(time*2/3)
      phase_2_text = f"""
      You will get a <span style="color: green; font-weight: bold;">bonus</span> if you complete the task in less than <span style="color: green; font-weight: bold;">{int(threshold)}</span> seconds. 
      
      You have a <span style="color: red; font-weight: bold;">time-limit</span>{time_str}. Try to reuse what you learned as best you can.

      If you retrieve the wrong object, the episode ends early. You have 1 try.

      """
      return phase_2_text
else:
  instruct_text = f"""
          This experiment tests how people learn to navigate maps.

          It will consist of blocks with two phases each: **one** where you navigate to objects, and **another** where you navigate to other objects.
  """
  def make_phase_2_text(time=30, include_time=True):
      time_str = f' of <span style="color: red; font-weight: bold;">{time}</span> seconds' if include_time else ""
      threshold = int(time*2/3)
      phase_2_text = f"""
      You will get a <span style="color: green; font-weight: bold;">bonus</span> if you complete the task in less than <span style="color: green; font-weight: bold;">{int(threshold)}</span> seconds.

      You have a <span style="color: red; font-weight: bold;">time-limit</span>{time_str}.

      If you retrieve the wrong object, the episode ends early. You have 1 try.
      """
      return phase_2_text

def make_phase_1_text():  
    phase_1_text = f"""
    Please learn to obtain these objects. You need to succeed {min_success_task} times per object.

    If you retrieve the wrong object, the episode ends early.
    """
    return phase_1_text



####################
# practice block
####################

block_groups, block_char2idx = permute_groups(block_groups)
practice_block = make_block(
    eval_duration=30,
    min_success=4 if not DEBUG else 1,
    max_episodes=10,
    make_env_kwargs=dict(force_room=True),
    phase_1_text=make_phase_1_text(),
    phase_1_maze_name='big_practice_maze',
    phase_2_text=make_phase_2_text(30, include_time=False),
    phase_2_cond1_maze_name='big_practice_maze',
    block_groups=block_groups,
    block_char2idx=block_char2idx,
    metadata=dict(manipulation=-1, desc="practice", long="practice")
)
####################
# (3) paths manipulation: reusing longer of two paths matching training path
####################
block_groups, block_char2idx = permute_groups(block_groups)
path_manipulation_block = make_block(
    phase_1_text=make_phase_1_text(),
    phase_1_maze_name='big_m3_maze1',
    phase_2_text=make_phase_2_text(),
    phase_2_cond1_maze_name='big_m3_maze1',
    phase_2_cond1_name='big_m3_maze1_eval',
    block_groups=block_groups,
    block_char2idx=block_char2idx,
    eval_duration=TIMER,
    make_env_kwargs=dict(custom_data_fn=manip3_data_fn),
    metadata=dict(
        manipulation=3,
        desc="reusing longer of two paths which matches training path",
        long=f"""
        Here there are two paths to the test object. We predict that people will take the path that was used to get to the training object.
        """)
)

####################
# (2) Start manipulation: Faster when on-path but further than off-path but closer
####################
block_groups, block_char2idx = permute_groups(block_groups)
start_manipulation_block = make_block(
    phase_1_text=make_phase_1_text(),
    phase_1_maze_name='big_m2_maze2',
    phase_2_text=make_phase_2_text(),
    phase_2_cond1_maze_name='big_m2_maze2_onpath',
    phase_2_cond2_maze_name='big_m2_maze2_offpath',
    block_groups=block_groups,
    block_char2idx=block_char2idx,
    eval_duration=30,
    metadata=dict(
        manipulation=2,
        desc="faster when on-path but further than off-path but closer",
        long=f"""
        In both tests, a shortcut is introduced. In the first, the agent is tested on the same path it trained on. In the second, the agent is tested on a different path.
        """)
)


####################
# (4) planning manipulation (short plan)
####################
block_groups, block_char2idx = permute_groups(block_groups)
plan_manipulation_block_short = make_block(
    # special case for short planning maze
    min_success=min_success_task,
    max_episodes=30,
    eval_duration=5,
    make_env_kwargs=dict(force_room=True),
    # regular commands
    phase_1_text=make_phase_1_text(),
    phase_1_maze_name='big_m4_maze_short',
    phase_2_text=make_phase_2_text(time=5),
    phase_2_cond1_maze_name='big_m4_maze_short_eval_same',
    phase_2_cond2_maze_name='big_m4_maze_short_eval_diff',
    block_groups=block_groups,
    block_char2idx=block_char2idx,
    metadata=dict(
        manipulation=4,
        desc="See if faster off train path than planning (short)",
        long=f"""
        Here there are two branches from a training path. We predict that people will have a shorter response time when an object is in the same location it was in phase 1.
        """)
)


####################
# (4) planning manipulation (long plan)
####################
block_groups, block_char2idx = permute_groups(block_groups)
plan_manipulation_block_long = make_block(
    # special case for long planning maze
    min_success=min_success_task,
    max_episodes=30,
    eval_duration=15,
    make_env_kwargs=dict(force_room=True),
    # regular commands
    phase_1_text=make_phase_1_text(),
    phase_1_maze_name='big_m4_maze_long',
    phase_2_text=make_phase_2_text(time=15),
    phase_2_cond1_maze_name='big_m4_maze_long_eval_same',
    phase_2_cond2_maze_name='big_m4_maze_long_eval_diff',
    block_groups=block_groups,
    block_char2idx=block_char2idx,
    metadata=dict(
        manipulation=4,
        desc="See if faster off train path than planning (long)",
        long=f"""
        Here there are two branches from a training path. We predict that people will have a shorter response time when an object is in the same location it was in phase 1.
        """)
)

##########################
# Feedback Block
##########################
feedback_block = Block(
  stages=[FeedbackStage(name='maze3_feedback', display_fn=feedback_display_fn)],
  metadata=dict(desc="feedback")
)
##########################
# Combining all together
##########################
manipulations = [
    path_manipulation_block,
    start_manipulation_block,
    plan_manipulation_block_short,
    plan_manipulation_block_long,
]
if NMAN > 0:
    manipulations = manipulations[:NMAN]
else:
    manipulations = manipulations

instruct_block = Block([
    Stage(
        name='Experiment instructions',
        body=instruct_text,
        display_fn=instruct_display_fn,
    ),
], metadata=dict(desc="instructions", long="instructions"))

all_blocks = []
if GIVE_INSTRUCTIONS:
    all_blocks.extend([instruct_block, practice_block])

all_blocks.extend(manipulations + [feedback_block])

all_stages = stages.prepare_blocks(all_blocks)

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
    randomized_blocks = list(all_blocks[offset:-1])
    random_order = jax.random.permutation(rng_key, len(randomized_blocks)) + offset
    block_order = jnp.concatenate([
        fixed_blocks,  # instruction blocks
        random_order,  # experiment blocks
        np.array([len(all_blocks)-1], dtype=np.int32),  # feedback block
    ]).astype(jnp.int32)
    block_order = block_order.tolist()

    stage_order = stages.generate_stage_order(all_blocks, block_order, rng_key)

    return block_order, stage_order
