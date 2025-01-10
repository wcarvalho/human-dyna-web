from typing import Optional, List, NamedTuple, Callable
import functools

from flax import struct
from flax.traverse_util import unflatten_dict
from gymnax.environments import environment
from glob import glob
import jax
import jax.numpy as jnp
from safetensors.flax import load_file
import numpy as np
import polars as pl
import pickle
import os
from flax import serialization
from flax.core import FrozenDict

from jaxneurorl.agents import value_based_basics as vbb
from housemaze.human_dyna import utils
from housemaze.human_dyna import multitask_env
from housemaze.human_dyna import web_env
from housemaze.human_dyna import mazes
from housemaze.human_dyna import experiments as housemaze_experiments

from nicewebrl.dataframe import DataFrame, concat_list

# for tqdm both in notebook and terminal
try:
  from IPython import get_ipython

  if "IPKernelApp" in get_ipython().config:
    from tqdm.notebook import tqdm

    try:
      import ipywidgets
    except:
      pass
  else:
    from tqdm import tqdm
except (ImportError, AttributeError):
  from tqdm import tqdm

image_data = utils.load_image_dict()
image_keys = image_data["keys"]

num_groups = 2
char2idx, groups, task_objects = mazes.get_group_set(num_groups)
task_objects = groups.reshape(-1)
task_runner = multitask_env.TaskRunner(task_objects=task_objects)

maze_to_manipulation = dict(
  big_m1_maze3=1,
  big_m1_maze3_shortcut=1,
  big_m2_maze2=2,
  big_m2_maze2_onpath=2,
  big_m2_maze2_offpath=2,
  big_m3_maze1=3,
  big_m4_maze_short=4,
  big_m4_maze_short_eval_same=4,
  big_m4_maze_short_eval_diff=4,
  big_m4_maze_short_blind=4,
  big_m4_maze_short_eval_same_blind=4,
  big_m4_maze_long=4,
  big_m4_maze_long_eval_same=4,
  big_m4_maze_long_eval_diff=4,
  big_m4_maze_long_blind=4,
  big_m4_maze_long_eval_same_blind=4,
)


class EpisodeData(NamedTuple):
  actions: jax.Array
  timesteps: struct.PyTreeNode  # housemaze.human_dyna.multitask_env.TimeStep
  positions: jax.Array = None
  reaction_times: jax.Array = None
  transitions: struct.PyTreeNode = None


def load_params_config(path: str, file: str, config: bool = True):
  filename = f"{path}/{file}.safetensors"
  flattened_dict = load_file(filename)
  params = unflatten_dict(flattened_dict, sep=",")

  if config:
    with open(f"{path}/{file}.config", "rb") as f:
      config = pickle.load(f)
  return params, config


def make_env_params(maze_str):
  return mazes.get_maze_reset_params(
    groups=groups,
    char2key=char2idx,
    maze_str=maze_str,
    randomize_agent=False,
    make_env_params=True,
  )


def load_env(num_categories: int = 200):
  task_runner = multitask_env.TaskRunner(task_objects=task_objects)
  base_env = web_env.HouseMaze(
    task_runner=task_runner,
    num_categories=num_categories,
  )
  env = utils.AutoResetWrapper(base_env)
  return env


def get_in_episode(timestep):
  # get mask for within episode
  non_terminal = timestep.discount
  is_last = timestep.last()
  term_cumsum = jnp.cumsum(is_last, -1)
  in_episode = (term_cumsum + non_terminal) < 2
  return in_episode


def success(e: EpisodeData):
  in_episode = get_in_episode(e.timesteps)
  rewards = e.timesteps.reward[in_episode]
  # return rewards
  assert rewards.ndim == 1, "this is only defined over vector, e.g. 1 episode"
  success = rewards > 0.5
  return success.any().astype(np.float32)


def path_length(e: EpisodeData):
  in_episode = get_in_episode(e.timesteps)
  return sum(in_episode)


def total_reward(e: EpisodeData):
  in_episode = get_in_episode(e.timesteps)
  return e.timesteps.reward[in_episode].sum()


###################
# Deep RL models
###################


def execute_trajectory(
  runner_state: vbb.RunnerState,
  num_steps: int,
  actor_step_fn: vbb.ActorStepFn,
  actions: jax.Array,
  env_step_fn: vbb.EnvStepFn,
  env_params: environment.EnvParams,
):
  def _env_step(state: vbb.RunnerState, action):
    """_summary_

    Buffer is updated with:
    - input agent state: s_{t-1}
    - agent obs input: x_t
    - agent prediction outputs: p_t
    - agent's action: a_t

    Args:
        rs (RunnerState): _description_
        unused (_type_): _description_

    Returns:
        _type_: _description_
    """
    # things that will be used/changed
    rng = state.rng
    prior_timestep = state.timestep
    prior_agent_state = state.agent_state

    # prepare rngs for actions and step
    rng, rng_a, rng_s = jax.random.split(rng, 3)

    preds, ingored_action, agent_state = actor_step_fn(
      state.train_state, prior_agent_state, prior_timestep, rng_a
    )

    transition = vbb.Transition(
      prior_timestep,
      action=action,
      extras=FrozenDict(preds=preds, agent_state=prior_agent_state),
    )

    # take step in env
    timestep = env_step_fn(rng_s, prior_timestep, action, env_params)

    state = state._replace(
      timestep=timestep,
      agent_state=agent_state,
      rng=rng,
    )

    return state, transition

  return jax.lax.scan(f=_env_step, init=runner_state, xs=actions, length=num_steps)


@struct.dataclass
class Algorithm:
  config: dict = None
  train_state: Callable = None
  actor: Callable = None
  network: Callable = None
  reset_fn: Callable = None
  eval_fn: Callable = None
  execute_fn: Callable = None
  path: str = None
  name: str = None

  def seed(self):
    return self.path.split("/")[-1]


def load_algorithm(
  config: dict,
  agent_params: dict,
  example_env_params: environment.EnvParams,
  env: environment.Environment,
  make_agent: vbb.MakeAgentFn,
  make_optimizer: vbb.MakeOptimizerFn,
  make_actor: vbb.MakeActorFn,
  num_episodes: int = 25,
  max_steps: int = 600,
  path: Optional[str] = None,
  name: Optional[str] = None,
):
  """Loads and evaluates a trained algorithm using the same interface as make_train.

  Args:
      config (dict): Configuration dictionary
      env (environment.Environment): Environment class
      make_agent (MakeAgentFn): Function to create agent
      make_optimizer (MakeOptimizerFn): Function to create optimizer
      make_actor (MakeActorFn): Function to create actor
      env_params (Optional[environment.EnvParams]): Test environment parameters
      tasks (Optional[List[int]]): List of tasks to evaluate. Defaults to [0].
      n_episodes (int): Number of episodes per task. Defaults to 1.
      path (Optional[str]): Path to saved model
      name (Optional[str]): Name of model file

  Returns:
      List[List[EpisodeData]]: Evaluation episodes for each task
  """

  config["NUM_ENVS"] = num_episodes

  def vmap_reset(rng, env_params):
    return jax.vmap(env.reset, in_axes=(0, None))(
      jax.random.split(rng, num_episodes), env_params
    )

  def vmap_step(rng, env_state, action, env_params):
    return jax.vmap(env.step, in_axes=(0, 0, 0, None))(
      jax.random.split(rng, num_episodes), env_state, action, env_params
    )

  # Initialize environment and agent
  rng = jax.random.PRNGKey(config["SEED"])
  rng, rng_ = jax.random.split(rng)
  example_timestep = vmap_reset(rng_, example_env_params)

  # Create agent
  agent, _, reset_fn = make_agent(
    config=config,
    env=env,
    env_params=example_env_params,
    example_timestep=example_timestep,
    rng=rng_,
  )

  # Create actor
  rng, rng_ = jax.random.split(rng)
  actor = make_actor(config=config, agent=agent, rng=rng_)

  # Initialize train state
  train_state = vbb.CustomTrainState.create(
    apply_fn=agent.apply,
    params=agent_params,
    target_network_params=agent_params,
    tx=make_optimizer(config),  # unnecessary
  )

  @jax.jit
  def eval_episode(rng, env_params, task_w):
    """Run a single evaluation episode"""
    rng, rng_ = jax.random.split(rng)
    env_params = env_params.replace(
      task_probs=task_w,
    )
    init_timestep = vmap_reset(rng=rng_, env_params=env_params)
    # Set task and initialize agent state
    agent_state = reset_fn(train_state.params, init_timestep, rng_)

    # Create runner state and collect trajectory
    runner_state = vbb.RunnerState(
      train_state=train_state,
      timestep=init_timestep,
      agent_state=agent_state,
      rng=rng,
    )

    _, transitions = vbb.collect_trajectory(
      runner_state=runner_state,
      num_steps=max_steps,
      actor_step_fn=actor.eval_step,
      env_step_fn=vmap_step,
      env_params=env_params,
    )

    # [T, N, ....] --> # [N, T, ....]
    transitions = jax.tree.map(lambda x: jnp.swapaxes(x, 1, 0), transitions)

    return EpisodeData(
      timesteps=transitions.timestep,
      actions=transitions.action,
      transitions=transitions,
      positions=transitions.timestep.state.agent_pos,
      reaction_times=None,
    )

  @jax.jit
  def execute_fn(init_timestep, env_params, actions):
    agent_state = reset_fn(train_state.params, init_timestep, rng_)

    # Create runner state and collect trajectory
    runner_state = vbb.RunnerState(
      train_state=train_state,
      timestep=init_timestep,
      agent_state=agent_state,
      rng=rng,
    )

    _, transitions = execute_trajectory(
      runner_state=runner_state,
      num_steps=max_steps,
      actor_step_fn=actor.eval_step,
      actions=actions,
      env_step_fn=vmap_step,
      env_params=env_params,
    )

    # [T, N, ....] --> # [N, T, ....]
    transitions = jax.tree.map(lambda x: jnp.swapaxes(x, 1, 0), transitions)
    return transitions

  return Algorithm(
    config=config,
    network=agent,
    reset_fn=reset_fn,
    actor=actor,
    train_state=train_state,
    eval_fn=jax.jit(eval_episode),
    execute_fn=jax.jit(execute_fn),
    path=path,
    name=name,
  )


def get_algorithm_data(
  algorithm: Algorithm,
  overwrite_episodes: bool = False,
  overwrite_df: bool = False,
  extra_info=None,
  data_task_runner=None,
  maze_names: Optional[List[str]] = None,
  path: str = None,
):
  extra_info = extra_info or {}
  _, _, _, label2name = housemaze_experiments.exp2(algorithm.config, analysis_eval=True)

  maze_names = maze_names or list(label2name.values())

  base_path = path or f"{algorithm.path}/analysis/"
  os.makedirs(base_path, exist_ok=True)
  timesteps_filename = f"{base_path}/{algorithm.name}_timesteps.pickle"
  df_filename = f"{base_path}/{algorithm.name}_df.csv"

  train_tasks = groups[:1, 0]
  test_tasks = groups[:1, 1]
  tasks = jnp.concatenate((train_tasks, test_tasks))
  rng = jax.random.PRNGKey(42)

  ##############################
  # First try to load episodes
  ##############################
  all_episodes = None
  if not overwrite_episodes and os.path.exists(timesteps_filename):
    with open(timesteps_filename, "rb") as f:
      print(f"{algorithm.name}: Loading from {timesteps_filename}")
      serialized_data = f.read()

      # Create template episode structure
      task_vector = data_task_runner.task_vector(groups[0, 0])
      maze_name = next(iter(label2name.values()))
      env_params = make_env_params(getattr(mazes, maze_name))
      example_episodes = algorithm.eval_fn(rng, env_params, task_vector)
      example1 = jax.tree.map(lambda x: x[0], example_episodes)

      # Two-step deserialization
      attempt1 = serialization.from_bytes(None, serialized_data)
      nepisodes = len(attempt1)
      all_episodes = serialization.from_bytes([example1] * nepisodes, serialized_data)
      print(f"{algorithm.name}: Loaded {nepisodes} episodes")

  ##############################
  # If no episodes, generate them
  ##############################
  if all_episodes is None:
    all_episodes = []
    for maze_name in tqdm(
      maze_names, desc=f"{algorithm.name}: Generating maze episodes"
    ):
      env_params = make_env_params(getattr(mazes, maze_name))
      for task in tasks:
        task_vector = data_task_runner.task_vector(task)
        episodes = algorithm.eval_fn(rng, env_params, task_vector)
        nepisodes = episodes.actions.shape[0]

        # Split episodes
        for i in range(nepisodes):
          # minimize space requirements
          episode = jax.tree.map(lambda x: x[i], episodes)
          in_episode = get_in_episode(episode.timesteps)
          episode = jax.tree.map(lambda x: x[in_episode], episode)
          all_episodes.append(episode)

    # Save serialized data
    with open(timesteps_filename, "wb") as f:
      serialized_data = serialization.to_bytes(all_episodes)
      f.write(serialized_data)

  ##############################
  # Try to load DataFrame
  ##############################
  df = None
  if not overwrite_df and os.path.exists(df_filename):
    df = pl.read_csv(df_filename)

  ##############################
  # If no DataFrame, compute metrics
  ##############################

  def went_to_junction(episode_data, junction=(0, 11)):
    positions = episode_data.timesteps.state.agent_pos
    match = jnp.array(junction) == positions
    match = (match).sum(-1) == 2  # both x and y matches
    return int(match.any())

  if df is None:
    # Calculate number of episodes per task/maze combination
    all_info = []
    episode_idx = 0
    for maze_name in tqdm(
      maze_names, desc=f"{algorithm.name}: Generating maze information"
    ):
      nepisodes = len(all_episodes) // len(label2name)
      for _ in range(nepisodes):
        episode = all_episodes[episode_idx]
        task = episode.timesteps.state.task_object[0]
        info = dict(
          eval=bool(task in test_tasks),
          algo=algorithm.name,
          room=0,
          task=task,
          maze=maze_name,
          total_reward=float(total_reward(episode)),
          success=float(success(episode)),
          path_length=int(path_length(episode)),
          task_vector=str(episode.timesteps.state.task_w[0]),
          manipulation=maze_to_manipulation.get(maze_name),
          **extra_info,
        )
        if maze_name == "big_m1_maze3_shortcut":
          info["reuse"] = int(went_to_junction(episode, (2, 14)))
        elif maze_name == "big_m3_maze1":
          info["reuse"] = int(went_to_junction(episode, (14, 25)))
        else:
          info["reuse"] = 2  # neither
        all_info.append(info)
        episode_idx += 1

    df = pl.DataFrame(all_info)
    df.write_csv(df_filename)

  return df, all_episodes


def get_qlearning_data(
  paths: str,
  num_episodes: int = 25,
  max_steps: int = 200,
  overwrite_episodes: bool = False,
  overwrite_df: bool = False,
):
  from simulations.networks import CategoricalHouzemazeObsEncoder
  from simulations import qlearning

  paths_str = paths
  paths = glob(paths)
  if len(paths) == 0:
    raise ValueError(f"No paths found for {paths_str}")

  # Create environment once outside the loop
  env = load_env()
  dummy_env_params = make_env_params(mazes.big_practice_maze)
  model_df_list = []
  model_episodes_list = []

  for path in tqdm(paths):
    seed = path.split("/")[-1].split("=")[-1]
    agent_params, config = load_params_config(path, "qlearning")

    HouzemazeObsEncoder = functools.partial(
      CategoricalHouzemazeObsEncoder,
      num_categories=10000,
      embed_hidden_dim=config["EMBED_HIDDEN_DIM"],
      mlp_hidden_dim=config["MLP_HIDDEN_DIM"],
      num_embed_layers=config["NUM_EMBED_LAYERS"],
      num_mlp_layers=config["NUM_MLP_LAYERS"],
      activation=config["ACTIVATION"],
      norm_type=config.get("NORM_TYPE", "none"),
    )

    algorithm = load_algorithm(
      config=config,
      agent_params=agent_params,
      env=env,
      example_env_params=dummy_env_params,
      make_agent=functools.partial(
        qlearning.make_agent,
        ObsEncoderCls=HouzemazeObsEncoder,
      ),
      num_episodes=num_episodes,
      max_steps=max_steps,
      make_optimizer=qlearning.make_optimizer,
      make_actor=qlearning.make_actor,
      path=path,
      name="qlearning",
    )

    df, episodes = get_algorithm_data(
      algorithm=algorithm,
      overwrite_episodes=overwrite_episodes,
      overwrite_df=overwrite_df,
      extra_info=dict(seed=int(seed)),
      data_task_runner=task_runner,
    )
    model_df_list.append(df)
    model_episodes_list.extend(episodes)

  return DataFrame(
    df=pl.concat(model_df_list, how="diagonal_relaxed"),
    episodes=model_episodes_list,
  )


def get_usfa_data(
  paths: str,
  num_episodes: int = 25,
  max_steps: int = 200,
  overwrite_episodes: bool = False,
  overwrite_df: bool = False,
  vis_coeff=0.1,
  **kwargs,
):
  from housemaze.human_dyna import sf_task_runner
  from simulations.networks import CategoricalHouzemazeObsEncoder
  from simulations import usfa

  paths_str = paths
  paths = glob(paths)
  if len(paths) == 0:
    raise ValueError(f"No paths found for {paths_str}")

  dummy_env_params = make_env_params(mazes.big_practice_maze)
  # first map (all same objects)
  train_objects = dummy_env_params.reset_params.train_objects[0]

  eval_task_runner_sf = sf_task_runner.TaskRunner(
    task_objects=task_objects, vis_coeff=vis_coeff, radius=5
  )
  train_tasks = jnp.array([eval_task_runner_sf.task_vector(o) for o in train_objects])

  ###################
  # Get environment
  ###################
  # Create environment once outside the loop

  base_env = web_env.HouseMaze(
    task_runner=eval_task_runner_sf,
    num_categories=200,
  )
  env = utils.AutoResetWrapper(base_env)

  model_df_list = []
  model_episodes_list = []

  for path in tqdm(paths):
    seed = path.split("/")[-1].split("=")[-1]
    agent_params, config = load_params_config(path, "usfa")

    HouzemazeObsEncoder = functools.partial(
      CategoricalHouzemazeObsEncoder,
      num_categories=10000,
      embed_hidden_dim=config["EMBED_HIDDEN_DIM"],
      mlp_hidden_dim=config["MLP_HIDDEN_DIM"],
      num_embed_layers=config["NUM_EMBED_LAYERS"],
      num_mlp_layers=config["NUM_MLP_LAYERS"],
      activation=config["ACTIVATION"],
      norm_type=config.get("NORM_TYPE", "none"),
    )

    algorithm = load_algorithm(
      config=config,
      agent_params=agent_params,
      env=env,
      example_env_params=dummy_env_params,
      make_agent=functools.partial(
        usfa.make_agent,
        train_tasks=train_tasks,
        ObsEncoderCls=HouzemazeObsEncoder,
      ),
      num_episodes=num_episodes,
      max_steps=max_steps,
      make_optimizer=usfa.make_optimizer,
      make_actor=functools.partial(usfa.make_actor, remove_gpi_dim=False),
      path=path,
      name="usfa",
    )

    df, episodes = get_algorithm_data(
      algorithm=algorithm,
      overwrite_episodes=overwrite_episodes,
      overwrite_df=overwrite_df,
      extra_info=dict(seed=int(seed)),
      data_task_runner=eval_task_runner_sf,
      **kwargs,
    )
    model_df_list.append(df)
    model_episodes_list.extend(episodes)

  return DataFrame(
    df=pl.concat(model_df_list, how="diagonal_relaxed"),
    episodes=model_episodes_list,
  )


def get_dyna_data(
  paths: str,
  num_episodes: int = 25,
  max_steps: int = 200,
  overwrite_episodes: bool = False,
  overwrite_df: bool = False,
  **kwargs,
):
  """

  NOTE: since planning is only used for learning, we don't need to load an planning machinery.
  """

  from simulations.networks import CategoricalHouzemazeObsEncoder
  from simulations import offtask_dyna

  paths_str = paths
  paths = glob(paths)
  if len(paths) == 0:
    raise ValueError(f"No paths found for {paths_str}")

  # Create environment once outside the loop
  env = load_env()
  dummy_env_params = make_env_params(mazes.big_practice_maze)
  model_df_list = []
  model_episodes_list = []

  for path in tqdm(paths):
    seed = path.split("/")[-1].split("=")[-1]
    agent_params, config = load_params_config(path, "dynaq_shared")

    HouzemazeObsEncoder = functools.partial(
      CategoricalHouzemazeObsEncoder,
      num_categories=10000,
      embed_hidden_dim=config["EMBED_HIDDEN_DIM"],
      mlp_hidden_dim=config["MLP_HIDDEN_DIM"],
      num_embed_layers=config["NUM_EMBED_LAYERS"],
      num_mlp_layers=config["NUM_MLP_LAYERS"],
      activation=config["ACTIVATION"],
      norm_type=config.get("NORM_TYPE", "none"),
    )

    algorithm = load_algorithm(
      config=config,
      agent_params=agent_params,
      env=env,
      example_env_params=dummy_env_params,
      make_agent=functools.partial(
        offtask_dyna.make_agent,
        ObsEncoderCls=HouzemazeObsEncoder,
      ),
      num_episodes=num_episodes,
      max_steps=max_steps,
      make_optimizer=offtask_dyna.make_optimizer,
      make_actor=offtask_dyna.make_actor,
      path=path,
      name="dynaq_shared",
    )

    df, episodes = get_algorithm_data(
      algorithm=algorithm,
      overwrite_episodes=overwrite_episodes,
      overwrite_df=overwrite_df,
      extra_info=dict(seed=int(seed)),
      data_task_runner=task_runner,
      **kwargs,
    )
    model_df_list.append(df)
    model_episodes_list.extend(episodes)

  return DataFrame(
    df=pl.concat(model_df_list, how="diagonal_relaxed"),
    episodes=model_episodes_list,
  )


###################
# Search Algorithms
###################


def actions_from_search(env_params, rng, task, algo, budget):
  map_init = jax.tree.map(lambda x: x[0], env_params.reset_params.map_init)
  grid = np.asarray(map_init.grid)
  agent_pos = tuple(int(o) for o in map_init.agent_pos)
  goal = np.array([task])
  path, _ = algo(grid, agent_pos, goal, key=rng, budget=budget)
  actions = utils.actions_from_path(path)
  return actions


def collect_search_episodes(
  env, env_params, task_vector, algorithm: str, rng, budget=None, n: int = 100
):
  budget = budget or 1e8
  env_params = env_params.replace(
    task_probs=task_vector,
  )
  default_init_timestep = env.reset(rng, env_params)
  task = default_init_timestep.state.task_object

  @jax.jit
  def concat_first_rest(first, rest):
    """concat first pytree with sequence of pytrees
    Args:
        first (struct.PyTree): [...]
        rest (struct.PyTree): [T, ...]

    Returns:
        struct.PyTree: [T+1, ...]
    """

    def concat_pytrees(tree1, tree2, **kwargs):
      return jax.tree.map(lambda x, y: jnp.concatenate((x, y), **kwargs), tree1, tree2)

    def add_time(v):
      return jax.tree.map(lambda x: x[None], v)

    return concat_pytrees(add_time(first), rest)

  @jax.jit
  def step_fn(carry, action):
    rng, timestep = carry
    rng, step_rng = jax.random.split(rng)
    next_timestep = env.step(step_rng, timestep, action, env_params)
    return (rng, next_timestep), next_timestep

  @jax.jit
  def collect_episode(actions, rng):
    init_timestep = env.reset(rng, env_params)
    initial_carry = (rng, init_timestep)
    (rng, _), timesteps = jax.lax.scan(step_fn, initial_carry, actions)
    init_timestep = jax.tree.map(jnp.asarray, init_timestep)
    timesteps = jax.tree.map(jnp.asarray, timesteps)
    return concat_first_rest(init_timestep, timesteps)

  #######################
  # first get actions from n different runs
  #######################
  all_actions = []
  rngs = jax.random.split(rng, n)

  # First, get all actions
  for idx in tqdm(range(n), f"{algorithm}: planning"):
    actions = actions_from_search(
      env_params, rngs[idx], task, algo=getattr(utils, algorithm), budget=budget
    )
    all_actions.append(actions)

  # Find the maximum length among all action sequences
  max_length = max(len(actions) for actions in all_actions)

  # Pad each action sequence to the maximum length
  padded_actions = []
  for actions in all_actions:
    padding = [0] * (max_length - len(actions))
    padded_actions.append(np.concatenate((actions, np.array(padding, dtype=np.int32))))

  # Convert to numpy array
  all_actions = np.array(padded_actions, dtype=np.int32)

  # Now compute all episodes
  # Vectorize collect_episode over batch dimension
  vmapped_collect_episode = jax.jit(jax.vmap(collect_episode))
  all_episodes = vmapped_collect_episode(all_actions[:, :-1], rngs)

  return EpisodeData(timesteps=all_episodes, actions=all_actions)


def get_bfs_dfs_data(
  path: str,
  overwrite_episodes: bool = False,
  overwrite_df: bool = False,
  budget=None,
  num_episodes: int = 100,
  **kwargs,
):
  env = load_env()

  model_df_list = []
  model_episodes_list = []

  for algorithm in ["bfs", "dfs"]:

    def eval_fn(rng, env_params, task_vector):
      return collect_search_episodes(
        env=env,
        env_params=env_params,
        task_vector=task_vector,
        algorithm=algorithm,
        rng=rng,
        budget=budget,
        n=num_episodes,
      )

    df, episodes = get_algorithm_data(
      algorithm=Algorithm(
        config={},
        name=algorithm,
        eval_fn=eval_fn,
        path=path,
      ),
      overwrite_episodes=overwrite_episodes,
      overwrite_df=overwrite_df,
      data_task_runner=task_runner,
      **kwargs,
    )
    model_df_list.append(df)
    model_episodes_list.extend(episodes)

  return DataFrame(
    df=pl.concat(model_df_list, how="diagonal_relaxed"),
    episodes=model_episodes_list,
  )


def get_model_data(
  qlearning_path: str,
  sf_path: str,
  dyna_path: str,
  search_path: str,
  overwrite_episodes: bool = False,
  overwrite_df: bool = False,
  cache_dir: str = None,
):
  """Load and process data from different model types.

  NOTE: here, we use pickle since different models use different EpisodeData structures
    internal function deserialize in a more portable way with
    flax.serialization.from_bytes

  Args:
      qlearning_path: Path to Q-learning model data
      sf_path: Path to Successor Features model data
      dyna_path: Path to Dyna model data
      search_path: Path to search algorithms data
      overwrite_episodes: If True, regenerate episode data even if it exists
      overwrite_df: If True, regenerate DataFrame even if it exists

  Returns:
      DataFrame containing combined model data
  """
  # Create cache filenames based on the paths
  cache_dir = cache_dir or os.path.dirname(qlearning_path)
  cache_base = os.path.join(cache_dir, "model_data_cache")
  df_cache_path = f"{cache_base}_df.csv"
  episodes_cache_path = f"{cache_base}_episodes.pickle"

  # Try to load cached data if not overwriting
  if (
    not (overwrite_episodes or overwrite_df)
    and os.path.exists(df_cache_path)
    and os.path.exists(episodes_cache_path)
  ):
    try:
      df = pl.read_csv(df_cache_path)
      with open(episodes_cache_path, "rb") as f:
        episodes = pickle.load(f)
      return DataFrame(df=df, episodes=episodes)
    except Exception as e:
      print(f"Error loading cached model data: {e}")

  # If we need to regenerate the data:
  ##############################
  # Q-learning
  ##############################
  qlearning_df = get_qlearning_data(
    qlearning_path, overwrite_episodes=overwrite_episodes, overwrite_df=overwrite_df
  )

  ##############################
  # Successor Features
  ##############################
  sf_df = get_usfa_data(
    sf_path, overwrite_episodes=overwrite_episodes, overwrite_df=overwrite_df
  )

  ##############################
  # Multitask Preplay/Off-task Dyna
  ##############################
  dyna_df = get_dyna_data(
    dyna_path,
    overwrite_episodes=overwrite_episodes,
    overwrite_df=overwrite_df,
  )

  ##############################
  # Breadth-first search and Depth-first search
  ##############################
  search_df = get_bfs_dfs_data(
    path=search_path,
    overwrite_episodes=overwrite_episodes,
    overwrite_df=overwrite_df,
    num_episodes=100,
  )

  model_df = concat_list(qlearning_df, sf_df, dyna_df, search_df)

  # Cache the results
  os.makedirs(os.path.dirname(cache_base), exist_ok=True)
  model_df._df.write_csv(df_cache_path)
  with open(episodes_cache_path, "wb") as f:
    pickle.dump(model_df.episodes, f)
    print(f"Cached model data to {episodes_cache_path}")

  return model_df


if __name__ == "__main__":
  data_dir = "/Users/wilka/git/research/results/human_dyna"
  get_model_data(
    qlearning_path=f"{data_dir}/model_data/ql/save_data/ql-big-2/tota=40000000,exp=exp2/seed=*",
    sf_path=f"{data_dir}/model_data/usfa/save_data/usfa-big-10-search/sf_h=1024,num_=2,tota=40000000,exp=exp2/seed=*",
    dyna_path=f"{data_dir}/model_data/dynaq_shared/save_data/dynaq-big-4/alg=dynaq_shared,agen=256,tota=100000000,exp=exp2/seed=*",
    search_path=f"{data_dir}/search_algos",
    overwrite_episodes=True,
    overwrite_df=True,
    cache_dir=f"{data_dir}/model_data/cache",
  )
