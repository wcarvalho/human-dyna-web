import sys
sys.path.append("simulations")
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
from nicewebrl.dataframe import DataFrame, concat_list
from simulations.craftax_web_env import CraftaxMultiGoalSymbolicWebEnvNoAutoReset, active_task_vectors
from simulations import craftax_simulation_configs
from simulations import craftax_experiment_configs
from nicewebrl import TimestepWrapper
from analysis import craftax_user_data

world_seed_to_idx = {}
for idx, config in enumerate(craftax_experiment_configs.PATHS_CONFIGS):
  world_seed_to_idx[config.world_seed] = idx

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

#image_data = utils.load_image_dict()
#image_keys = image_data["keys"]

#num_groups = 2
#char2idx, groups, task_objects = mazes.get_group_set(num_groups)
#task_objects = groups.reshape(-1)
#task_runner = multitask_env.TaskRunner(task_objects=task_objects)

#maze_to_manipulation = dict(
#  big_m1_maze3=1,
#  big_m1_maze3_shortcut=1,
#  big_m2_maze2=2,
#  big_m2_maze2_onpath=2,
#  big_m2_maze2_offpath=2,
#  big_m3_maze1=3,
#  big_m4_maze_short=4,
#  big_m4_maze_short_eval_same=4,
#  big_m4_maze_short_eval_diff=4,
#  big_m4_maze_short_blind=4,
#  big_m4_maze_short_eval_same_blind=4,
#  big_m4_maze_long=4,
#  big_m4_maze_long_eval_same=4,
#  big_m4_maze_long_eval_diff=4,
#  big_m4_maze_long_blind=4,
#  big_m4_maze_long_eval_same_blind=4,
#)

def load_env():

  env = CraftaxMultiGoalSymbolicWebEnvNoAutoReset()
  env = TimestepWrapper(env)
  return env


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
  model_filename: str = None
  model_name: str = None

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
  num_episodes: int = 1,
  max_steps: int = 600,
  path: Optional[str] = None,
  model_filename: Optional[str] = None,
  model_name: Optional[str] = None,
  overwrite: bool = True,
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
  agent, init_params, reset_fn = make_agent(
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
    params=agent_params if overwrite else init_params,
    target_network_params=agent_params,
    tx=make_optimizer(config),  # unnecessary
  )

  @jax.jit
  def eval_episode(rng, env_params):
    """Run a single evaluation episode"""
    rng, rng_ = jax.random.split(rng)
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
    transitions = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 0), transitions)

    return EpisodeData(
      timesteps=transitions.timestep,
      actions=transitions.action,
      transitions=transitions,
      positions=transitions.timestep.state.player_position,
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
    transitions = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 0), transitions)
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
    model_filename=model_filename,
    model_name=model_name or model_filename,
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
  base_path = path or f"{algorithm.path}/analysis/"
  os.makedirs(base_path, exist_ok=True)
  timesteps_filename = f"{base_path}/{algorithm.model_filename}_timesteps.pickle"
  df_filename = f"{base_path}/{algorithm.model_filename}_df.csv"

  extra_info = extra_info or {}
  rng = jax.random.PRNGKey(42)

  train_configs = craftax_simulation_configs.TRAIN_EVAL_CONFIGS
  test_configs = craftax_simulation_configs.TEST_CONFIGS

  ##############################
  # First try to load episodes
  ##############################
  all_episodes = []
  episode_configs = []  # Store config info for each episode

  if not overwrite_episodes and os.path.exists(timesteps_filename):
    with open(timesteps_filename, "rb") as f:
      print(f"{algorithm.model_name}: Loading from {timesteps_filename}")
      serialized_data = f.read()

      env_params = craftax_simulation_configs.make_multigoal_env_params(jax.tree_map(lambda x: x[:1], train_configs))
      example_episodes = algorithm.eval_fn(rng, env_params)
      example1 = jax.tree_util.tree_map(lambda x: x[0], example_episodes)

      # Two-step deserialization
      attempt1 = serialization.from_bytes(None, serialized_data)
      nepisodes = len(attempt1)
      all_episodes = serialization.from_bytes([example1] * nepisodes, serialized_data)
      print(f"{algorithm.model_name}: Loaded {nepisodes} episodes")
  else:
    # If episodes don't exist or we need to regenerate them
    def generate_episodes(configs, eval=False, nepisodes_per_eval=1):
      episodes_list = []
      configs_list = []
      nparams = configs.world_seed.shape[0]
      for i in range(nparams):
        env_params = craftax_simulation_configs.make_multigoal_env_params(jax.tree_map(lambda x: x[i:i+1], configs))
        episodes = algorithm.eval_fn(rng, env_params)
        nepisodes = nepisodes_per_eval
        # Split episodes
        task_config = jax.tree_map(lambda x: x[0], env_params.task_configs)
        for j in range(nepisodes):
          # minimize space requirements
          episode = jax.tree_util.tree_map(lambda x: x[j], episodes)
          in_episode = get_in_episode(episode.timesteps)
          episode = jax.tree_util.tree_map(lambda x: x[in_episode], episode)
          episodes_list.append(episode)
          configs_list.append((task_config, eval))
      return episodes_list, configs_list

    # Generate episodes from both train and test configs
    train_episodes, train_configs_info = generate_episodes(train_configs, eval=False)
    test_episodes, test_configs_info = generate_episodes(test_configs, eval=True)
    all_episodes = train_episodes + test_episodes
    episode_configs = train_configs_info + test_configs_info

    # Save serialized data
    with open(timesteps_filename, "wb") as f:
      serialized_data = serialization.to_bytes(all_episodes)
      f.write(serialized_data)
      print(f"Serialized {len(all_episodes)} episodes to {timesteps_filename}")

  ##############################
  # Next, try to load DataFrame
  ##############################
  df = None

  if not overwrite_df and os.path.exists(df_filename):
    df = pl.read_csv(df_filename)
    print(f"Loaded DataFrame from {df_filename}")
  else:
    # If we don't have episode_configs but we loaded episodes, we need to recreate configs
    if len(episode_configs) == 0:
      # Get count of training configurations
      train_count = len(train_configs.world_seed)
      test_count = len(test_configs.world_seed)
      
      # Recreate configurations for train episodes
      for i in range(min(train_count, len(all_episodes))):
        env_params = craftax_simulation_configs.make_multigoal_env_params(
            jax.tree_map(lambda x: x[i % train_count:i % train_count + 1], train_configs))
        task_config = jax.tree_map(lambda x: x[0], env_params.task_configs)
        episode_configs.append((task_config, False))
      
      # Recreate configurations for test episodes
      for i in range(max(0, len(all_episodes) - train_count)):
        test_idx = i % test_count
        env_params = craftax_simulation_configs.make_multigoal_env_params(
            jax.tree_map(lambda x: x[test_idx:test_idx + 1], test_configs))
        task_config = jax.tree_map(lambda x: x[0], env_params.task_configs)
        episode_configs.append((task_config, True))

    # Create info dictionary for each episode
    all_info = []
    def make_name(world_seed, eval):
      name = f"paths_{world_seed_to_idx[world_seed]}"
      if eval:
        name += "_eval1"
      else:
        name += "_training"
      return name

    for episode, (task_config, is_eval) in zip(all_episodes, episode_configs):
      name = make_name(int(task_config.world_seed), is_eval)

      info = dict(
        eval=is_eval,
        global_episode_idx=len(all_info),
        name=name,
        algo=algorithm.model_name or algorithm.model_filename,
        world_seed=int(task_config.world_seed),
        start_pos=str(task_config.start_position),
        task=int(task_config.goal_object),
        placed_goals=str(task_config.placed_goals),
        placed_achievements=str(task_config.placed_achievements),
        room=0,
        maze=int(task_config.world_seed),
        total_reward=float(total_reward(episode)),
        success=float(success(episode)),
        path_length=int(path_length(episode)),
        task_vector=str(episode.timesteps.observation.task_w[0]),
        **extra_info,
      )
      all_info.append(info)
    
    # Create and save DataFrame
    df = pl.DataFrame(all_info)
    _temp_df = DataFrame(df, all_episodes)
    _temp_df = craftax_user_data.add_reuse_columns(_temp_df, overlap_threshold=0.1)
    df = _temp_df._df
    df.write_csv(df_filename)

    print(f"Created new DataFrame with {len(all_info)} rows: {df_filename}")

  return df, all_episodes

def get_qlearning_data(
  paths: str,
  num_episodes: int = 1,
  max_steps: int = 100,
  overwrite_episodes: bool = False,
  overwrite_df: bool = False,
):
  from simulations import qlearning_craftax as qlearning

  paths_str = paths
  paths = glob(paths)
  if len(paths) == 0:
    raise ValueError(f"No paths found for {paths_str}")

  # Create environment once outside the loop
  env = load_env()
  dummy_env_params = craftax_simulation_configs.default_params
  model_df_list = []
  model_episodes_list = []

  for path in tqdm(paths):
    seed = path.split("/")[-1].split("=")[-1]
    agent_params, config = load_params_config(path, "qlearning")

    algorithm = load_algorithm(
      config=config,
      agent_params=agent_params,
      env=env,
      example_env_params=dummy_env_params,
      make_agent=qlearning.make_multigoal_craftax_agent,
      num_episodes=num_episodes,
      max_steps=max_steps,
      make_optimizer=qlearning.make_optimizer,
      make_actor=qlearning.make_actor,
      path=path,
      model_filename="qlearning",
      overwrite=overwrite_episodes,
    )

    df, episodes = get_algorithm_data(
      algorithm=algorithm,
      overwrite_episodes=overwrite_episodes,
      overwrite_df=overwrite_df,
      extra_info=dict(seed=int(seed)),
    )
    model_df_list.append(df)
    model_episodes_list.extend(episodes)

  return DataFrame(
    df=pl.concat(model_df_list, how="diagonal_relaxed"),
    episodes=model_episodes_list,
  )


def get_usfa_data(
  paths: str,
  num_episodes: int = 1,
  max_steps: int = 100,
  overwrite_episodes: bool = False,
  overwrite_df: bool = False,
  vis_coeff=0.1,
  **kwargs,
):

  paths_str = paths
  paths = glob(paths)
  if len(paths) == 0:
    raise ValueError(f"No paths found for {paths_str}")

  # Create environment once outside the loop
  env = load_env()
  dummy_env_params = craftax_simulation_configs.default_params

  model_df_list = []
  model_episodes_list = []

  from simulations import usfa_craftax as usfa
  for path in tqdm(paths):
    seed = path.split("/")[-1].split("=")[-1]
    agent_params, config = load_params_config(path, "usfa")

    algorithm = load_algorithm(
      config=config,
      agent_params=agent_params,
      env=env,
      example_env_params=dummy_env_params,
      make_agent=functools.partial(
        usfa.make_multigoal_craftax_agent,
        train_tasks=active_task_vectors,
      ),
      num_episodes=num_episodes,
      max_steps=max_steps,
      make_optimizer=usfa.make_optimizer,
      make_actor=functools.partial(usfa.make_actor, remove_gpi_dim=False),
      path=path,
      model_filename="usfa",
      overwrite=overwrite_episodes,
    )

    df, episodes = get_algorithm_data(
      algorithm=algorithm,
      overwrite_episodes=overwrite_episodes,
      overwrite_df=overwrite_df,
      extra_info=dict(seed=int(seed)),
      **kwargs,
    )
    model_df_list.append(df)
    model_episodes_list.extend(episodes)

  return DataFrame(
    df=pl.concat(model_df_list, how="diagonal_relaxed"),
    episodes=model_episodes_list,
  )


def get_preplay_data(
  paths: str,
  num_episodes: int = 1,
  max_steps: int = 100,
  overwrite_episodes: bool = False,
  overwrite_df: bool = False,
  **kwargs,
):
  """

  NOTE: since planning is only used for learning, we don't need to load an planning machinery.
  """

  from simulations import multitask_preplay_craftax_v2 as preplay

  paths_str = paths
  paths = glob(paths)
  if len(paths) == 0:
    raise ValueError(f"No paths found for {paths_str}")

  # Create environment once outside the loop
  env = load_env()
  dummy_env_params = craftax_simulation_configs.default_params
  model_df_list = []
  model_episodes_list = []

  for path in tqdm(paths):
    seed = path.split("/")[-1].split("=")[-1]
    agent_params, config = load_params_config(path, "preplay")

    algorithm = load_algorithm(
      config=config,
      agent_params=agent_params,
      env=env,
      example_env_params=dummy_env_params,
      make_agent=preplay.make_multigoal_agent,
      num_episodes=num_episodes,
      max_steps=max_steps,
      make_optimizer=preplay.make_optimizer,
      make_actor=preplay.make_actor,
      path=path,
      model_filename="preplay",
      model_name="preplay",
      overwrite=overwrite_episodes,
    )

    df, episodes = get_algorithm_data(
      algorithm=algorithm,
      overwrite_episodes=overwrite_episodes,
      overwrite_df=overwrite_df,
      extra_info=dict(seed=int(seed)),
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
  map_init = jax.tree_util.tree_map(lambda x: x[0], env_params.reset_params.map_init)
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
      return jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate((x, y), **kwargs), tree1, tree2
      )

    def add_time(v):
      return jax.tree_util.tree_map(lambda x: x[None], v)

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
    init_timestep = jax.tree_util.tree_map(jnp.asarray, init_timestep)
    timesteps = jax.tree_util.tree_map(jnp.asarray, timesteps)
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


def get_model_data(
  qlearning_path: str = None,
  sf_path: str = None,
  dyna_path: str = None,
  preplay_path: str = None,
  #search_path: str,
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
        print(f"Loaded cached model data from {episodes_cache_path}")
      return DataFrame(df=df, episodes=episodes)
    except Exception as e:
      print(f"Error loading cached model data: {e}")

  # If we need to regenerate the data:
  ##############################
  # Q-learning
  ##############################
  to_concat = []
  if qlearning_path is not None:
    qlearning_df = get_qlearning_data(
      qlearning_path, overwrite_episodes=overwrite_episodes, overwrite_df=overwrite_df
    )
    to_concat.append(qlearning_df)

  ##############################
  # Successor Features
  ##############################
  if sf_path is not None:
    sf_df = get_usfa_data(
      sf_path, overwrite_episodes=overwrite_episodes, overwrite_df=overwrite_df
    )
    to_concat.append(sf_df)
  ##############################
  # Dyna
  ##############################
  if dyna_path is not None:
    dyna_df = get_dyna_data(
      dyna_path,
      overwrite_episodes=overwrite_episodes,
      overwrite_df=overwrite_df
    )
    to_concat.append(dyna_df)
  ##############################
  # Multitask Preplay/Off-task Dyna
  ##############################
  if preplay_path is not None:
    preplay_df = get_preplay_data(
      preplay_path,
      overwrite_episodes=overwrite_episodes,
      overwrite_df=overwrite_df,
    )
    to_concat.append(preplay_df)
  ###############################
  ## Breadth-first search and Depth-first search
  ###############################
  #search_df = get_bfs_dfs_data(
  #  path=search_path,
  #  overwrite_episodes=overwrite_episodes,
  #  overwrite_df=overwrite_df,
  #  num_episodes=100,
  #)

  model_df = concat_list(*to_concat)

  # Cache the results
  os.makedirs(os.path.dirname(cache_base), exist_ok=True)
  model_df._df.write_csv(df_cache_path)
  with open(episodes_cache_path, "wb") as f:
    pickle.dump(model_df.episodes, f)
    print(f"Cached model data to {episodes_cache_path}")

  return model_df

if __name__ == "__main__":
  data_dir = "/Users/wilka/git/research/results/human_dyna/"
  from analysis import craftax_download_data
  ################
  # Load model data
  ################
  # %debug
  model_df = get_model_data(
    qlearning_path=f"{craftax_download_data.qlearning_local_dir}/seed=*",
    sf_path=f"{craftax_download_data.sf_local_dir}/seed=*",
    #dyna_path=f"{craftax_download_data.dyna_local_dir}/seed=*",
    preplay_path=f"{craftax_download_data.preplay_local_dir}/seed=*",
    # search_path=f"{data_dir}/search_algos",
    overwrite_episodes=True,
    overwrite_df=True,
    cache_dir='craftax_cache/model_data_cache'
  )