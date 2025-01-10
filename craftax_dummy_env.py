"""
A dummy version of the Craftax environment just for debugging.
"""

import jax
from enum import Enum
from typing import Tuple
from flax import struct
import jax.numpy as jnp
from gymnax.environments import spaces, environment


class Action(Enum):
  NOOP = 0  #
  LEFT = 1  # a
  RIGHT = 2  # d
  UP = 3  # w
  DOWN = 4  # s
  DO = 5  # space
  SLEEP = 6  # tab
  PLACE_STONE = 7  # r
  PLACE_TABLE = 8  # t
  PLACE_FURNACE = 9  # f
  PLACE_PLANT = 10  # p
  MAKE_WOOD_PICKAXE = 11  # 1
  MAKE_STONE_PICKAXE = 12  # 2
  MAKE_IRON_PICKAXE = 13  # 3
  MAKE_WOOD_SWORD = 14  # 5
  MAKE_STONE_SWORD = 15  # 6
  MAKE_IRON_SWORD = 16  # 7
  REST = 17  # e
  DESCEND = 18  # >
  ASCEND = 19  # <
  MAKE_DIAMOND_PICKAXE = 20  # 4
  MAKE_DIAMOND_SWORD = 21  # 8
  MAKE_IRON_ARMOUR = 22  # y
  MAKE_DIAMOND_ARMOUR = 23  # u
  SHOOT_ARROW = 24  # i
  MAKE_ARROW = 25  # o
  CAST_FIREBALL = 26  # g
  CAST_ICEBALL = 27  # h
  PLACE_TORCH = 28  # j
  DRINK_POTION_RED = 29  # z
  DRINK_POTION_GREEN = 30  # x
  DRINK_POTION_BLUE = 31  # c
  DRINK_POTION_PINK = 32  # v
  DRINK_POTION_CYAN = 33  # b
  DRINK_POTION_YELLOW = 34  # n
  READ_BOOK = 35  # m
  ENCHANT_SWORD = 36  # k
  ENCHANT_ARMOUR = 37  # l
  MAKE_TORCH = 38  # [
  LEVEL_UP_DEXTERITY = 39  # ]
  LEVEL_UP_STRENGTH = 40  # -
  LEVEL_UP_INTELLIGENCE = 41  # =
  ENCHANT_BOW = 42  # ;


# ACHIEVEMENTS
class Achievement(Enum):
  COLLECT_WOOD = 0
  PLACE_TABLE = 1
  EAT_COW = 2
  COLLECT_SAPLING = 3
  COLLECT_DRINK = 4
  MAKE_WOOD_PICKAXE = 5
  MAKE_WOOD_SWORD = 6
  PLACE_PLANT = 7
  DEFEAT_ZOMBIE = 8
  COLLECT_STONE = 9
  PLACE_STONE = 10
  EAT_PLANT = 11
  DEFEAT_SKELETON = 12
  MAKE_STONE_PICKAXE = 13
  MAKE_STONE_SWORD = 14
  WAKE_UP = 15
  PLACE_FURNACE = 16
  COLLECT_COAL = 17
  COLLECT_IRON = 18
  COLLECT_DIAMOND = 19
  MAKE_IRON_PICKAXE = 20
  MAKE_IRON_SWORD = 21

  MAKE_ARROW = 22
  MAKE_TORCH = 23
  PLACE_TORCH = 24

  COLLECT_SAPPHIRE = 54
  COLLECT_RUBY = 59
  MAKE_DIAMOND_PICKAXE = 60
  MAKE_DIAMOND_SWORD = 25
  MAKE_IRON_ARMOUR = 26
  MAKE_DIAMOND_ARMOUR = 27

  ENTER_GNOMISH_MINES = 28
  ENTER_DUNGEON = 29
  ENTER_SEWERS = 30
  ENTER_VAULT = 31
  ENTER_TROLL_MINES = 32
  ENTER_FIRE_REALM = 33
  ENTER_ICE_REALM = 34
  ENTER_GRAVEYARD = 35

  DEFEAT_GNOME_WARRIOR = 36
  DEFEAT_GNOME_ARCHER = 37
  DEFEAT_ORC_SOLIDER = 38
  DEFEAT_ORC_MAGE = 39
  DEFEAT_LIZARD = 40
  DEFEAT_KOBOLD = 41
  DEFEAT_KNIGHT = 65
  DEFEAT_ARCHER = 66
  DEFEAT_TROLL = 42
  DEFEAT_DEEP_THING = 43
  DEFEAT_PIGMAN = 44
  DEFEAT_FIRE_ELEMENTAL = 45
  DEFEAT_FROST_TROLL = 46
  DEFEAT_ICE_ELEMENTAL = 47
  DAMAGE_NECROMANCER = 48
  DEFEAT_NECROMANCER = 49

  EAT_BAT = 50
  EAT_SNAIL = 51

  FIND_BOW = 52
  FIRE_BOW = 53

  LEARN_FIREBALL = 55
  CAST_FIREBALL = 56
  LEARN_ICEBALL = 57
  CAST_ICEBALL = 58

  OPEN_CHEST = 61
  DRINK_POTION = 62
  ENCHANT_SWORD = 63
  ENCHANT_ARMOUR = 64


@struct.dataclass
class EnvParams:
  max_timesteps: int = 100000
  day_length: int = 300
  always_diamond: bool = False
  mob_despawn_distance: int = 14
  max_attribute: int = 5
  god_mode: bool = False
  world_seeds: Tuple[int, ...] = tuple()
  possible_goals: Tuple[int, ...] = tuple()
  active_goals: Tuple[int, ...] = tuple()
  current_goal: int = 0
  fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)


@struct.dataclass
class StaticEnvParams:
  map_size: Tuple[int, int] = (48, 48)
  num_levels: int = 9
  max_melee_mobs: int = 3
  max_passive_mobs: int = 3
  max_growing_plants: int = 10
  max_ranged_mobs: int = 2
  max_mob_projectiles: int = 3
  max_player_projectiles: int = 3
  use_precondition: bool = False
  initial_crafting_tables: bool = True
  initial_strength: int = 20


@struct.dataclass
class EnvState:
  # Basic map state
  map: jnp.ndarray
  item_map: jnp.ndarray
  mob_map: jnp.ndarray
  light_map: jnp.ndarray
  down_ladders: jnp.ndarray
  up_ladders: jnp.ndarray
  chests_opened: jnp.ndarray
  monsters_killed: jnp.ndarray

  # Player state
  player_position: jnp.ndarray
  player_level: int
  player_direction: int
  player_health: float
  player_food: int
  player_drink: int
  player_energy: int
  player_mana: int
  is_sleeping: bool
  is_resting: bool

  # Player attributes
  player_recover: float
  player_hunger: float
  player_thirst: float
  player_fatigue: float
  player_recover_mana: float
  player_xp: int
  player_dexterity: int
  player_strength: int
  player_intelligence: int

  # Inventory and achievements
  inventory: dict  # Simplified from original Inventory class
  achievements: jnp.ndarray

  # Mob states
  melee_mobs: dict  # Simplified from original Mobs class
  passive_mobs: dict
  ranged_mobs: dict
  mob_projectiles: dict
  mob_projectile_directions: jnp.ndarray
  player_projectiles: dict
  player_projectile_directions: jnp.ndarray

  # Plant states
  growing_plants_positions: jnp.ndarray
  growing_plants_age: jnp.ndarray
  growing_plants_mask: jnp.ndarray

  # Other game state
  potion_mapping: jnp.ndarray
  learned_spells: jnp.ndarray
  sword_enchantment: int
  bow_enchantment: int
  armour_enchantments: jnp.ndarray
  boss_progress: int
  boss_timesteps_to_spawn_this_round: int
  light_level: float
  state_rng: jnp.ndarray
  timestep: int
  current_goal: int
  fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)


class CraftaxSymbolicWebEnvNoAutoReset(environment.Environment):
  """
  A dummy version of the Craftax environment with empty core functions.
  """

  def __init__(self, static_env_params: StaticEnvParams = None):
    super().__init__()
    if static_env_params is None:
      static_env_params = self.default_static_params()
    self.static_env_params = static_env_params

  @property
  def default_params(self) -> EnvParams:
    return EnvParams()

  @staticmethod
  def default_static_params() -> StaticEnvParams:
    return StaticEnvParams()

  def step_env(
    self, key: jnp.ndarray, state: EnvState, action: int, params: EnvParams
  ) -> Tuple[jnp.ndarray, EnvState, float, bool, dict]:
    """Empty step function to be implemented"""
    # Return dummy values
    reward = 0.0
    done = False
    info = {}
    obs = self.get_obs(state, params)
    return obs, state, reward, done, info

  def reset_env(
    self, key: jnp.ndarray, params: EnvParams
  ) -> Tuple[jnp.ndarray, EnvState]:
    """Empty reset function to be implemented"""
    static_params = self.static_env_params
    state = EnvState(
      map=jnp.zeros((1)),
      item_map=jnp.zeros((1)),
      mob_map=jnp.zeros((1)),
      light_map=jnp.zeros((1)),
      down_ladders=jnp.zeros((1)),
      up_ladders=jnp.zeros((1)),
      chests_opened=jnp.zeros((1)),
      monsters_killed=jnp.zeros((1)),
      player_position=jnp.zeros((1)),
      player_direction=jnp.asarray(0, dtype=jnp.int32),
      player_level=jnp.asarray(0, dtype=jnp.int32),
      player_health=jnp.asarray(9.0, dtype=jnp.float32),
      player_food=jnp.asarray(9, dtype=jnp.int32),
      player_drink=jnp.asarray(9, dtype=jnp.int32),
      player_energy=jnp.asarray(9, dtype=jnp.int32),
      player_mana=jnp.asarray(9, dtype=jnp.int32),
      player_recover=jnp.asarray(0.0, dtype=jnp.float32),
      player_hunger=jnp.asarray(0.0, dtype=jnp.float32),
      player_thirst=jnp.asarray(0.0, dtype=jnp.float32),
      player_fatigue=jnp.asarray(0.0, dtype=jnp.float32),
      player_recover_mana=jnp.asarray(0.0, dtype=jnp.float32),
      is_sleeping=False,
      is_resting=False,
      player_xp=jnp.asarray(0, dtype=jnp.int32),
      player_dexterity=jnp.asarray(1, dtype=jnp.int32),
      # NOTE: MAIN DIFFERENCE: Humans start with 20 strength
      player_strength=jnp.asarray(static_params.initial_strength, dtype=jnp.int32),
      player_intelligence=jnp.asarray(1, dtype=jnp.int32),
      inventory=jnp.zeros((1)),
      sword_enchantment=jnp.asarray(0, dtype=jnp.int32),
      bow_enchantment=jnp.asarray(0, dtype=jnp.int32),
      armour_enchantments=jnp.array([0, 0, 0, 0], dtype=jnp.int32),
      melee_mobs=jnp.zeros((1)),
      ranged_mobs=jnp.zeros((1)),
      passive_mobs=jnp.zeros((1)),
      mob_projectiles=jnp.zeros((1)),
      mob_projectile_directions=jnp.zeros((1)),
      player_projectiles=jnp.zeros((1)),
      player_projectile_directions=jnp.zeros((1)),
      growing_plants_positions=jnp.zeros((1)),
      growing_plants_age=jnp.zeros((1)),
      growing_plants_mask=jnp.zeros((1)),
      potion_mapping=jnp.zeros((1)),
      learned_spells=jnp.array([False, False], dtype=bool),
      boss_progress=jnp.asarray(0, dtype=jnp.int32),
      boss_timesteps_to_spawn_this_round=jnp.asarray(0, dtype=jnp.int32),
      achievements=jnp.zeros((1)),
      light_level=jnp.asarray(0.0, dtype=jnp.float32),
      state_rng=jnp.zeros((1)),
      timestep=jnp.asarray(0, dtype=jnp.int32),
      current_goal=jnp.asarray(params.current_goal, dtype=jnp.int32),
    )
    obs = self.get_obs(state, params)
    return obs, state

  def get_obs(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
    """Returns a simple 8x8 image with a square in the middle"""
    image = jnp.full((40, 40, 3), params.world_seeds[0], dtype=jnp.uint8)
    jax.debug.print("world_seed: {seed}", seed=params.world_seeds[0])
    return image

  def observation_space(self, params: EnvParams) -> spaces.Box:
    """Define observation space"""
    return spaces.Box(0, 1, (1,))

  def action_space(self, params: EnvParams) -> spaces.Discrete:
    """Define action space"""
    return spaces.Discrete(5)  # Assuming 5 actions as in original

  @property
  def name(self) -> str:
    return "Craftax-Symbolic-NoAutoReset-v1"

  @property
  def num_actions(self) -> int:
    return 5  # Assuming 5 actions as in original
