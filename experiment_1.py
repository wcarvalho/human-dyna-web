from housemaze.human_dyna import env
from housemaze.human_dyna import utils
from housemaze.human_dyna import mazes

from fastwebrl.stages import ConsentStage, EnvStage

char2key, task_group_set, task_objects = mazes.get_group_set(3)
image_data = utils.load_image_dict()

task_runner = env.TaskRunner(task_objects=task_objects)
keys = image_data['keys']

jax_env = env.HouseMaze(
    task_runner=task_runner,
    num_categories=len(keys),
    use_done=True,
)
jax_env = utils.AutoResetWrapper(jax_env)

stages = [
  ConsentStage(name='consent'),

]