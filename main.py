
from dotenv import load_dotenv
import json
from nicegui import app, ui
import nicewebrl.nicejax
import nicewebrl.stages
from tortoise import Tortoise
from tortoise.contrib.pydantic import pydantic_model_creator
import os

##import experiment_1s
import nicewebrl
from nicewebrl.stages import ExperimentData


load_dotenv()

APP_TITLE = 'Human Dyna Test'
DATABASE_FILE = os.environ.get('DB_FILE', 'db.sqlite')
DEBUG = int(os.environ.get('DEBUG', 0))
EXPERIMENT = int(os.environ.get('EXP', 1))
if EXPERIMENT == 0:
  import experiment_test as experiment
elif EXPERIMENT == 1:
  import experiment_1 as experiment
else:
   raise NotImplementedError
   

DATABASE_FILE = f'{DATABASE_FILE}_exp{EXPERIMENT}'
stages = experiment.stages
#####################################
# Consent Form
#####################################


def make_consent_form(
    meta_container, stage_container, button_container
):
  ui.markdown('## Consent Form')
  ui.markdown('Please agree to this experiment.')
  ui.checkbox(
    'I agree to participate.',
    on_change=lambda: start_experiment(
       meta_container, stage_container, button_container))

#####################################
# Start/load experiment
#####################################

async def start_experiment(
      meta_container,
      stage_container,
      button_container):
  app.storage.user['experiment_started'] = True

  if app.storage.user.get('experiment_finished', False):
    await finish_experiment(
       meta_container, stage_container, button_container)
    return

  nicewebrl.get_user_session_minutes()
  meta_container.clear()
  ui.on('key_pressed', 
        lambda e: handle_key_press(e, meta_container, stage_container, button_container))
  await load_stage(meta_container, stage_container, button_container)

async def handle_key_press(e, meta_container, stage_container, button_container):
  stage = stages[app.storage.user['stage_idx']]
  await stage.handle_key_press(e, stage_container)
  if stage.get_user_data('finished', False):
    app.storage.user['stage_idx'] += 1
    await load_stage(meta_container, stage_container, button_container)

async def handle_button_press(*args, **kwargs):
  stage = stages[app.storage.user['stage_idx']]
  await stage.handle_button_press()
  if stage.get_user_data('finished', False):
    app.storage.user['stage_idx'] += 1
    await load_stage(*args, **kwargs)


async def load_stage(meta_container, stage_container, button_container):
    """Default behavior for progressing through stages."""
    if app.storage.user['stage_idx'] >= len(stages):
        await finish_experiment(meta_container, stage_container, button_container)
        return

    stage = stages[app.storage.user['stage_idx']]
    with stage_container.style('align-items: center;'):
      await stage.activate(stage_container)

    with button_container.style('align-items: center;'):
      button_container.clear()
      button = ui.button('Next page').bind_visibility_from(stage, 'next_button')
      await button.clicked()
      await handle_button_press(meta_container, stage_container, button_container)


async def finish_experiment(meta_container, stage_container, button_container):
    meta_container.clear()
    stage_container.clear()
    button_container.clear()

    app.storage.user['experiment_finished'] = True
    app.storage.user['data_saved'] = app.storage.user.get(
        'data_saved', False)
    if not app.storage.user['data_saved']:
      with meta_container:
        meta_container.clear()
        ui.markdown(f"# Saving data. Please wait")
      await save_data()
      app.storage.user['data_saved'] = True

    with meta_container:
        meta_container.clear()
        ui.markdown("# Experiment over")
        ui.markdown("## Data saved")
        ui.markdown("### You may close the browser")


async def save_data():
    # Create a Pydantic model from your Tortoise model
    ExperimentDataPydantic = pydantic_model_creator(ExperimentData)
    ExperimentDataPydantic.model_config['from_attributes'] = True

    user_experiment_data = await ExperimentData.filter(
        session_id=app.storage.browser['id']).all()

    data_dicts = [ExperimentDataPydantic.model_validate(
        data).model_dump() for data in user_experiment_data]

    user_seed = app.storage.user['seed']
    user_data_file = f'data/user_{user_seed}.json'
    with open(user_data_file, 'w') as f:
      json.dump(data_dicts, f)
    print(f'saved: {user_data_file}')


#####################################
# Setup database
#####################################
directory = 'data'
if not os.path.exists(directory):
    os.mkdir(directory)

async def init_db() -> None:
    await Tortoise.init(
       db_url=f'sqlite://data/{DATABASE_FILE}',
       modules={'models': ['models']})
    await Tortoise.generate_schemas()

async def close_db() -> None:
    await Tortoise.close_connections()

app.on_startup(init_db)
app.on_shutdown(close_db)

#####################################
# Start page
#####################################


def check_if_over(*args, **kwargs):
   minutes_passed = nicewebrl.get_user_session_minutes()
   if minutes_passed > 60:
      app.storage.user['stage_idx'] = 1000
      finish_experiment(*args, **kwargs)

def footer():
  with ui.row():
      ui.label().bind_text_from(
          app.storage.user, 'seed',
          lambda v: f"user id: {v}.")
      ui.label()
      ui.label().bind_text_from(
          app.storage.user, 'stage_idx',
          lambda v: f"stage: {v}.")
      ui.label()
      ui.label().bind_text_from(
          app.storage.user, 'session_duration',
          lambda v: f"minutes passed: {int(v)}.")

@ui.page('/')
async def index():
    nicewebrl.initialize_user(debug=DEBUG)
    basic_javascript_file = nicewebrl.basic_javascript_file()
    with open(basic_javascript_file) as f:
        ui.add_body_html('<script>' + f.read() + '</script>')

    card = ui.card(align_items=['center']).classes('fixed-center')

    with card:
      stage_container = ui.card()
      button_container = ui.column()
      with ui.column() as meta_container:
        # Run every 5 minutes
        ui.timer(60, lambda: check_if_over(
            meta_container, stage_container, button_container))

        if app.storage.user.get('experiment_started', False):
          await start_experiment(
             meta_container, stage_container, button_container)
        else: # very initial page
          make_consent_form(
             meta_container, stage_container, button_container)

      footer()

#secret_key = secrets.token_urlsafe(32)
ui.run(
   storage_secret='private key to secure the browser session cookie',
   reload='FLY_ALLOC_ID' not in os.environ,
   title=APP_TITLE,
   )
