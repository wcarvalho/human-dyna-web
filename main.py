
from dotenv import load_dotenv
import json
from nicegui import app, ui
import nicewebrl.nicejax
import nicewebrl.stages
from tortoise import Tortoise
from tortoise.contrib.pydantic import pydantic_model_creator
import os

##import experiment_1s
import experiment_test as experiment
import nicewebrl
from nicewebrl.stages import ExperimentData

stages = experiment.stages

load_dotenv()

APP_TITLE = 'Human Dyna Test'
DATABASE_FILE = os.environ.get('DB_FILE', 'db.sqlite')
DEBUG = int(os.environ.get('DEBUG', 0))

#####################################
# Consent Form
#####################################
def make_consent_form(container):
  ui.markdown('## Consent Form')
  ui.markdown('Please agree to this experiment.')
  ui.checkbox('I agree to participate.',
              on_change=lambda: start_experiment(container))

#####################################
# Start/load experiment
#####################################
async def start_experiment(container):
  app.storage.user['experiment_started'] = True
  await load_experiment(container)


async def load_experiment(container):
  with container.style('align-items: center;'):
    stage = stages[app.storage.user['stage_idx']]
    await stage.run(container)
    ui.on('key_pressed', lambda e: handle_key_press(e, container))

    button = stage.get_user_data('button')
    if button is not None:
       await button.clicked()
       await handle_button_press(container)


async def handle_button_press(container):
  stage = stages[app.storage.user['stage_idx']]
  await stage.handle_button_press()
  if stage.get_user_data('finished', False):
    await update_stage(container)

async def handle_key_press(e, container):
  stage = stages[app.storage.user['stage_idx']]
  await stage.handle_key_press(e, container)
  if stage.get_user_data('finished', False):
    await update_stage(container)

async def update_stage(container):
    """Default behavior for progressing through stages."""
    app.storage.user['stage_idx'] += 1
    if app.storage.user['stage_idx'] >= len(stages):
        await finish_experiment(container)
        return

    container.clear()
    stage = stages[app.storage.user['stage_idx']]
    await stage.run(container)

async def finish_experiment(container):
    app.storage.user['experiment_finished'] = True
    app.storage.user['data_saved'] = app.storage.user.get(
        'data_saved', False)
    if not app.storage.user['data_saved']:
      with container:
        container.clear()
        ui.markdown(f"# Saving data. Please wait")
      await save_data()
      app.storage.user['data_saved'] = True

    with container:
        container.clear()
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
def footer():
  with ui.row():
      ui.label().bind_text_from(
          app.storage.user, 'seed',
          lambda v: f"user id: {v}.")
      ui.label()
      ui.label().bind_text_from(
          app.storage.user, 'stage_idx',
          lambda v: f"stage: {v}.")
      #ui.label(f"user id:")
      #ui.label().bind_text_from(app.storage.user, 'seed')
      #ui.label(f"stage:")
      #ui.label().bind_text_from(app.storage.user, 'stage_idx')

@ui.page('/')
async def index():
    nicewebrl.initialize_user(debug=DEBUG)
    basic_javascript_file = nicewebrl.basic_javascript_file()
    with open(basic_javascript_file) as f:
        ui.add_body_html('<script>' + f.read() + '</script>')

    experiment_started = app.storage.user.get(
       'experiment_started', False)
    experiment_ended = app.storage.user.get(
       'experiment_finished', False)

    card = ui.card(align_items=['center']).classes('fixed-center')
    with card:
      with ui.column() as container:
        if experiment_ended: # final page
          await finish_experiment(container)
        elif experiment_started: # intermediary pages
          await load_experiment(container)
        else: # very initial page
          make_consent_form(container)
      footer()

#secret_key = secrets.token_urlsafe(32)
ui.run(
   storage_secret='private key to secure the browser session cookie',
   reload='FLY_ALLOC_ID' not in os.environ,
   title=APP_TITLE,
   )
