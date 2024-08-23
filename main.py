
from dotenv import load_dotenv
import json
import jax.numpy as jnp
from nicegui import app, ui
import nicewebrl
import nicewebrl.nicejax
import nicewebrl.stages
import nicewebrl.utils
from fastapi import Request
from tortoise import Tortoise
from tortoise.contrib.pydantic import pydantic_model_creator
import os

import gcs
import nicewebrl
from nicewebrl.stages import ExperimentData


load_dotenv()

DATABASE_FILE = os.environ.get('DB_FILE', 'db.sqlite')
DEBUG = int(os.environ.get('DEBUG', 0))
DEBUG_SEED = int(os.environ.get('DEBUG_SEED', 42))
EXPERIMENT = int(os.environ.get('EXP', 1))

if EXPERIMENT == 0:
  import experiment_test as experiment
  APP_TITLE = 'Human Dyna Test'
elif EXPERIMENT == 1:
  import experiment_1 as experiment
  APP_TITLE = 'Human Dyna 1'
else:
   raise NotImplementedError
all_stages = experiment.all_stages

DATABASE_FILE = f'{DATABASE_FILE}_exp{EXPERIMENT}_debug{DEBUG}'
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
  if DEBUG == 0:
    ui.run_javascript('document.documentElement.requestFullscreen()')
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
  if DEBUG == 0 and not await nicewebrl.utils.check_fullscreen():
    ui.notify(
       'Please enter fullscreen mode to continue experiment',
       type='negative', timeout=10)
    return
  stage = all_stages[app.storage.user['stage_idx']]
  await stage.handle_key_press(e, stage_container)
  if stage.get_user_data('finished', False):
    app.storage.user['stage_idx'] += 1
    await load_stage(meta_container, stage_container, button_container)

async def handle_button_press(*args, **kwargs):
  if DEBUG == 0 and not await nicewebrl.utils.check_fullscreen():
    ui.notify('Please enter fullscreen mode to continue experiment',
              type='negative')
    return
  stage = all_stages[app.storage.user['stage_idx']]
  await stage.handle_button_press()
  if stage.get_user_data('finished', False):
    app.storage.user['stage_idx'] += 1
    await load_stage(*args, **kwargs)


async def load_stage(meta_container, stage_container, button_container):
    """Default behavior for progressing through stages."""
    if app.storage.user['stage_idx'] >= len(all_stages):
        await finish_experiment(meta_container, stage_container, button_container)
        return

    stage = all_stages[app.storage.user['stage_idx']]
    with stage_container.style('align-items: center;'):
      await stage.activate(stage_container)

    with button_container.style('align-items: center;'):
      button_container.clear()
      ui.button('Next page',
                on_click=lambda: handle_button_press(
                    meta_container, stage_container, button_container)
                ).bind_visibility_from(stage, 'next_button')


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
        ui.markdown(f"## Saving data. Please wait")
        ui.markdown(
           "**Once the data is uploaded, this app will automatically move to the next screen**")
      await save_data()
      app.storage.user['data_saved'] = True

    with meta_container:
        meta_container.clear()
        ui.markdown("## Experiment over")
        ui.markdown("### Data saved")
        ui.markdown("#### You may close the browser")


async def save_data():
    # Create a Pydantic model from your Tortoise model
    ExperimentDataPydantic = pydantic_model_creator(ExperimentData)
    ExperimentDataPydantic.model_config['from_attributes'] = True

    user_experiment_data = await ExperimentData.filter(
        session_id=app.storage.browser['id']).all()

    data_dicts = [ExperimentDataPydantic.model_validate(
        data).model_dump() for data in user_experiment_data]

    user_seed = app.storage.user['seed']
    user_data_file = f'data/data_user={user_seed}_exp={EXPERIMENT}_debug={DEBUG}.json'
    with open(user_data_file, 'w') as f:
      json.dump(data_dicts, f)

    await save_to_gcs(user_data=data_dicts, filename=user_data_file)

    # Now delete the data from the database
    await ExperimentData.filter(session_id=app.storage.browser['id']).delete()

async def save_to_gcs(user_data, filename):
    bucket = gcs.initialize_storage_client()
    blob = bucket.blob(filename)
    blob.upload_from_string(data=json.dumps(
        user_data), content_type='application/json')
    print(f'Saved {filename} in bucket {bucket.name}')


def check_if_over(*args, episode_limit=60, ** kwargs):
   minutes_passed = nicewebrl.get_user_session_minutes()
   if minutes_passed > episode_limit:
      print(f"experiment timed out after {minutes_passed} minutes")
      app.storage.user['stage_idx'] = 1000
      finish_experiment(*args, **kwargs)

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
# Home page
#####################################

def footer(card):
  with card:
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
    ui.button(
        'Toggle fullscreen', icon='fullscreen',
        on_click=nicewebrl.utils.toggle_fullscreen).props('flat')

def initalize_user():
  nicewebrl.initialize_user(debug=DEBUG, debug_seed=DEBUG_SEED)
  if 'stage_order' not in app.storage.user:
    init_rng_key = jnp.array(
        app.storage.user['init_rng_key'], dtype=jnp.uint32)
    stage_order = experiment.generate_stage_order(init_rng_key)
    app.storage.user['stage_order'] = stage_order

@ui.page('/')
async def index(request: Request):
    initalize_user()

    ui.run_javascript(f'window.debug = {DEBUG}')
    ################
    # Get user data and save to GCS
    ################
    user_info = dict(
        worker_id=request.query_params.get('workerId', 'default_worker'),
        hit_id=request.query_params.get('hitId', 'default_hit'),
        assignment_id=request.query_params.get(
            'assignmentId', 'default_assignment')
    )
    user_seed = app.storage.user['seed']
    await save_to_gcs(
        user_data=user_info,
        filename=f'data/info_user={user_seed}_exp={EXPERIMENT}_debug={DEBUG}.json')

    ################
    # Start experiment
    ################
    basic_javascript_file = nicewebrl.basic_javascript_file()
    with open(basic_javascript_file) as f:
        ui.add_body_html('<script>' + f.read() + '</script>')

    card = ui.card(align_items=['center']).classes('fixed-center')
    with card:
      stage_container = ui.column()
      button_container = ui.column()
      with ui.column() as meta_container:
        # Run every 5 minutes
        episode_limit = 60 if DEBUG else 60
        ui.timer(
           1,  # check every minute
           lambda: check_if_over(
               episode_limit=episode_limit,
               meta_container=meta_container, 
               stage_container=stage_container,
               button_container=button_container))

        if app.storage.user.get('experiment_started', False):
          await start_experiment(
             meta_container, stage_container, button_container)
        else: # very initial page
          make_consent_form(
             meta_container, stage_container, button_container)
      footer(card)

ui.run(
   storage_secret='private key to secure the browser session cookie',
   reload='FLY_ALLOC_ID' not in os.environ,
   title=APP_TITLE,
   )
