import asyncio

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
import random
from datetime import datetime, timedelta


import gcs
import nicewebrl
from nicewebrl.stages import ExperimentData
from nicewebrl.utils import wait_for_button_or_keypress

from google.auth.exceptions import TransportError
from load_data import get_block_stage_description, dict_to_string

load_dotenv()

DATABASE_FILE = os.environ.get('DB_FILE', 'db.sqlite')
NAME = os.environ.get('NAME', 'exp')
DEBUG = int(os.environ.get('DEBUG', 0))
DEBUG_SEED = int(os.environ.get('SEED', 42))
EXPERIMENT = int(os.environ.get('EXP', 1))

if EXPERIMENT == 0:
  import experiment_test as experiment
  APP_TITLE = 'Human Dyna Test'
elif EXPERIMENT == 1:
  import experiment_1 as experiment
  APP_TITLE = 'Human Dyna 1'
elif EXPERIMENT == 2:
  import experiment_2 as experiment
  APP_TITLE = 'Dyna 2'
else:
   raise NotImplementedError
all_stages = experiment.all_stages

DATABASE_FILE = f'{DATABASE_FILE}_name={NAME}_exp={EXPERIMENT}_debug={DEBUG}'

#####################################
# Consent Form
#####################################

def make_consent_form(
    meta_container, stage_container, button_container
):
  ui.markdown('## Consent Form')
  with open('consent.md', 'r') as consent_file:
      consent_text = consent_file.read()
  ui.markdown(consent_text)
  ui.checkbox(
    'I agree to participate.',
    on_change=lambda: collect_demographic_info(
       meta_container, stage_container, button_container))


def collect_demographic_info(meta_container, stage_container, button_container):
    # Create a markdown title for the section
    meta_container.clear()
    with meta_container:
      ui.markdown('## Demographic Info')
      ui.markdown('Please fill out the following information.')

      with ui.column():
        with ui.column():
          ui.label('Biological Sex')
          sex_input = ui.radio(['Male', 'Female'], value='Male').props('inline')

        # Collect age with a textbox input
        age_input = ui.input('Age')


      # Button to submit and store the data
      async def submit():
          age = age_input.value
          sex = sex_input.value

          # Validation for age input
          if not age.isdigit() or not (0 < int(age) < 100):
              ui.notify(
                  "Please enter a valid age between 1 and 99.", type="warning")
              return
          app.storage.user['age'] = int(age)
          app.storage.user['sex'] = sex

          print("started experiment for user:", app.storage.user['seed'])
          print(f"age: {int(age)}, sex: {sex}")
          await start_experiment(meta_container, stage_container, button_container)

      ui.button('Submit', on_click=submit)


#####################################
# Start/load experiment
#####################################
def get_stage(stage_idx):
   if app.storage.user.get('experiment_finished', False):
      return all_stages[-1]
   stage_order = app.storage.user['stage_order']
   stage_idx = stage_order[stage_idx]
   return all_stages[stage_idx]

def get_block_idx(stage):
  # says which current block we're in
  # e.g. 3. from [0, 1, 3, 2]
  block_order = stage.metadata['block_metadata']['idx']

  # for 3, I'd want to get but 2.
  # how do we get that?
  block_idx = app.storage.user['block_order_to_idx'][str(block_order)]
  return block_idx

def block_progress():
   """Return a 2-digit rounded decimal of the progress."""
   return float(f"{(app.storage.user.get('block_idx')+1)/len(experiment.all_blocks):.2f}")

async def start_experiment(
      meta_container,
      stage_container,
      button_container):
  if DEBUG == 0:
    ui.run_javascript(
       'document.documentElement.requestFullscreen()',
       timeout=10)
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
  stage = get_stage(app.storage.user['stage_idx'])
  await stage.handle_key_press(e, stage_container)
  if stage.get_user_data('finished', False):
    app.storage.user['stage_idx'] += 1
    await load_stage(meta_container, stage_container, button_container)

async def handle_button_press(*args, button_container, **kwargs):
  if DEBUG == 0 and not await nicewebrl.utils.check_fullscreen():
    ui.notify('Please enter fullscreen mode to continue experiment',
              type='negative')
    return
  button_container.clear()
  stage = get_stage(app.storage.user['stage_idx'])
  await stage.handle_button_press()
  if stage.get_user_data('finished', False):
    app.storage.user['stage_idx'] += 1
    await load_stage(*args, button_container=button_container, **kwargs)


async def handle_timer_finished(*args, button_container, **kwargs):
  if DEBUG == 0 and not await nicewebrl.utils.check_fullscreen():
    ui.notify('Please enter fullscreen mode to continue experiment',
              type='negative')
    return
  button_container.clear()
  stage = get_stage(app.storage.user['stage_idx'])
  notification = ui.notification(
      'The timer has run out.',
      position='center', type='info')
  await stage.finish_stage()
  with button_container:
    button = ui.button("click to continue")
    await button.clicked()
    notification.dismiss()
  if stage.get_user_data('finished', False):
    app.storage.user['stage_idx'] += 1
    await load_stage(*args, button_container=button_container, **kwargs)

async def save_on_new_block():
    if app.storage.user['block_idx'] == 0: return
    prior_stage = get_stage(app.storage.user['stage_idx']-1)
    stage = get_stage(app.storage.user['stage_idx'])
    prior_block = prior_stage.metadata['block_metadata']['desc']
    block = stage.metadata['block_metadata']['desc']

    if block != prior_block:
       print("-"*10)
       print(f"Saving results from block: `{prior_block}`")
       asyncio.create_task(save_data(final_save=False))

async def load_stage(meta_container, stage_container, button_container):
    """Default behavior for progressing through stages."""
    if app.storage.user['stage_idx'] >= len(all_stages):
        await finish_experiment(meta_container, stage_container, button_container)
        return

    await save_on_new_block()
    #########
    # Activate new stage
    #########
    stage_idx = app.storage.user['stage_idx']
    stage = get_stage(app.storage.user['stage_idx'])
    app.storage.user['block_idx'] = get_block_idx(stage)
    app.storage.user['block_progress'] = block_progress()
    with stage_container.style('align-items: center;'):
      await stage.activate(stage_container)

    with button_container.style('align-items: center;'):
      button_container.clear()
      ####################
      # Timer
      ####################
      if stage.duration:
        # get ending
        default_end_time = datetime.now() + timedelta(seconds=stage.duration)

        # either re-use stored end time, or if none, use end time above
        app.storage.user[f'{stage_idx}_end'] = app.storage.user.get(
            f'{stage_idx}_end', default_end_time)
        with ui.element('div').classes('p-2 bg-orange-100'):
          countdown_label = ui.label(f"Seconds left: {stage.duration}")

          async def update_countdown():
            remaining = app.storage.user[f'{stage_idx}_end'] - datetime.now()
            if remaining.total_seconds() <= 0:
                await handle_timer_finished(
                    meta_container=meta_container,
                    stage_container=stage_container,
                    button_container=button_container)
            else:
                countdown_label.set_text(
                    f"Seconds left: {remaining.seconds:02d}")
          ui.timer(0.1, update_countdown)

      ####################
      # Button to go to next page
      ####################
      button = ui.button('Next page').bind_visibility_from(stage, 'next_button')
      if stage.next_button:
        await wait_for_button_or_keypress(button)
        await handle_button_press(
                    meta_container=meta_container,
                    stage_container=stage_container,
                    button_container=button_container)

async def finish_experiment(meta_container, stage_container, button_container):
    meta_container.clear()
    stage_container.clear()
    button_container.clear()

    experiment_finished = app.storage.user.get('experiment_finished', False)

    if experiment_finished and not DEBUG:
      # in case called multiple times
      return

    #########################
    # Save data
    #########################
    async def submit(feedback):
      app.storage.user['experiment_finished'] = True
      with meta_container:
        meta_container.clear()
        ui.markdown(f"## Saving data. Please wait")
        ui.markdown(
          "**Once the data is uploaded, this app will automatically move to the next screen**")

      # wait 5 seconds to make sure data from stages are saved
      if not DEBUG:
        await asyncio.sleep(5)
      # when over, delete user data.
      await save_data(final_save=True, feedback=feedback)
      app.storage.user['data_saved'] = True


    app.storage.user['data_saved'] = app.storage.user.get(
        'data_saved', False)
    if not app.storage.user['data_saved']:
      with meta_container:
        meta_container.clear()
        ui.markdown("Please provide feedback on the experiment here. For example, please describe if anything went wrong or if you have any suggestions for the experiment.")
        text = ui.textarea().style('width: 80%;')  # Set width to 80% of the container
        button = ui.button("Submit")
        await button.clicked()
        await submit(text.value)

    #########################
    # Final screen
    #########################
    with meta_container:
        meta_container.clear()

        ui.markdown("# Experiment over")
        ui.markdown("## Data saved")
        ui.markdown("### Please record the following code which you will need to provide for compensation")
        ui.markdown(
            f'### "gershman_dyna"')
        ui.markdown("#### You may close the browser")

async def compute_bonus(data_dicts):

    train_successes = 0
    train_episodes = 0
    eval_successes = 0
    eval_episodes = 0
    keys = set()
    for datum in data_dicts[::-1]:
       info = get_block_stage_description(datum)
       desc = dict_to_string(info)
       if 'practice' in datum['metadata']['block_metadata']['desc']:
          continue
       if desc not in keys:
          keys.add(desc)
          if datum['metadata']['eval']:
            eval_successes += datum['metadata']['nsuccesses']
            eval_episodes += datum['metadata']['episode_idx']
          else:
            train_successes += datum['metadata']['nsuccesses']
            train_episodes += datum['metadata']['episode_idx']
    #train_sr = (train_successes / max(1, train_episodes))
    eval_sr = (eval_successes / max(1, eval_episodes))
    #train_sr = train_sr*(train_sr > .5)
    #train_bonus = 1*train_sr  # bounded between 0 and 1
    #eval_bonus = eval_sr*(train_sr > .5)
    if eval_sr < .25:
       return 0
    elif eval_sr < .5:
       return 1
    elif eval_sr < .75:
       return 2
    else:
       return 3

async def save_data(final_save=True, feedback=None, **kwargs):
    # Create a Pydantic model from your Tortoise model
    ExperimentDataPydantic = pydantic_model_creator(ExperimentData)
    ExperimentDataPydantic.model_config['from_attributes'] = True

    user_experiment_data = await ExperimentData.filter(
        session_id=app.storage.browser['id']).all()

    data_dicts = [ExperimentDataPydantic.model_validate(
        data).model_dump() for data in user_experiment_data]

    if final_save:
      bonus = await compute_bonus(data_dicts)
      data_dicts.append(dict(
         finished=True,
         feedback=feedback,
         bonus=bonus,
         **kwargs,
         ))
    user_seed = app.storage.user['seed']
    user_data_file = f'data/data_user={user_seed}_name={NAME}_exp={EXPERIMENT}_debug={DEBUG}.json'
    with open(user_data_file, 'w') as f:
      json.dump(data_dicts, f)

    await save_to_gcs(user_data=data_dicts, filename=user_data_file)

    # Now delete the data from the database
    if final_save:
      await ExperimentData.filter(session_id=app.storage.browser['id']).delete()

async def save_to_gcs(user_data, filename):
    try:
      bucket = gcs.initialize_storage_client()
      blob = bucket.blob(filename)
      blob.upload_from_string(data=json.dumps(
          user_data), content_type='application/json')
      print(f'Saved {filename} in bucket {bucket.name}')
    except TransportError as te:
       print(te)
       print("No internet connection maybe?")
    except Exception as e:
       raise e

async def check_if_over(*args, episode_limit=60, ** kwargs):
   minutes_passed = nicewebrl.get_user_session_minutes()
   minutes_passed = app.storage.user['session_duration']
   if minutes_passed > episode_limit:
      print(f"experiment timed out after {minutes_passed} minutes")
      app.storage.user['stage_idx'] = len(all_stages)
      await finish_experiment(*args, **kwargs)

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
        ui.label()
        ui.label().bind_text_from(
            app.storage.user, 'block_idx',
            lambda v: f"block: {int(v)+1}/{len(experiment.all_blocks)}.")

    ui.linear_progress(
      value=block_progress()).bind_value_from(app.storage.user, 'block_progress')
    ui.button(
        'Toggle fullscreen', icon='fullscreen',
        on_click=nicewebrl.utils.toggle_fullscreen).props('flat')

def initalize_user():

  nicewebrl.initialize_user(debug=DEBUG, debug_seed=DEBUG_SEED)
  print(f"Initialized user: {app.storage.user['seed']}")
  app.storage.user['stage_idx'] = app.storage.user.get('stage_idx', 0)
  app.storage.user['block_idx'] = app.storage.user.get('block_idx', 0)
  app.storage.user['block_progress'] = app.storage.user.get('block_progress', 0.)

  stage_order = app.storage.user.get('stage_order', None)
  block_order_to_idx = app.storage.user.get('block_order_to_idx', None)

  print(f"Loaded block: {app.storage.user['block_idx']}")
  print(f"Loaded stage: {app.storage.user['stage_idx']}")
  if not stage_order:
    init_rng_key = jnp.array(
        app.storage.user['init_rng_key'], dtype=jnp.uint32)
    # example block order
    # [e.g., 0, 1, 3, 2]

    block_order, stage_order = experiment.generate_block_stage_order(init_rng_key)
    block_order_to_idx = {str(i): int(idx) for idx, i in enumerate(block_order)}

  app.storage.user['stage_order'] = stage_order
  # this will be used to track which block you're currently in
  
  app.storage.user['block_order_to_idx'] = block_order_to_idx

@ui.page('/')
async def index(request: Request):
    initalize_user()
    ui.on('ping', lambda e: print(e.args))

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
        filename=f'data/info_user={user_seed}_name={NAME}_exp={EXPERIMENT}_debug={DEBUG}.json')

    ################
    # Start experiment
    ################
    basic_javascript_file = nicewebrl.basic_javascript_file()
    with open(basic_javascript_file) as f:
        ui.add_body_html('<script>' + f.read() + '</script>')

    card = ui.card(align_items=['center']).classes('fixed-center').style(
        'max-width: 90vw;'  # Set the max width of the card
        'max-height: 90vh;'  # Ensure the max height is 90% of the viewport height
        'overflow: auto;'  # Allow scrolling inside the card if content overflows
        'display: flex;'  # Use flexbox for centering
        'flex-direction: column;'  # Stack content vertically
        'justify-content: flex-start;'
        'align-items: center;'
    )
    with card:
      episode_limit = 120
      ui.timer(
        1,  # check every minute
        lambda: check_if_over(
            episode_limit=episode_limit,
            meta_container=meta_container, 
            stage_container=stage_container,
            button_container=button_container))
      stage_container = ui.column()
      button_container = ui.column()
      with ui.column() as meta_container:
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
