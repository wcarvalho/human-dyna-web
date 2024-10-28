import logging
import inspect
import collections
import asyncio

from dotenv import load_dotenv
import json
import jax.numpy as jnp
from nicegui import app, ui
import nicewebrl
import sys
import nicewebrl.nicejax
import nicewebrl.stages
import nicewebrl.utils
from fastapi import Request
from tortoise import Tortoise
from tortoise.contrib.pydantic import pydantic_model_creator
import os
import random
from pprint import pprint
from datetime import datetime, timedelta


import gcs
import nicewebrl
from nicewebrl.stages import ExperimentData
from nicewebrl.utils import wait_for_button_or_keypress, clear_element
from nicewebrl.logging import setup_logging, get_logger

from google.auth.exceptions import TransportError
from load_data import get_block_stage_description, dict_to_string, time_diff
from google.cloud import exceptions as gcs_exceptions

load_dotenv()

DATABASE_FILE = os.environ.get('DB_FILE', 'db.sqlite')
NAME = os.environ.get('NAME', 'exp')
LOG_DIR = os.environ.get('LOG_DIR', 'data/')
DEBUG = int(os.environ.get('DEBUG', 0))
DEBUG_SEED = int(os.environ.get('SEED', 0))
EXPERIMENT = int(os.environ.get('EXP', 4))
LIGHT = int(os.environ.get('LIGHT', 0))
os.makedirs(LOG_DIR, exist_ok=True)

setup_logging(LOG_DIR,
              nicegui_storage_user_key='user_id')
logger = get_logger('main')

if EXPERIMENT == 0:
  #import experiment_test as experiment
  #APP_TITLE = 'Human Dyna Test'
  pass
elif EXPERIMENT == 1:
  #import experiment_1 as experiment
  #APP_TITLE = 'Human Dyna 1'
  pass
elif EXPERIMENT == 2:
  #import experiment_2 as experiment
  #APP_TITLE = 'Dyna 2'
  pass
elif EXPERIMENT == 3:
  import experiment_3 as experiment
  APP_TITLE = 'Dyna 3'
elif EXPERIMENT == 4:
  import experiment_4 as experiment
  APP_TITLE = 'Dyna 4'
else:
   raise NotImplementedError
all_stages = experiment.all_stages

DATABASE_FILE = f'{DATABASE_FILE}_name={NAME}_debug={DEBUG}'

def user_log_file(log_dir):
  user_id = app.storage.user.get('user_id')
  return os.path.join(log_dir, f'log_{user_id}.log')



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
    clear_element(meta_container)
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

          logger.info("started experiment for user:", app.storage.user['seed'])
          logger.info(f"age: {int(age)}, sex: {sex}")
          await start_experiment(meta_container, stage_container, button_container)

      ui.button('Submit', on_click=submit)


#####################################
# Start/load experiment
#####################################:
def update_stage():
  #-------------------
  # Update stage index
  #-------------------
  stage_idx = app.storage.user['stage_idx']
  if app.storage.user.get('experiment_finished', False):
    stage_idx = len(all_stages)
  if stage_idx < len(all_stages):
    stage_idx += 1
  else:
     stage_idx = len(all_stages)
  app.storage.user['stage_idx'] = stage_idx

  # -------------------
  # Print stage information
  # -------------------
  # Get the current frame and the caller's frame
  current_frame = inspect.currentframe()
  caller_frame = current_frame.f_back
  # Extract the name of the calling function
  fn_name = caller_frame.f_code.co_name if caller_frame else "Unknown"

  if stage_idx >= len(all_stages):
    name = "Finished experiment"
    order_stage_idx = len(all_stages)
  else:
    stage_order = app.storage.user['stage_order']
    order_stage_idx = stage_order[stage_idx]
    name = all_stages[order_stage_idx].name

    block_idx = app.storage.user['stage_to_block_idx'][order_stage_idx][0]
    manipulation = all_stages[order_stage_idx].metadata['block_metadata'].get('short', 'generic')
    desc = f"stage: {stage_idx}/{len(all_stages)}. "
    desc += f"{manipulation} block: idx {block_idx}: {name}"
    logger.info(desc)
    logger.info(desc)

  return stage_idx

def get_stage(raw_stage_idx):
  if app.storage.user.get('experiment_finished', False):
    return all_stages[-1]
  if raw_stage_idx >= len(all_stages):
     return all_stages[-1]
  stage_order = app.storage.user['stage_order']
  try:
    order_stage_idx = stage_order[raw_stage_idx]
  except IndexError as e:
    msg = f"raw_stage_idx: {raw_stage_idx}/{len(stage_order)}, order_stage_idx: {order_stage_idx}/{len(stage_order)}"
    logger.info(msg)
    msg = f"stage order: {stage_order}"
    logger.info(msg)
    raise RuntimeError(msg)
  except Exception as e:
    raise e
  return all_stages[order_stage_idx]

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
       'document.documentElement.requestFullscreen()')
  app.storage.user['experiment_started'] = True

  if app.storage.user.get('experiment_finished', False):
    await finish_experiment(
       meta_container, stage_container, button_container)
    return

  nicewebrl.get_user_session_minutes()
  clear_element(meta_container)
  ui.on('key_pressed', 
        lambda e: handle_key_press(e, meta_container, stage_container, button_container))
  await load_stage(meta_container, stage_container, button_container)

async def handle_key_press(e, meta_container, stage_container, button_container):
  if DEBUG == 0 and not await nicewebrl.utils.check_fullscreen():
    ui.notify(
       'Please enter fullscreen mode to continue experiment',
       type='negative')
    return
  stage = get_stage(app.storage.user['stage_idx'])
  await stage.handle_key_press(e, stage_container)
  if stage.get_user_data('finished', False):
    update_stage()
    await load_stage(meta_container, stage_container, button_container)

async def handle_button_press(*args, button_container, **kwargs):
  if DEBUG == 0 and not await nicewebrl.utils.check_fullscreen():
    ui.notify('Please enter fullscreen mode to continue experiment',
              type='negative')
    return
  clear_element(button_container)
  stage = get_stage(app.storage.user['stage_idx'])
  await stage.handle_button_press()
  if stage.get_user_data('finished', False):
    update_stage()
    await load_stage(*args, button_container=button_container, **kwargs)

async def handle_timer_finished(*args, button_container, **kwargs):
  if DEBUG == 0 and not await nicewebrl.utils.check_fullscreen():
    ui.notify('Please enter fullscreen mode to continue experiment',
              type='negative')
    return
  clear_element(button_container)
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
    update_stage()
    await load_stage(*args, button_container=button_container, **kwargs)

async def save_on_new_block():
    if app.storage.user['block_idx'] == 0: return
    prior_stage = get_stage(app.storage.user['stage_idx']-1)
    stage = get_stage(app.storage.user['stage_idx'])
    prior_block = prior_stage.metadata['block_metadata'].get('desc', None)
    block = stage.metadata['block_metadata'].get('desc', None)
    if block is None or prior_block is None:
      return

    if block != prior_block:
       logger.info("-"*10)
       logger.info(f"Saving results from block: `{prior_block}`")
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

    if stage.get_user_data('finished', False):
      update_stage()
      return await load_stage(meta_container, stage_container, button_container)


    with button_container.style('align-items: center;'):
      clear_element(button_container)
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
            if stage.get_user_data('finished', False):
               clear_element(button_container)
               return 
            current_end_time = app.storage.user[f'{stage_idx}_end']
            if not isinstance(current_end_time, datetime):
              current_end_time = datetime.fromisoformat(current_end_time)
            remaining = current_end_time - datetime.now()
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
    clear_element(meta_container)
    clear_element(stage_container)
    clear_element(button_container)

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
        clear_element(meta_container)
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
        clear_element(meta_container)
        ui.markdown("Please provide feedback on the experiment here. For example, please describe if anything went wrong or if you have any suggestions for the experiment.")
        text = ui.textarea().style('width: 80%;')  # Set width to 80% of the container
        button = ui.button("Submit")
        await button.clicked()
        await submit(text.value)

    #########################
    # Final screen
    #########################
    with meta_container:
        clear_element(meta_container)
        key = {
           0: "Ym3sa",
           1: "Mja2S",
           2: "Ujas14",
           3: "Ukla0j",
        }[app.storage.user['bonus']]
        ui.markdown("# Experiment over")
        ui.markdown("## Data saved")
        ui.markdown("### Please record the following code which you will need to provide for compensation")
        ui.markdown(
            f'### gershman.dyna.{key}')
        ui.markdown("#### You may close the browser")

async def compute_bonus(data_dicts):

    train_successes = 0
    train_episodes = 0
    eval_successes = 0
    eval_episodes = 0
    keys = set()
    successes = 0
    npossible = 0
    for datum in data_dicts[::-1]:
       if 'practice' in datum['metadata']['block_metadata'].get('desc', ''):
          continue
       if 'feedback' in datum['metadata']['block_metadata'].get('desc', ''):
          continue
       info = get_block_stage_description(datum)
       desc = dict_to_string(info)
       if desc not in keys:
          keys.add(desc)
          user_data = await ExperimentData.filter(
              session_id=app.storage.browser['id'],
              name=datum['name'],
          )
          first = user_data[0].data['image_seen_time']
          last = user_data[-1].data['action_taken_time']
          seconds = time_diff(first, last)/1000
          timelimit = user_data[0].data['timelimit']
          if timelimit is not None:
            successes += seconds < timelimit
            npossible += 1
          if datum['metadata'].get('eval', False):
            eval_successes += datum['metadata']['nsuccesses']
            eval_episodes += datum['metadata']['episode_idx']
          else:
            train_successes += datum['metadata']['nsuccesses']
            train_episodes += datum['metadata']['episode_idx']
    train_sr = (train_successes / max(1, train_episodes))
    bonus_sr = successes / max(1, npossible)
    bonus_sr = bonus_sr*(train_sr > .5)

    if bonus_sr < .25:
       return 0
    elif bonus_sr < .5:
       return 1
    elif bonus_sr < .75:
       return 2
    else:
       return 3

async def save_data(final_save=True, feedback=None, **kwargs):
    # Create a Pydantic model from your Tortoise model
    ExperimentDataPydantic = pydantic_model_creator(ExperimentData)
    ExperimentDataPydantic.model_config['from_attributes'] = True

    user_experiment_data = await ExperimentData.filter(
        session_id=app.storage.browser['id']).order_by('id').all()

    data_dicts = [ExperimentDataPydantic.model_validate(
        data).model_dump() for data in user_experiment_data]

    bonus = 0
    if final_save:
      bonus = await compute_bonus(data_dicts)
      app.storage.user['bonus'] = bonus
      user_storage = nicewebrl.nicejax.make_serializable(dict(app.storage.user))
      data_dicts.append(dict(
         finished=True,
         feedback=feedback,
         bonus=bonus,
         user_storage=user_storage,
         **kwargs,
         ))
    user_seed = app.storage.user['seed']
    user_data_file = f'data/data_user={user_seed}_name={NAME}_debug={DEBUG}.json'
    with open(user_data_file, 'w') as f:
      json.dump(data_dicts, f)

    if not DEBUG:
        if final_save:
            max_retries = 5
            retry_delay = 5  # seconds
            for attempt in range(max_retries):
                try:
                    saved = await save_to_gcs(user_data=data_dicts, filename=user_data_file)
                    if not saved: continue
                    log_file = user_log_file(LOG_DIR)
                    bucket = gcs.initialize_storage_client()
                    blob = bucket.blob(
                        f'logs/user={user_seed}_name={NAME}_debug={DEBUG}.log')
                    blob.upload_from_filename(log_file)
                    logger.info(f"Successfully saved data to GCS on attempt {attempt + 1}")
                    break
                except (TransportError, gcs_exceptions.GoogleCloudError) as e:
                    if attempt < max_retries - 1:
                        logger.info(f"Error saving to GCS: {e}. Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.info(f"Failed to save to GCS after {max_retries} attempts: {e}")
        else:
            # Non-final save, just attempt once
            try:
                await save_to_gcs(user_data=data_dicts, filename=user_data_file)
            except Exception as e:
                logger.info(f"Error saving to GCS (non-final save): {e}")

    # Now delete the data from the database
    if final_save:
      await ExperimentData.filter(session_id=app.storage.browser['id']).delete()


async def save_to_gcs(user_data, filename):
    try:
        bucket = gcs.initialize_storage_client()
        blob = bucket.blob(filename)
        blob.upload_from_string(data=json.dumps(user_data), content_type='application/json')
        logger.info(f'Saved {filename} in bucket {bucket.name}')
        return True  # Successfully saved
    except (TransportError, gcs_exceptions.GoogleCloudError) as e:
        logger.info(f"Error saving to GCS: {e}")
    except Exception as e:
        logger.info(f"Unexpected error: {e}")
        logger.info("Skipping GCS upload")
    
    return False  # Failed to save

async def check_if_over(*args, episode_limit=60, ** kwargs):
   minutes_passed = nicewebrl.get_user_session_minutes()
   minutes_passed = app.storage.user['session_duration']
   if minutes_passed > episode_limit:
      logger.info(f"experiment timed out after {minutes_passed} minutes")
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


def initalize_user(user_info):
  #########
  # User settings
  #########
  nicewebrl.initialize_user(debug_seed=DEBUG_SEED)

  app.storage.user['user_id'] = user_info['worker_id'] or app.storage.user['seed']

  #########
  # Stage settings
  #########
  app.storage.user['stage_idx'] = app.storage.user.get('stage_idx', 0)
  app.storage.user['block_idx'] = app.storage.user.get('block_idx', 0)
  app.storage.user['block_progress'] = app.storage.user.get('block_progress', 0.)

  stage_order = app.storage.user.get('stage_order', None)
  block_order_to_idx = app.storage.user.get('block_order_to_idx', None)

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

  #########
  # Logging
  #########

  logger.info(f"Initialized user: {app.storage.user['seed']}")
  logger.info(f"Loaded block: {app.storage.user['block_idx']}")
  logger.info(f"Loaded stage order: {stage_order}")
  logger.info(f"Loaded stage: {app.storage.user['stage_idx']}")
  stage_names = collections.OrderedDict()
  stage_to_block_idx = {}
  for i, stage_idx in enumerate(stage_order):
      stage = all_stages[stage_idx]
      block = stage.metadata.get('block_metadata', {}).get('idx', -1)
      stage_names[block] = stage_names.get(block, {})
      stage_names[block].update({i: (stage_idx, stage.name)})
  
  block, block_pieces = next(iter(stage_names.items()))
  for block, block_pieces in stage_names.items():
     for gloabl_idx, (idx_in_block, name) in block_pieces.items():
        stage_to_block_idx[gloabl_idx] = (idx_in_block % len(block_pieces), name)

  app.storage.user['stage_to_block_idx'] = stage_to_block_idx
  app.storage.user['stage_names'] = stage_names
  logger.info(f"Total stages: {len(all_stages)}")


@ui.page('/')
async def index(request: Request):
    user_info = dict(
        worker_id=request.query_params.get('workerId', None),
        hit_id=request.query_params.get('hitId', None),
        assignment_id=request.query_params.get(
            'assignmentId', None)
    )
    initalize_user(user_info)
    def print_ping(e):
      logger.info(str(e.args))
    ui.on('ping', print_ping)

    ui.run_javascript(f'window.debug = {DEBUG}')
    ################
    # Get user data and save to GCS
    ################
    user_seed = app.storage.user['seed']
    await save_to_gcs(
        user_data=user_info,
        filename=f'data/info_user={user_seed}_name={NAME}_debug={DEBUG}.json')

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
        if app.storage.user.get('experiment_started', False) or DEBUG:
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

