
from nicegui import app, ui
from pathlib import Path
import nicewebrl.nicejax
import nicewebrl.stages
from tortoise import Tortoise

##import experiment_1
import experiment_test as experiment
import nicewebrl
import models

stages = experiment.stages


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
        return finish_experiment(container)

    container.clear()
    stage = stages[app.storage.user['stage_idx']]
    await stage.run(container)

def finish_experiment(container):
    app.storage.user['experiment_finished'] = app.storage.user.get(
        'experiment_finished', True)
    with container:
        container.clear()
        ui.markdown(f"# Experiment over. You may close the browser")

#####################################
# Setup database
#####################################
async def init_db() -> None:
    await Tortoise.init(db_url='sqlite://db.sqlite2', modules={'models': ['models']})
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
      ui.label(f"user id: {app.storage.user['seed']}.")
      ui.label()
      ui.label(f"stage: {app.storage.user['stage_idx']}.")

@ui.page('/')
async def index():
    nicewebrl.initialize_user()
    basic_javascript_file = nicewebrl.basic_javascript_file()
    with open(basic_javascript_file) as f:
        ui.add_body_html('<script>' + f.read() + '</script>')

    experiment_started = app.storage.user.get('experiment_started', False)
    experiment_ended = app.storage.user.get('experiment_finished', False)

    with ui.card(align_items=['center']):
      with ui.column() as container:
        if experiment_ended:
          # final page
          finish_experiment(container)
        else:
          if experiment_started:
            # intermediary pages
            await load_experiment(container)
          else:
            # very initial page
            make_consent_form(container)
      footer()

#secret_key = secrets.token_urlsafe(32)
ui.run(storage_secret='private key to secure the browser session cookie')
