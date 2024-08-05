
from nicegui import app, ui
from pathlib import Path
import nicewebrl.nicejax
from tortoise import Tortoise

##import experiment_1
import experiment_test as experiment
import nicewebrl
from nicewebrl.stages import get_latest_stage_state
import models

stages = experiment.stages


#####################################
# Consent Form
#####################################
def make_consent_form(container):
  ui.markdown('## Consent Form')
  ui.markdown('Please agree to this experiment.')
  ui.checkbox('I consent.', on_change=lambda: start_experiment(container))

#####################################
# Start/load experiment
#####################################
async def start_experiment(container):
  app.storage.user['experiment_started'] = True
  await load_experiment(container)

async def load_experiment(container):
  stage = stages[app.storage.user['stage_idx']]
  #latest_stage_state = await get_latest_stage_state(cls=stage.state_cls)
  await stage.load(container, None)

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
      ui.label(f"user: {app.storage.user['seed']}.")
      ui.label(f"stage: {app.storage.user['stage_idx']}")

@ui.page('/')
async def index():
    nicewebrl.nicejax.init_rng()
    app.storage.user['stage_idx'] = app.storage.user.get('stage_idx', 0)

    basic_javascript_file = nicewebrl.basic_javascript_file()
    with open(basic_javascript_file) as f:
        ui.add_body_html('<script>' + f.read() + '</script>')

    with ui.card(align_items=['center']):
      with ui.column() as container:
        if not app.storage.user.get('experiment_started', False):
          make_consent_form(container)
        else:
          await load_experiment(container)
      footer()

#secret_key = secrets.token_urlsafe(32)
ui.run(storage_secret='private key to secure the browser session cookie')
