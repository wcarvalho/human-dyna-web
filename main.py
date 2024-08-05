
from nicegui import app, ui
from pathlib import Path

##import experiment_1
import experiment_test as experiment
import nicewebrl

stages = experiment.stages


def footer():
  with ui.row():
      ui.label(f"user: {app.storage.user['seed']}.")
      #ui.label(f"rng_key = {app.storage.user['rng_key']}")
      #ui.label(f"rng_splits = {app.storage.user['rng_splits']}")
      ui.label(f"stage: {app.storage.user['stage_idx']}")

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
def start_experiment(container):
  app.storage.user['experiment_started'] = True
  load_experiment(container)

def load_experiment(container):
  stage = stages[app.storage.user['stage_idx']]
  stage.load(container)

#####################################
# Start page
#####################################
@ui.page('/')
def index():
    app.storage.user['stage_idx'] = app.storage.user.get('stage_idx', 0)

    basic_javascript_file = nicewebrl.basic_javascript_file()
    with open(basic_javascript_file) as f:
        ui.add_body_html('<script>' + f.read() + '</script>')

    with ui.card(align_items=['center']):
      with ui.column() as container:
        if not app.storage.user['experiment_started']:
          make_consent_form(container)
        else:
          load_experiment(container)
      footer()



#secret_key = secrets.token_urlsafe(32)
ui.run(storage_secret='private key to secure the browser session cookie')
