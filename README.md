# preplay
Web experiments for studying Human Dyna experiments




## Install (Dev)

**add submodules**
```
mkdir libraries
git submodule add https://github.com/wcarvalho/jaxneurorl libraries/jaxneurorl
git submodule add https://github.com/wcarvalho/JaxHouseMaze libraries/housemaze
git submodule add https://github.com/wcarvalho/nicewebrl libraries/nicewebrl
```

**Install**
```
mamba create -n preplay python=3.10 pip wheel -y
mamba activate preplay
pip install -e libraries/housemaze -e libraries/nicewebrl -e libraries/jaxneurorl -r requirements.txt
pip install -U jupyterlab matplotlib

# setting up mamba activation
mkdir -p $CONDA_PREFIX/etc/conda/activate.d/
echo 'export PYTHONPATH=$PYTHONPATH:`pwd`' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export GOOGLE_CREDENTIALS=keys/datastore-key.json' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

## Install (regular)

**adding submodules**

```
conda create -n preplay python=3.10 pip wheel -y
conda activate preplay

# otherwise, use github installs
pip install git+https://github.com/wcarvalho/jaxneurorl.git git+https://github.com/wcarvalho/JaxHouseMaze.git git+https://github.com/wcarvalho/nicewebrl.git -r requirements.txt

# setting up conda activation
echo 'export PYTHONPATH=$PYTHONPATH:`pwd`' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export GOOGLE_CREDENTIALS=keys/datastore-key.json' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

```




## Testing locally
```
python main.py
```

If you want to change an experiment, set `EXP=$number`. for debugging with a constant seed and max 1 episode per stage, set `DEBUG=1`. example:
```
DEBUG=1 EXP=2 python main.py 
```


## Deploying with fly.io

I followed [these instructions](https://github.com/zauberzeug/nicegui/wiki/fly.io-Deployment).

1. Install [flyctl](https://fly.io/docs/flyctl/install/), the command line interface (CLI) to administer your fly.io deployment. I used `brew`, i.e. `brew install flyctl` with a mac.
2. Create an account. With the terminal, you can use command `fly auth signup` or login with `fly auth login`.

### Custom urls for custom experiments

Right now, `main.py` accepts different experiment settings via environment variable `EXP`. We can leverage this to have custom URLs per experiments as follows:
```sh
# create app
flyctl launch \
--dockerfile Dockerfile \
--name ${name} \
--config configs/${name}.toml \
--env EXP=${exp}

# deploy online
flyctl deploy --config ${name}.toml

```
1. use `--name` to name the app. using same name for the config file is easier for tracking.
2. use `--env` to set environment variables for the app

some examples:
```sh
# test experiment
flyctl launch \
--dockerfile Dockerfile \
--name human-dyna-test \
--config configs/human-dyna-test.toml \
--env EXP=0

flyctl deploy --config configs/human-dyna-test.toml

# experiment 1
flyctl launch \
--dockerfile Dockerfile \
--name human-dyna-1 \
--config configs/human-dyna-1.toml \
--env EXP=1 \
--vm-size 'shared-cpu-4x'

flyctl deploy --config configs/human-dyna-1.toml
```

### Deleting apps

```sh
flyctl apps destroy $name --yes

# example
flyctl apps destroy human-dyna-test --yes
```



## Setting up google cloud for storing/loading data

Create a [Google Service Account](https://console.cloud.google.com/iam-admin/serviceaccounts?) for accessing your Database. Select "CREATE SERVICE ACCOUNT", give it some name and the following permissions:
- Storage Object Creator (for uploading/saving data)
- Storage Object Viewer and Storage Object Admin (for viewing/downloading data)

- create a bucket using [this link](https://console.cloud.google.com/storage/). this will be used to store data.
- create a key using [this link](https://console.cloud.google.com/iam-admin/serviceaccounts/details/111959560397464491265/keys?project=human-web-rl). this will be used to authenticate uploading and downloading of data. store the key as `keys/datastore-key.json` and make `GOOGLE_CREDENTIALS` point to it. e.g. `export GOOGLE_CREDENTIALS=keys/datastore-key.json`. The Dockerfile assumes this.