# human-dyna-web
Web experiments for studying Human Dyna experiments


## Install

```
mamba create -n human-dyna-web python=3.10 pip wheel -y
mamba activate human-dyna-web
# optionally add   --no-deps if just need library and dependencies already installed
pip install -e libraries/housemaze -e libraries/fast-web-rl

```

## Testing locally
```
python main.py
```


## Deploying with fly.io

I followed [these instructions](https://github.com/zauberzeug/nicegui/wiki/fly.io-Deployment).

1. Install [flyctl](https://fly.io/docs/flyctl/install/), the command line interface (CLI) to administer your fly.io deployment. I used `brew`, i.e. `brew install flyctl` with a mac.
2. Create an account. With the terminal, you can use command `fly auth signup` or login with `fly auth login`.

### Custom urls for custom experiments

Right now, `main.py` accepts different experiment settings via environment variable `EXP`. We can leverage this to have custom URLs per experiments as follows:

1. `flyctl launch --name ${app_name} --config configs/${app_name}.toml --env EXP=${exp} --dockerfile Dockerfile`.
   1. use `--name` to name the app. using same name for the config file is easier for tracking.
   2. 1. use `--env` to set environment variables for the app

examples:
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