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
3. Run `fly launch` to create the `fly.toml` file.
4. Run `fly deploy` to launch online.



## Setting up google cloud for storing/loading data

Create a [Google Service Account](https://console.cloud.google.com/iam-admin/serviceaccounts?) for accessing your Database. Select "CREATE SERVICE ACCOUNT", give it some name and the following permissions:
- Storage Object Creator (for uploading/saving data)
- Storage Object Viewer and Storage Object Admin (for viewing/downloading data)

- create a bucket using [this link](https://console.cloud.google.com/storage/). this will be used to store data.
- create a key using [this link](https://console.cloud.google.com/iam-admin/serviceaccounts/details/111959560397464491265/keys?project=human-web-rl). this will be used to authenticate uploading and downloading of data. store the key somewhere and make `GOOGLE_CREDENTIALS` point to it. e.g. `export GOOGLE_CREDENTIALS=path/to/key.json`.