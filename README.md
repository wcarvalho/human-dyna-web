# human-dyna-web
Web experiments for studying Human Dyna experiments


## Install

```
mamba create -n human-dyna-web python=3.10 pip wheel -y
mamba activate human-dyna-web
# optionally add   --no-deps if just need library and dependencies already installed
pip install -e libraries/housemaze -e libraries/fast-web-rl

```