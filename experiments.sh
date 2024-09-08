# delete local data (useful for testing)
rm -r data .nicegui;

########################################################
# Experiment 1
# no reversal of blocks
##########################################
# debugging command to test
INST=0 DEBUG=1 EXP=1 NAME='r0' REV=0 SEED=43 python main.py

# create the config for putting this online
flyctl launch \
--dockerfile Dockerfile \
--name human-dyna-1-r0 \
--config configs/human-dyna-1-r0.toml \
--env EXP=1 \
--env REV=0 \
--env NAME='r0' \
--vm-size 'shared-cpu-4x'

# set maximum to 10 machines running
flyctl scale count 5 --config configs/human-dyna-1-r0.toml

# launch the website
flyctl deploy --config configs/human-dyna-1-r0.toml

# to display status

##########################################
# Experiment 1
# reversal of blocks
##########################################
INST=0 DEBUG=1 EXP=1 NAME='r1' REV=1 SEED=43 python main.py

flyctl launch \
--dockerfile Dockerfile \
--name human-dyna-1-r1 \
--config configs/human-dyna-1-r1.toml \
--env EXP=1 \
--env REV=1 \
--env NAME='r1' \
--vm-size 'shared-cpu-4x'

flyctl scale count 10 --config configs/human-dyna-1-r1.toml

flyctl deploy --config configs/human-dyna-1-r1.toml
