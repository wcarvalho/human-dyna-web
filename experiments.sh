# delete local data (useful for testing)
rm -r data .nicegui;

########################################################
# Experiment 2
# no reversal of blocks
# evaluate objects ARE visible
##########################################
# debugging command to test
rm -r data/*r0-exp2-obj1-v2* .nicegui; INST=0 DEBUG=1 NMAN=1 EXP=2 REV=0 EVAL_OBJECTS=1 NAME='r0-exp2-obj1-v2' SEED=44 python main.py

# create the config for putting this online
flyctl launch \
--dockerfile Dockerfile \
--name human-dyna-r0-exp2-obj1-v2 \
--config configs/human-dyna-r0-exp2-obj1-v2.toml \
--env EXP=2 \
--env EVAL_OBJECTS=1 \
--env REV=0 \
--env NAME='r0-exp2-obj1-v2' \
--vm-size 'shared-cpu-4x'


# launch the website
flyctl deploy --config configs/human-dyna-r0-exp2-obj1-v2.toml

# add machine for each retgion
flyctl scale count 4 --config configs/human-dyna-r0-exp2-obj1-v2.toml --region iad,sea,lax,den

# to display status
flyctl logs --config configs/human-dyna-r0-exp2-obj1-v2.toml

# see machines
flyctl scale show --config configs/human-dyna-r0-exp2-obj1-v2.toml

########################################################
# Experiment 1
# no reversal of blocks
##########################################
# debugging command to test
rm -r data .nicegui; INST=0 DEBUG=1 NMAN=1 EXP=1 NAME='r0-v2' REV=0 SEED=44 python main.py

# create the config for putting this online
flyctl launch \
--dockerfile Dockerfile \
--name human-dyna-r0-v2 \
--config configs/human-dyna-r0-v3.toml \
--env EXP=1 \
--env REV=0 \
--env NAME='r0-v2' \
--vm-size 'shared-cpu-4x'


# launch the website
flyctl deploy --config configs/human-dyna-r0-v3.toml

# set maximum to 5 machines running
flyctl scale count 5 --config configs/human-dyna-r0-v3.toml

# to display status
flyctl logs --config configs/human-dyna-r0-v3.toml

##########################################
# Experiment 1
# reversal of blocks
##########################################
INST=0 DEBUG=1 EXP=1 NAME='r1' REV=1 SEED=43 python main.py

flyctl launch \
--dockerfile Dockerfile \
--name human-dyna-r1 \
--config configs/human-dyna-r1.toml \
--env EXP=1 \
--env REV=1 \
--env NAME='r1' \
--vm-size 'shared-cpu-4x'

flyctl scale count 10 --config configs/human-dyna-r1.toml

flyctl deploy --config configs/human-dyna-r1.toml
