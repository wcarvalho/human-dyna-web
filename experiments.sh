# delete local data (useful for testing)
rm -r data .nicegui;
########################################################
# Experiment 4 - no reuse or no timer
##########################################
## FIXED NAME
python launch.py exp3-v3-r1-t0 --env EXP=3 --env SAY_REUSE=1 --env TIMER=0
https://human-dyna-exp3-v3-r1-t30.fly.dev
flyctl deploy --config configs/human-dyna-exp3-v3-r1-t30.toml

python launch.py exp3-v3-r0-t30 --env EXP=3 --env SAY_REUSE=0 --env TIMER=30
https://human-dyna-exp3-v3-r0-t30.fly.dev
flyctl deploy --config configs/human-dyna-exp3-v3-r0-t30.toml


########################################################
# Experiment 3 - launches
##########################################
## FIXED NAME
python launch.py exp3-v2-r1-t30 --env EXP=3 --env SAY_REUSE=1 --env TIMER=30
https://human-dyna-exp3-v2-r1-t30.fly.dev
flyctl deploy --config configs/human-dyna-exp3-v2-r1-t30.toml

python launch.py exp3-v2-r0-t30 --env EXP=3 --env SAY_REUSE=0 --env TIMER=30
https://human-dyna-exp3-v2-r0-t30.fly.dev
flyctl deploy --config configs/human-dyna-exp3-v2-r0-t30.toml


## FIRST RUN
python launch.py exp3-v1-r1-t45 --env EXP=3 --env SAY_REUSE=1 --env TIMER=45
https://exp3-v1-r1-t45.fly.dev
flyctl deploy --config configs/human-dyna-exp3-v1-r1-t45.toml

python launch.py exp3-v1-r0-t45 --env EXP=3 --env SAY_REUSE=0 --env TIMER=45
https://exp3-v1-r0-t45.fly.dev
flyctl deploy --config configs/human-dyna-exp3-v1-r0-t45.toml

python launch.py exp3-v1-r1-t30 --env EXP=3 --env SAY_REUSE=1 --env TIMER=30
https://exp3-v1-r1-t30.fly.dev
flyctl deploy --config configs/human-dyna-exp3-v1-r1-t30.toml

python launch.py exp3-v1-r0-t30 --env EXP=3 --env SAY_REUSE=0 --env TIMER=30
https://exp3-v1-r0-t30.fly.dev
flyctl deploy --config configs/human-dyna-exp3-v1-r0-t30.toml

#######################
# Experiment 3 - testing
#######################
# FULL debugging command to test
rm -r data/*exp3-v1* .nicegui; INST=1 DEBUG=0 NMAN=0 EXP=3 NAME='exp3-v1' SEED=45 python main.py

# debugging command to test
rm -r data .nicegui; INST=1 DEBUG=1 NMAN=0 EXP=3 SAY_REUSE=0 TIMER=0 NAME='exp3-v1' SEED=44 python main.py

# create the config for putting this online
# added planning manipulation
python launch.py exp3-v3 --env EXP=3 --env EVAL_OBJECTS=1 --env REV=0

# to display status
flyctl logs --config configs/human-dyna-exp3-v3.tom1

# see machines
flyctl scale show --config configs/human-dyna-exp3-v3.tom1

# delete machines
flyctl machine destroy --config configs/human-dyna-exp3-v3.tom1



########################################################
# Experiment 2 - launches
##########################################

# added planning manipulation
python launch.py exp2-v3 --env EXP=2 --env NAME=exp2-v3


#######################
# Experiment 2 - testing
#######################
# FULL debugging command to test
rm -r data/*exp2-v3* .nicegui; INST=1 DEBUG=0 NMAN=0 EXP=2 NAME='exp2-v3' SEED=45 python main.py

# debugging command to test
rm -r data/*exp2-v3* .nicegui; INST=0 DEBUG=1 NMAN=1 EXP=2 NAME='exp2-v3' SEED=44 python main.py

# create the config for putting this online
# added planning manipulation
python launch.py exp2-v3 --env EXP=2 --env EVAL_OBJECTS=1 --env REV=0 --env NAME=exp2-v3

# to display status
flyctl logs --config configs/human-dyna-exp2-v3.toml

# see machines
flyctl scale show --config configs/human-dyna-exp2-v3.toml

# delete machines
flyctl machine destroy --config configs/human-dyna-exp2-v3.toml

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
