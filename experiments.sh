########################################################
# Experiment 10 - CRAFTAX
##########################################

# LOCAL DEBUGGING: RUN
rm -r craftax_data .nicegui; DEBUG=1 \
  DATA_DIR='craftax_data' \
  GIVE_INSTRUCTIONS=1 \
  MANIPULATION="paths" \
  SAY_REUSE=1 \
  EVAL_SHOW_MAP=0 \
  DEBUG=0 \
  DUMMY_ENV=0 \
  python craftax_web_app.py


# parameters of interest
# MANIPULATION: {"paths", "juncture"}
# SAY_REUSE: {1, 0}
# EVAL_SHOW_MAP: {1, 0}

# (paths) tell reuse + no map
python launch_craftax.py crafting-v1-paths-r1-m0 --environment='craftax' --env MANIPULATION="paths" --env SAY_REUSE=1 --env EVAL_SHOW_MAP=0

# https://crafting-v1-paths-r1-m0.fly.dev
flyctl scale count 4 --config -a human-dyna-crafting-v1-paths-r1-m0.toml --region iad,sea,lax,den --yes
flyctl scale memory 32768 --config configs/human-dyna-craftax-crafting-v1-paths-r1-m0.toml

flyctl deploy --config configs/human-dyna-craftax-crafting-v1-paths-r1-m0.toml
flyctl logs --config configs/human-dyna-craftax-crafting-v1-paths-r1-m0.toml



python launch_craftax.py crafting-v1-paths-r0-m0 --environment='craftax' --env MANIPULATION="paths" --env SAY_REUSE=0 --env EVAL_SHOW_MAP=0
# https://crafting-v1-paths-r0-m0.fly.dev
flyctl deploy --config configs/crafting-v1-paths-r0-m0.toml

python launch_craftax.py crafting-v1-juncture-r1-m0 --environment='craftax' --env MANIPULATION="juncture" --env SAY_REUSE=1 --env EVAL_SHOW_MAP=0
# https://crafting-v1-juncture-r1-m0.fly.dev
flyctl deploy --config configs/crafting-v1-juncture-r1-m0.toml

python launch_craftax.py crafting-v1-juncture-r0-m0 --environment='craftax' --env MANIPULATION="juncture" --env SAY_REUSE=0 --env EVAL_SHOW_MAP=0
# https://crafting-v1-juncture-r0-m0.fly.dev
flyctl deploy --config configs/crafting-v1-juncture-r0-m0.toml

########################################################
# Experiment 9 - CRAFTAX
##########################################

rm -r craftax_data .nicegui; DEBUG=1 \
  JAX_COMPILATION_CACHE_DIR="/tmp/craftax_jax_cache" \
  DATA_DIR='craftax_data' \
  NAME='craftax_exp' \
  SEED=1 \
  WORLD_SEED=1 \
  python craftax_webapp.py

########################################################
# Experiment 8 - adding rest and randomizing
##########################################

rm -r data3 .nicegui; DEBUG=1 TIMER=5 TIME_LIMIT=5 FEEDBACK=0 EXP=4 MAN='plan' SAY_REUSE=1 TIMER=30 DATA_DIR='data3' python housemaze_webapp.py

flyctl logs --config configs/human-dyna-exp8-v1-r1-t0-plan.toml


#2. start: no timer, tell
python launch.py exp5-v3-r1-t0-start --env EXP=4 --env MAN="start" --env SAY_REUSE=1 --env TIMER=0 --env COND2_TRAIN=1
#https://human-dyna-exp5-v3-r1-t0-start.fly.dev
flyctl deploy --config configs/human-dyna-exp5-v3-r1-t0-start.toml

#3. plan: no timer, tell
python launch.py exp5-v2-r1-t0-plan --env EXP=4 --env MAN="plan" --env SAY_REUSE=1 --env TIMER=0
#https://human-dyna-exp5-v2-r1-t0-plan.fly.dev
flyctl deploy --config configs/human-dyna-exp5-v2-r1-t0-plan.toml

#4. plan: no timer, don't tell
python launch.py exp5-v2-r0-t0-plan --env EXP=4 --env MAN="plan" --env SAY_REUSE=0 --env TIMER=0
#https://human-dyna-exp5-v2-r0-t0-plan.fly.dev
flyctl deploy --config configs/human-dyna-exp5-v2-r0-t0-plan.toml

#1. paths: no timer, tell
python launch.py exp5-v2-r0-t0-paths --env EXP=4 --env MAN="paths" --env SAY_REUSE=0 --env TIMER=0
#https://human-dyna-exp5-v2-r0-t0-paths.fly.dev
flyctl deploy --config configs/human-dyna-exp5-v2-r0-t0-paths.toml

#5. shortcut: no timer tell
python launch.py exp5-v3-r1-t0-shortcut --env EXP=4 --env MAN="shortcut" --env SAY_REUSE=1 --env TIMER=0
#https://human-dyna-exp5-v3-r1-t0-shortcut.fly.dev
flyctl deploy --config configs/human-dyna-exp5-v3-r1-t0-shortcut.toml


########################################################
# Experiment 7 - manipulations done separately
##########################################
flyctl logs --config configs/human-dyna-exp5-v1-r1-t0-plan.toml

#1. paths: no timer, tell
python launch.py exp5-v1-r1-t0-paths --env EXP=4 --env MAN="paths" --env SAY_REUSE=1 --env TIMER=0
#https://human-dyna-exp5-v1-r1-t0-paths.fly.dev
flyctl deploy --config configs/human-dyna-exp5-v1-r1-t0-paths.toml

#2. start: no timer, tell
python launch.py exp5-v1-r1-t0-start --env EXP=4 --env MAN="start" --env SAY_REUSE=1 --env TIMER=0
#https://human-dyna-exp5-v1-r1-t0-start.fly.dev
flyctl deploy --config configs/human-dyna-exp5-v1-r1-t0-start.toml

#3. plan: no timer, tell
python launch.py exp5-v1-r1-t0-plan --env EXP=4 --env MAN="plan" --env SAY_REUSE=1 --env TIMER=0
#https://human-dyna-exp5-v1-r1-t0-plan.fly.dev
flyctl deploy --config configs/human-dyna-exp5-v1-r1-t0-plan.toml

#4. plan: no timer, don't tell
python launch.py exp5-v1-r0-t0-plan --env EXP=4 --env MAN="plan" --env SAY_REUSE=0 --env TIMER=0
#https://human-dyna-exp5-v1-r0-t0-plan.fly.dev
flyctl deploy --config configs/human-dyna-exp5-v1-r0-t0-plan.toml

#1. paths: no timer, tell
python launch.py exp5-v1-r1-t0-paths --env EXP=4 --env MAN="paths" --env SAY_REUSE=1 --env TIMER=0
#https://human-dyna-exp5-v1-r1-t0-paths.fly.dev
flyctl deploy --config configs/human-dyna-exp5-v1-r1-t0-paths.toml

#5. shortcut: no timer tell
python launch.py exp5-v1-r1-t0-shortcut --env EXP=4 --env MAN="shortcut" --env SAY_REUSE=1 --env TIMER=0
#https://human-dyna-exp5-v1-r1-t0-shortcut.fly.dev
flyctl deploy --config configs/human-dyna-exp5-v1-r1-t0-shortcut.toml


########################################################
# Experiment 6 - manipulations done separately
##########################################
rm -r data/*exp3-v1* .nicegui; INST=1 DEBUG=0 NMAN=0 EXP=4 NAME='exp3-v1' SEED=45 python housemaze_webapp.py

# debugging command to test
rm -r data .nicegui; INST=0 DEBUG=2 NTRAIN=1 EXP=4 MAN='plan' SAY_REUSE=0 SEED=44 python housemaze_webapp.py


# FULL command
rm -r data .nicegui; EXP=4 MAN='shortcut' SAY_REUSE=0 SEED=2764371760 python housemaze_webapp.py





#1. paths: no timer, tell
python launch.py exp4-v1-r1-t0-paths --env EXP=4 --env MAN="paths" --env SAY_REUSE=1 --env TIMER=0
#https://human-dyna-exp4-v1-r1-t0-paths.fly.dev
flyctl deploy --config configs/human-dyna-exp4-v1-r1-t0-paths.toml

#2. start: no timer, tell
python launch.py exp4-v1-r1-t0-start --env EXP=4 --env MAN="start" --env SAY_REUSE=1 --env TIMER=0
#https://human-dyna-exp4-v1-r1-t0-start.fly.dev
flyctl deploy --config configs/human-dyna-exp4-v1-r1-t0-start.toml

#3. plan: no timer, tell
python launch.py exp4-v1-r1-t0-plan --env EXP=4 --env MAN="plan" --env SAY_REUSE=1 --env TIMER=0
#https://human-dyna-exp4-v1-r1-t0-plan.fly.dev
flyctl deploy --config configs/human-dyna-exp4-v1-r1-t0-plan.toml

#4. plan: no timer, don't tell
python launch.py exp4-v1-r0-t0-plan --env EXP=4 --env MAN="plan" --env SAY_REUSE=0 --env TIMER=0
#https://human-dyna-exp4-v1-r0-t0-plan.fly.dev
flyctl deploy --config configs/human-dyna-exp4-v1-r0-t0-plan.toml

#5. shortcut: no timer tell
python launch.py exp4-v1-r1-t0-shortcut --env EXP=4 --env MAN="shortcut" --env SAY_REUSE=1 --env TIMER=0  --env FEEDBACK=1
#https://human-dyna-exp4-v1-r1-t0-shortcut.fly.dev
flyctl deploy --config configs/human-dyna-exp4-v1-r1-t0-shortcut.toml


flyctl logs --config configs/human-dyna-exp4-v1-r0-t0-plan.toml
########################################################
# Experiment 5 - three conditions, new map
##########################################
rm -r data/*exp3-v1* .nicegui; INST=1 DEBUG=0 NMAN=0 EXP=3 NAME='exp3-v1' SEED=45 python main.py

# debugging command to test
rm -r data .nicegui; INST=1 DEBUG=1 NMAN=0 EXP=3 SAY_REUSE=0 TIMER=0 NAME='exp3-v1' SEED=44 python main.py

python launch.py exp3-v5-r1-t0 --env EXP=3 --env SAY_REUSE=1 --env TIMER=0
https://human-dyna-exp3-v5-r1-t0.fly.dev
flyctl deploy --config configs/human-dyna-exp3-v5-r1-t0.toml

python launch.py exp3-v5-r0-t30 --env EXP=3 --env SAY_REUSE=0 --env TIMER=30
https://human-dyna-exp3-v5-r0-t30.fly.dev
flyctl deploy --config configs/human-dyna-exp3-v5-r0-t30.toml


python launch.py exp3-v5-r1-t30 --env EXP=3 --env SAY_REUSE=1 --env TIMER=30
https://human-dyna-exp3-v5-r1-t30.fly.dev
flyctl deploy --config configs/human-dyna-exp3-v5-r1-t30.toml



########################################################
# Experiment 4 - no reuse or no timer
##########################################

python launch.py exp3-v3-r1-t0 --env EXP=3 --env SAY_REUSE=1 --env TIMER=0
https://human-dyna-exp3-v3-r1-t30.fly.dev
flyctl deploy --config configs/human-dyna-exp3-v3-r1-t0.toml

python launch.py exp3-v3-r0-t30 --env EXP=3 --env SAY_REUSE=0 --env TIMER=30
https://human-dyna-exp3-v3-r0-t30.fly.dev
flyctl deploy --config configs/human-dyna-exp3-v3-r0-t30.toml


python launch.py exp3-v2-r1-t30 --env EXP=3 --env SAY_REUSE=1 --env TIMER=30
https://human-dyna-exp3-v2-r1-t30.fly.dev
flyctl deploy --config configs/human-dyna-exp3-v2-r1-t30.toml

########################################################
# Experiment 3 - launches
##########################################


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
