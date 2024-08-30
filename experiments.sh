# delete local data for testin
rm -r data .nicegui;

# Experiment 1
# no reversal of blocks
INST=0 DEBUG=1 EXP=1 REV=0 SEED=43 python main.py

flyctl launch \
--dockerfile Dockerfile \
--name human-dyna-1 \
--config configs/human-dyna-1.toml \
--env EXP=1 \
--vm-size 'shared-cpu-4x'

flyctl deploy --config configs/human-dyna-1.toml


# Experiment 1
# reversal of blocks

INST=0 DEBUG=1 EXP=1 REV=1 SEED=43 python main.py

flyctl launch \
--dockerfile Dockerfile \
--name human-dyna-1 \
--config configs/human-dyna-1r.toml \
--env EXP=1 REV=1 \
--vm-size 'shared-cpu-4x'

flyctl deploy --config configs/human-dyna-1r.toml
