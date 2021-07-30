#!/bin/bash
. /etc/profile.d/lmod.sh
module use /pkgs/environment-modules/
python models/trainers.py --config config/nlg_lin/config.yaml --config_name all --checkpoint_dir results/nlg/all

