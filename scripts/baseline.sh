#!/bin/bash
. /etc/profile.d/lmod.sh
module use /pkgs/environment-modules/
python models/baseline.py --config config/nlg_lin/baseline.yaml --config_name all --checkpoint_dir results/nlg/baseline

