#!/bin/bash

# 普通版 dote
cd /mydata/DOTE/networking_envs/save_models/cmp_pretrained
python3 /mydata/DOTE/dote.py --ecmp_topo GEANT-obj1 --paths_from sp --so_mode test --so_epochs 5 --so_batch_size 32 --opt_function MAXUTIL

