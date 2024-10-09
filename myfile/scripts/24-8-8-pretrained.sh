#!/bin/bash

# 普通版 dote
cd /mydata/DOTE/networking_envs/save_models/cmp_pretrained
python3 /mydata/DOTE/dote.py --ecmp_topo "Abilene-2-('0', '1')-('4', '6')" --paths_from sp --so_mode train --so_epochs 5 --so_batch_size 32 --opt_function MAXUTIL
# python3 /mydata/DOTE/dote.py --ecmp_topo GEANT-obj1 --paths_from sp --so_mode test --so_epochs 5 --so_batch_size 32 --opt_function MAXUTIL
# ———————————————————  begin —————————————————————————————————————
### choose two topo
# {source: "Abilene-2-('5', '8')-('6', '7')", target: "Abilene-2-('0', '1')-('4', '6')"}

### training phase

# pretrained model
cd /mydata/DOTE/networking_envs/save_models/cmp_pretrained
/mydata/miniconda3/envs/dote/bin/python /mydata/DOTE/dote.py --ecmp_topo "Abilene-2-('5', '8')-('6', '7')" --paths_from sp --so_mode train --so_epochs 5 --so_batch_size 32 --opt_function MAXUTIL

# tca # --ecmp_topo input target topo
cd /mydata/DOTE/networking_envs/save_models/cmp_pretrained/tca_saved
/mydata/miniconda3/envs/dote/bin/python /mydata/DOTE/tca_dote.py --ecmp_topo "Abilene-2-('0', '1')-('4', '6')" --paths_from sp --so_mode train --so_epochs 5 --so_batch_size 32 --opt_function MAXUTIL

### test on diff env

# test tca # 修改 dote.py func test 中的 `torch.load('model_dote_Abilene-squeeze-links-more1.pkl').to(device)`
cd /mydata/DOTE/networking_envs/save_models/cmp_pretrained/tca_saved
/mydata/miniconda3/envs/dote/bin/python /mydata/DOTE/dote.py --ecmp_topo "Abilene-2-('0', '1')-('4', '6')" --paths_from sp --so_mode test --so_epochs 1 --so_batch_size 32 --opt_function MAXUTIL


# pretrained model
cd /mydata/DOTE/networking_envs/save_models/cmp_pretrained/
/mydata/miniconda3/envs/dote/bin/python /mydata/DOTE/dote.py --ecmp_topo "Abilene-2-('0', '1')-('4', '6')" --paths_from sp --so_mode test --so_epochs 1 --so_batch_size 32 --opt_function MAXUTIL