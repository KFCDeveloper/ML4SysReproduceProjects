/mydata/miniconda3/envs/dote/bin/python /mydata/DOTE/networking_envs/data/loop_compute_opts.py "Abilene-obj2-2-('0', '1')-('0', '2')"

tmux new -s gendis05
conda activate dote
cd /mydata/DOTE/networking_envs/data/
/mydata/miniconda3/envs/dote/bin/python loop_gml_to_dote.py "Abilene" "('0', '1')" "('0', '2')"
# don't forget!! To compute the optimum for the demand matrices, go to /mydata/DOTE/networking_envs/data/Abilene and run /mydata/DOTE/networking_envs/data/compute_opts.py
cd "/mydata/DOTE/networking_envs/data/Abilene-obj2-2-('0', '1')-('0', '2')"
/mydata/miniconda3/envs/dote/bin/python /mydata/DOTE/networking_envs/data/loop_compute_opts.py "Abilene-obj2-2-('0', '1')-('0', '2')"

tmux new -s gendis025
conda activate dote
cd /mydata/DOTE/networking_envs/data/
/mydata/miniconda3/envs/dote/bin/python loop_gml_to_dote.py "Abilene" "('5', '8')" "('6', '7')"
# don't forget!! To compute the optimum for the demand matrices, go to /mydata/DOTE/networking_envs/data/Abilene and run /mydata/DOTE/networking_envs/data/compute_opts.py
cd "/mydata/DOTE/networking_envs/data/Abilene-obj2-2-('5', '8')-('6', '7')"
/mydata/miniconda3/envs/dote/bin/python /mydata/DOTE/networking_envs/data/loop_compute_opts.py "Abilene-obj2-2-('5', '8')-('6', '7')"

tmux new -s gendis10
conda activate dote
cd /mydata/DOTE/networking_envs/data/
/mydata/miniconda3/envs/dote/bin/python loop_gml_to_dote.py "Abilene" "('5', '8')" "('6', '7')"
# don't forget!! To compute the optimum for the demand matrices, go to /mydata/DOTE/networking_envs/data/Abilene and run /mydata/DOTE/networking_envs/data/compute_opts.py
cd "/mydata/DOTE/networking_envs/data/Abilene-obj2-dis10-2-('5', '8')-('6', '7')"
/mydata/miniconda3/envs/dote/bin/python /mydata/DOTE/networking_envs/data/loop_compute_opts.py "Abilene-obj2-dis10-2-('5', '8')-('6', '7')"

tmux new -s gendis125
cd "/mydata/DOTE/networking_envs/data/Abilene-obj2-dis125-2-('5', '8')-('6', '7')"
/mydata/miniconda3/envs/dote/bin/python /mydata/DOTE/networking_envs/data/loop_compute_opts.py "Abilene-obj2-dis125-2-('5', '8')-('6', '7')"