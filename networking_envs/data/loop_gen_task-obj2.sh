#!/bin/bash
# some error: not source ~/.bashrc(no env variable); not activate dote;
# /mydata/DOTE/networking_envs/data/loop_gen_task-obj2.sh > /mydata/DOTE/networking_envs/data/gen_log/obj2-4.log
# change `/mydata/DOTE/networking_envs/data/loop_gml_to_dote.py` line 14 from "-2-" to "-obj2-2-"
# conda activate dote
edges=("('0', '1')" "('0', '2')" "('1', '10')" "('2', '9')" "('3', '4')" "('3', '6')" "('4', '5')" "('4', '6')" "('5', '8')" "('6', '7')" "('7', '8')" "('7', '10')" "('8', '9')" "('9', '10')")
for ((i=9; i<13; i++)); # i=0; i<${#edges[@]};  0 2; 2 5;5 9;9 13
do
    for ((j=i+1; j<${#edges[@]}; j++));
    do  
        cd /mydata/DOTE/networking_envs/data/
        /mydata/miniconda3/envs/dote/bin/python loop_gml_to_dote.py "Abilene" "${edges[i]}" "${edges[j]}"
        # don't forget!! To compute the optimum for the demand matrices, go to /mydata/DOTE/networking_envs/data/Abilene and run /mydata/DOTE/networking_envs/data/compute_opts.py
        cd "/mydata/DOTE/networking_envs/data/Abilene-obj2-2-${edges[i]}-${edges[j]}"
        /mydata/miniconda3/envs/dote/bin/python /mydata/DOTE/networking_envs/data/loop_compute_opts.py "Abilene-obj2-2-${edges[i]}-${edges[j]}"
    done
done
/mydata/miniconda3/envs/dote/bin/python /mydata/DOTE/networking_envs/data/loop_compute_opts.py "Abilene-obj2-2-('0', '1')-('0', '2')"
############## reduce single link
# edges=("('0', '1')" "('0', '2')" "('1', '10')" "('2', '9')" "('3', '4')" "('3', '6')" "('4', '5')" "('4', '6')" "('5', '8')" "('6', '7')" "('7', '8')" "('7', '10')" "('8', '9')" "('9', '10')")
# for ((i=0; i<13; i++)); # i=0; i<${#edges[@]};  0 2; 2 5;5 9;9 13
# do
#     cd /mydata/DOTE/networking_envs/data/
#     /mydata/miniconda3/envs/dote/bin/python loop_gml_to_dote.py "Abilene" "${edges[i]}"  # 这里不要加 -num-
#     # don't forget!! To compute the optimum for the demand matrices, go to /mydata/DOTE/networking_envs/data/Abilene and run /mydata/DOTE/networking_envs/data/compute_opts.py
#     cd "/mydata/DOTE/networking_envs/data/Abilene-1-${edges[i]}"
#     /mydata/miniconda3/envs/dote/bin/python /mydata/DOTE/networking_envs/data/loop_compute_opts.py "Abilene-1-${edges[i]}"
# done

############## generate other two different topo
# /mydata/miniconda3/envs/dote/bin/python loop_gml_to_dote.py "Arnes" "('1', '23')"
# /mydata/miniconda3/envs/dote/bin/python /mydata/DOTE/networking_envs/data/loop_compute_opts.py "Arnes-1-('7', '23')"

# /mydata/miniconda3/envs/dote/bin/python loop_gml_to_dote.py "Abilene0" "('1', '10')"
# /mydata/miniconda3/envs/dote/bin/python /mydata/DOTE/networking_envs/data/loop_compute_opts.py "Abilene0-1-('1', '10')"
