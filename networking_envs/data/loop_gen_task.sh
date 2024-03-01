#!/bin/bash
# /data/ydy/myproject/DOTE/networking_envs/data/loop_gen_task.sh > /data/ydy/myproject/DOTE/networking_envs/data/gen_log/3.log
# conda activate ydy-dote
edges=("('0', '1')" "('0', '2')" "('1', '10')" "('2', '9')" "('3', '4')" "('3', '6')" "('4', '5')" "('4', '6')" "('5', '8')" "('6', '7')" "('7', '8')" "('7', '10')" "('8', '9')" "('9', '10')")
# for ((i=9; i<13; i++)); # i=0; i<${#edges[@]};  0 2; 2 5;5 9;9 13
# do
#     for ((j=i+1; j<${#edges[@]}; j++));
#     do  
#         cd /data/ydy/myproject/DOTE/networking_envs/data/
#         /home/amax/.conda/envs/ydy-dote/bin/python loop_gml_to_dote.py "Abilene" "${edges[i]}" "${edges[j]}"
#         # don't forget!! To compute the optimum for the demand matrices, go to /mydata/DOTE/networking_envs/data/Abilene and run /mydata/DOTE/networking_envs/data/compute_opts.py
#         cd "/data/ydy/myproject/DOTE/networking_envs/data/Abilene-${edges[i]}-${edges[j]}"
#         /home/amax/.conda/envs/ydy-dote/bin/python /data/ydy/myproject/DOTE/networking_envs/data/loop_compute_opts.py "Abilene-${edges[i]}-${edges[j]}"
#     done
# done

for ((i=0; i<13; i++)); # i=0; i<${#edges[@]};  0 2; 2 5;5 9;9 13
do
    cd /data/ydy/myproject/DOTE/networking_envs/data/
    /home/amax/.conda/envs/ydy-dote/bin/python loop_gml_to_dote.py "Abilene" "${edges[i]}"  # 这里不要加 -num-
    # don't forget!! To compute the optimum for the demand matrices, go to /mydata/DOTE/networking_envs/data/Abilene and run /mydata/DOTE/networking_envs/data/compute_opts.py
    cd "/data/ydy/myproject/DOTE/networking_envs/data/Abilene-1-${edges[i]}"
    /home/amax/.conda/envs/ydy-dote/bin/python /data/ydy/myproject/DOTE/networking_envs/data/loop_compute_opts.py "Abilene-1-${edges[i]}"
done
