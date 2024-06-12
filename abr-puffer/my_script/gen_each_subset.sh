#!/bin/bash
# Using trained model on 6 month to do inference on 8 month data.
# wget -O - https://raw.githubusercontent.com/KFCDeveloper/OnlineScripts/main/script_CausalSim/drop_data/drop_data.sh | bash
# copy directory CAUSALSIM_DIR-20-9-27-6/{cooked, subset_data} to CAUSALSIM_DIR-20-9-27-6monthmodel/{cooked, subset_data}
# copy {2020-11-27to2021-06-01_bola_basic_v1_trained_models, 2020-11-27to2021-06-01_bola_basic_v2_trained_models, 2020-11-27to2021-06-01_linear_bba_trained_models} to CAUSALSIM_DIR-20-9-27-6monthmodel
# change name (above three) 2020-11-27to2021-06-01* to 2020-09-27to2021-06-01*

# 先把 tune c 之前的东西全部跑完，需要跑完所有的 c 才行


cd /mydata/Unbiased-Trace-Driven-Simulation/abr-puffer
targets=("bola_basic_v1" "bola_basic_v2" "linear_bba")  # "bola_basic_v1" "bola_basic_v2" "linear_bba"
# root_dirs=("CAUSALSIM_DIR-dis-20-7-27/" "CAUSALSIM_DIR-dis-20-8-27/" "CAUSALSIM_DIR-dis-20-9-27/" "CAUSALSIM_DIR-dis-20-10-27/" "CAUSALSIM_DIR-dis-20-11-27/" "CAUSALSIM_DIR-dis-20-12-27/" "CAUSALSIM_DIR-dis-21-1-27/" "CAUSALSIM_DIR-dis-21-2-27/")
root_dirs=("CAUSALSIM_DIR-dis-21-3-27/" "CAUSALSIM_DIR-dis-21-4-27/")

c_s=("0.05" "0.1" "0.5" "1.0" "5.0" "10.0" "15.0" "20.0" "25.0" "30.0" "40.0") #"0.05" "0.1" "0.5" "1.0" "5.0" "10.0" "15.0" "20.0" "25.0" "30.0" "40.0"   # "0.05"  "0.1" "0.5"
# !! for each dir, need to run `python data_preparation/generate_subset_data.py --dir "$root_dir"`



for root_dir in "${root_dirs[@]}"
do  
    # 定义 tmux 会话名称
    session_name="gen_subset_${root_dir}"

    # 创建新的 tmux 会话并在其中运行第一个命令
    tmux new-session -d -s "$session_name"
    tmux send-keys -t "$session_name" "cd /mydata/Unbiased-Trace-Driven-Simulation/abr-puffer" C-m
    # train CasualSim
    # /mydata/miniconda3/envs/casual_sim_abr/bin/python training/train_subset.py --dir "$root_dir" --left_out_policy "$target" --C "$c" 
    tmux send-keys -t "$session_name" "/mydata/miniconda3/envs/casual_sim_abr/bin/python data_preparation/generate_subset_data.py --dir $root_dir" C-m
done