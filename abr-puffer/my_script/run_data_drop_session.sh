#!/bin/bash
# Using trained model on 6 month to do inference on 8 month data.
# wget -O - https://raw.githubusercontent.com/KFCDeveloper/OnlineScripts/main/script_CausalSim/drop_data/drop_data.sh | bash
# copy directory CAUSALSIM_DIR-20-9-27-6/{cooked, subset_data} to CAUSALSIM_DIR-20-9-27-6monthmodel/{cooked, subset_data}
# copy {2020-11-27to2021-06-01_bola_basic_v1_trained_models, 2020-11-27to2021-06-01_bola_basic_v2_trained_models, 2020-11-27to2021-06-01_linear_bba_trained_models} to CAUSALSIM_DIR-20-9-27-6monthmodel
# change name (above three) 2020-11-27to2021-06-01* to 2020-09-27to2021-06-01*

# 先把 tune c 之前的东西全部跑完，需要跑完所有的 c 才行


cd /mydata/Unbiased-Trace-Driven-Simulation/abr-puffer
targets=("bola_basic_v1" "bola_basic_v2" "linear_bba")  # "bola_basic_v1" "bola_basic_v2" "linear_bba"
# root_dirs=("CAUSALSIM_DIR-dis-20-7-27/" "CAUSALSIM_DIR-dis-20-8-27/" "CAUSALSIM_DIR-dis-20-9-27/" "CAUSALSIM_DIR-dis-20-10-27/" "CAUSALSIM_DIR-dis-20-11-27/" "CAUSALSIM_DIR-dis-20-12-27/" "CAUSALSIM_DIR-dis-21-1-27/" "CAUSALSIM_DIR-dis-21-2-27/")   # "CAUSALSIM_DIR/" "CAUSALSIM_DIR-20-9-27/" "CAUSALSIM_DIR-20-11-27/" 
root_dirs=("CAUSALSIM_DIR-dis-21-3-27/" "CAUSALSIM_DIR-dis-21-4-27/")
c_s=("0.05" "0.1" "0.5" "1.0" "5.0" "10.0" "15.0" "20.0" "25.0" "30.0" "40.0") #"0.05" "0.1" "0.5" "1.0" "5.0" "10.0" "15.0" "20.0" "25.0" "30.0" "40.0"   # "0.05"  "0.1" "0.5"
# !! for each dir, need to run `python data_preparation/generate_subset_data.py --dir "$root_dir"`
for c in "${c_s[@]}"
do  
    # 去掉 c 中的小数点
    c_clean=$(echo "$c" | tr -d '.')
    for target in "${targets[@]}"
    do
        for root_dir in "${root_dirs[@]}"
        do  
            # 定义 tmux 会话名称
            session_name="${c_clean}_${target}_${root_dir}"

            # !!! 创建新的 tmux 会话并在其中运行第一个命令  有的时候需要注销掉
            tmux new-session -d -s "$session_name"
            tmux send-keys -t "$session_name" "cd /mydata/Unbiased-Trace-Driven-Simulation/abr-puffer" C-m
            # train CasualSim
            # /mydata/miniconda3/envs/casual_sim_abr/bin/python training/train_subset.py --dir "$root_dir" --left_out_policy "$target" --C "$c" 

            ## Training
            #  Using CausalSim to extract and save the latent factors ** need
            tmux send-keys -t "$session_name" "/mydata/miniconda3/envs/casual_sim_abr/bin/python inference/extract_subset_latents.py --dir $root_dir --left_out_policy $target --C $c" C-m
            # Counterfactual Simulation
            tmux send-keys -t "$session_name" "/mydata/miniconda3/envs/casual_sim_abr/bin/python inference/expert_cfs.py --dir $root_dir" C-m # ExpertSim ** need  (pretty time-consuming)
            tmux send-keys -t "$session_name" "/mydata/miniconda3/envs/casual_sim_abr/bin/python inference/buffer_subset_cfs.py --dir $root_dir --left_out_policy $target --C $c" C-m # CausalSim (pretty time-consuming)
            # Calculate the average SSIM using the ground-truth data
            tmux send-keys -t "$session_name" "/mydata/miniconda3/envs/casual_sim_abr/bin/python analysis/original_subset_ssim.py --dir $root_dir --left_out_policy $target" C-m # ExpertSim ** need
            tmux send-keys -t "$session_name" "/mydata/miniconda3/envs/casual_sim_abr/bin/python analysis/subset_ssim.py --dir $root_dir --left_out_policy $target --C $c" C-m # CausalSim ** need
            # Calculate the simulated buffer distribution's Earth Mover Distance (EMD) using the fround-truth data
            tmux send-keys -t "$session_name" "/mydata/miniconda3/envs/casual_sim_abr/bin/python analysis/subset_EMD.py --dir $root_dir --left_out_policy $target --C $c" C-m # All{ExpertSim, SLSim, CausalSim} ** need
            # Tune CausalSim's hyper-parameters for buffer and SSIM prediction
            # /mydata/miniconda3/envs/casual_sim_abr/bin/python analysis/tune_buffer_hyperparams.py --dir "$root_dir"

            # # CausalSim model to generate counterfactual downloadtime trajectories
            # /mydata/miniconda3/envs/casual_sim_abr/bin/python inference/downloadtime_subset_cfs.py --dir "$root_dir" --left_out_policy "$target" --C "$c"  # (pretty time-consuming)
            # # calculate the average stall ratio
            # /mydata/miniconda3/envs/casual_sim_abr/bin/python analysis/original_subset_stall.py --dir "$root_dir" --left_out_policy "$target" # ExpertSim 
            # /mydata/miniconda3/envs/casual_sim_abr/bin/python analysis/subset_stall.py --dir "$root_dir" --left_out_policy "$target" --C "$c" # CausalSim 
            # # Tune CausalSim's hyper-parameters for downloadtime prediction
            # # /mydata/miniconda3/envs/casual_sim_abr/bin/python analysis/tune_downloadtime_hyperparameters.py --dir "$root_dir"  
        done
    done
done

# Tune CausalSim's hyper-parameters for buffer and SSIM prediction
# /mydata/miniconda3/envs/casual_sim_abr/bin/python analysis/tune_buffer_hyperparams.py --dir "$root_dir"

# ————————————————————————————————————
# for c in "${c_s[@]}"
# do  
#     # 去掉 c 中的小数点
#     c_clean=$(echo "$c" | tr -d '.')
#     for target in "${targets[@]}"
#     do
#         for root_dir in "${root_dirs[@]}"
#         do  
#             # 定义 tmux 会话名称
#             session_name="${c_clean}_${target}_${root_dir}"

#             # !!! 创建新的 tmux 会话并在其中运行第一个命令  有的时候需要注销掉
#             # tmux new-session -d -s "$session_name"
#             tmux send-keys -t "$session_name" "cd /mydata/Unbiased-Trace-Driven-Simulation/abr-puffer" C-m
#             # CausalSim model to generate counterfactual downloadtime trajectories
#             tmux send-keys -t "$session_name" "/mydata/miniconda3/envs/casual_sim_abr/bin/python inference/downloadtime_subset_cfs.py --dir $root_dir --left_out_policy $target --C $c" C-m # (pretty time-consuming)
#             # calculate the average stall ratio
#             tmux send-keys -t "$session_name" "/mydata/miniconda3/envs/casual_sim_abr/bin/python analysis/original_subset_stall.py --dir $root_dir --left_out_policy $target" C-m # ExpertSim 
#             tmux send-keys -t "$session_name" "/mydata/miniconda3/envs/casual_sim_abr/bin/python analysis/subset_stall.py --dir $root_dir --left_out_policy $target --C $c" C-m # CausalSim 
#             # Tune CausalSim's hyper-parameters for downloadtime prediction
#             # /mydata/miniconda3/envs/casual_sim_abr/bin/python analysis/tune_downloadtime_hyperparams.py --dir "$root_dir"  
#         done
#     done
# done
# ——————————————————————————————————————————————————————————————————
# Tune CausalSim's hyper-parameters for downloadtime prediction
# /mydata/miniconda3/envs/casual_sim_abr/bin/python analysis/tune_downloadtime_hyperparams.py --dir "$root_dir"  