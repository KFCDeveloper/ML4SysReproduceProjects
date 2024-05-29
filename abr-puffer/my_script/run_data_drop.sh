#!/bin/bash
# Using trained model on 6 month to do inference on 8 month data.
# wget -O - https://raw.githubusercontent.com/KFCDeveloper/OnlineScripts/main/script_CausalSim/drop_data/drop_data.sh | bash
# copy directory CAUSALSIM_DIR-20-9-27-6/{cooked, subset_data} to CAUSALSIM_DIR-20-9-27-6monthmodel/{cooked, subset_data}
# copy {2020-11-27to2021-06-01_bola_basic_v1_trained_models, 2020-11-27to2021-06-01_bola_basic_v2_trained_models, 2020-11-27to2021-06-01_linear_bba_trained_models} to CAUSALSIM_DIR-20-9-27-6monthmodel
# change name (above three) 2020-11-27to2021-06-01* to 2020-09-27to2021-06-01*

cd /mydata/Unbiased-Trace-Driven-Simulation/abr-puffer
targets=("bola_basic_v1" "bola_basic_v2" "linear_bba")  # "linear_bba"
root_dirs=("CAUSALSIM_DIR-20-9-27to20-11-27_total_view/")   # "CAUSALSIM_DIR/" "CAUSALSIM_DIR-20-9-27/" "CAUSALSIM_DIR-20-11-27/" 
c_s=( "0.05" ) # "0.1" "0.5" "1.0" "5.0" "10.0" "15.0" "20.0" "25.0" "30.0" "40.0"   # "0.05"
# !! for each dir, need to run `python data_preparation/generate_subset_data.py --dir "$root_dir"`
for target in "${targets[@]}"
do
    for root_dir in "${root_dirs[@]}"
    do  
        for c in "${c_s[@]}"
        do
            # train CasualSim
            # /mydata/miniconda3/envs/casual_sim_abr/bin/python training/train_subset.py --dir "$root_dir" --left_out_policy "$target" --C "$c" 

            ## Training
            #  Using CausalSim to extract and save the latent factors ** need
            /mydata/miniconda3/envs/casual_sim_abr/bin/python inference/extract_subset_latents.py --dir "$root_dir" --left_out_policy "$target" --C "$c" 
            # Counterfactual Simulation
            /mydata/miniconda3/envs/casual_sim_abr/bin/python inference/expert_cfs.py --dir "$root_dir" # ExpertSim ** need  (pretty time-consuming)
            /mydata/miniconda3/envs/casual_sim_abr/bin/python inference/buffer_subset_cfs.py --dir "$root_dir" --left_out_policy "$target" --C "$c" # CausalSim (pretty time-consuming)
            # Calculate the average SSIM using the ground-truth data
            /mydata/miniconda3/envs/casual_sim_abr/bin/python analysis/original_subset_ssim.py --dir "$root_dir" --left_out_policy "$target" # ExpertSim ** need
            /mydata/miniconda3/envs/casual_sim_abr/bin/python analysis/subset_ssim.py --dir "$root_dir" --left_out_policy "$target" --C "$c" # CausalSim ** need
            # Calculate the simulated buffer distribution's Earth Mover Distance (EMD) using the fround-truth data
            /mydata/miniconda3/envs/casual_sim_abr/bin/python analysis/subset_EMD.py --dir "$root_dir" --left_out_policy "$target" --C "$c" # All{ExpertSim, SLSim, CausalSim} ** need
            # Tune CausalSim's hyper-parameters for buffer and SSIM prediction
            # /mydata/miniconda3/envs/casual_sim_abr/bin/python analysis/tune_buffer_hyperparams.py --dir "$root_dir"

            # CausalSim model to generate counterfactual downloadtime trajectories
            /mydata/miniconda3/envs/casual_sim_abr/bin/python inference/downloadtime_subset_cfs.py --dir "$root_dir" --left_out_policy "$target" --C "$c"  # (pretty time-consuming)
            # calculate the average stall ratio
            /mydata/miniconda3/envs/casual_sim_abr/bin/python analysis/original_subset_stall.py --dir "$root_dir" --left_out_policy "$target" # ExpertSim 
            /mydata/miniconda3/envs/casual_sim_abr/bin/python analysis/subset_stall.py --dir "$root_dir" --left_out_policy "$target" --C "$c" # CausalSim 
            # Tune CausalSim's hyper-parameters for downloadtime prediction
            # /mydata/miniconda3/envs/casual_sim_abr/bin/python analysis/tune_downloadtime_hyperparameters.py --dir "$root_dir"  
        done
    done
done