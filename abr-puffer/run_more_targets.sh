#!/bin/bash

cd /mydata/Unbiased-Trace-Driven-Simulation/abr-puffer
targets=("bola_basic_v1" "bola_basic_v2")  # "linear_bba"
# root_dirs=("CAUSALSIM_DIR-20-7-27/" "CAUSALSIM_DIR-20-9-27/" "CAUSALSIM_DIR-20-11-27/" "CAUSALSIM_DIR-21-1-27/" "CAUSALSIM_DIR-21-3-27/")
root_dir="CAUSALSIM_DIR-21-3-27/"

for target in "${targets[@]}"
do
    # train CasualSim
    python training/train_subset.py --dir "$root_dir" --left_out_policy "$target" --C 0.05 

    ## Training
    #  Using CausalSim to extract and save the latent factors ** need
    python inference/extract_subset_latents.py --dir "$root_dir" --left_out_policy "$target" --C 0.05 
    # Counterfactual Simulation
    python inference/expert_cfs.py --dir "$root_dir" # ExpertSim ** need  (pretty time-consuming)
    python inference/buffer_subset_cfs.py --dir "$root_dir" --left_out_policy "$target" --C 0.05 # CausalSim (pretty time-consuming)
    # Calculate the average SSIM using the ground-truth data
    python analysis/original_subset_ssim.py --dir "$root_dir" --left_out_policy "$target" # ExpertSim ** need
    python analysis/subset_ssim.py --dir "$root_dir" --left_out_policy "$target" --C 0.05 # CausalSim ** need
    # Calculate the simulated buffer distribution's Earth Mover Distance (EMD) using the fround-truth data
    python analysis/subset_EMD.py --dir CAUSALSIM_DIR-20-11-27/ --left_out_policy "$target" --C 0.05 # All{ExpertSim, SLSim, CausalSim} ** need
    # Tune CausalSim's hyper-parameters for buffer and SSIM prediction
    # python analysis/tune_buffer_hyperparams.py --dir "$root_dir"

    # CausalSim model to generate counterfactual downloadtime trajectories
    python inference/downloadtime_subset_cfs.py --dir "$root_dir" --left_out_policy "$target" --C 0.05  # (pretty time-consuming)
    # calculate the average stall ratio
    python analysis/original_subset_stall.py --dir "$root_dir" --left_out_policy "$target" # ExpertSim 
    python analysis/subset_stall.py --dir "$root_dir" --left_out_policy "$target" --C 0.05 # CausalSim 
    # Tune CausalSim's hyper-parameters for downloadtime prediction
    # python analysis/tune_downloadtime_hyperparameters.py --dir "$root_dir"  
done



