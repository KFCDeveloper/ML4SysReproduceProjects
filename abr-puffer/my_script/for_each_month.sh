#!/bin/bash
cd /mydata/Unbiased-Trace-Driven-Simulation/abr-puffer

# {"CAUSALSIM_DIR-dis-20-7-27/","CAUSALSIM_DIR-dis-20-8-27/","CAUSALSIM_DIR-dis-20-9-27/","CAUSALSIM_DIR-dis-20-10-27/","CAUSALSIM_DIR-dis-20-11-27/","CAUSALSIM_DIR-dis-20-12-27/","CAUSALSIM_DIR-dis-21-1-27/","CAUSALSIM_DIR-dis-21-2-27/"}
# root_dirs=("CAUSALSIM_DIR-dis-20-7-27/" "CAUSALSIM_DIR-dis-20-8-27/" "CAUSALSIM_DIR-dis-20-9-27/" "CAUSALSIM_DIR-dis-20-10-27/" "CAUSALSIM_DIR-dis-20-11-27/" "CAUSALSIM_DIR-dis-20-12-27/" "CAUSALSIM_DIR-dis-21-1-27/" "CAUSALSIM_DIR-dis-21-2-27/")   #  "CAUSALSIM_DIR-dis-20-7-27/"
root_dirs=("CAUSALSIM_DIR-21-3-27_applied/")
c_s=("0.1" "0.5" "1.0" "5.0" "10.0" "15.0" "20.0" "25.0" "30.0" "40.0")  # "0.05" "0.1" "0.5" "1.0" "5.0" "10.0" "15.0" "20.0" "25.0" "30.0" "40.0";  "0.1" "0.5" 
targets=("bola_basic_v1" "bola_basic_v2" "linear_bba")  # "bola_basic_v1" "bola_basic_v2" "linear_bba"


for root_dir in "${root_dirs[@]}"
do  
    # /mydata/miniconda3/envs/casual_sim_abr/bin/python data_preparation/generate_subset_data.py --dir "$root_dir"
    for target in "${targets[@]}"
    do
        for c in "${c_s[@]}"
        do
            # train CasualSim
            /mydata/miniconda3/envs/casual_sim_abr/bin/python training/train_subset.py --dir "$root_dir" --left_out_policy "$target" --C "$c" 
        done
    done
done