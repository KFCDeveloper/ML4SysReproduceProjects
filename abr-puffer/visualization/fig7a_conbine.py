import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="root directory")
args = parser.parse_args()

policy_names = ['bola_basic_v2', 'bola_basic_v1', 'puffer_ttp_cl', 'puffer_ttp_20190202', 'linear_bba']
buffer_based_names = ['linear_bba']

# with open(f'{args.dir}tuned_hyperparams/buffer.pkl', 'rb') as f:
#     f = pickle.load(f)
#     bf_C = {policy: f[policy][1] for policy in buffer_based_names}
bf_C = {'linear_bba':'0.05'}
plt.figure(figsize=(5.25, 4.25))
label_dic={'CAUSALSIM_DIR/':10,'CAUSALSIM_DIR-20-9-27/':8,'CAUSALSIM_DIR-20-11-27/':6,'CAUSALSIM_DIR-21-1-27/':4,'CAUSALSIM_DIR-21-3-27/':2}
for root_dir in ['CAUSALSIM_DIR/','CAUSALSIM_DIR-20-9-27/','CAUSALSIM_DIR-20-11-27/','CAUSALSIM_DIR-21-1-27/','CAUSALSIM_DIR-21-3-27/']:
    sim_EMDs = []
    for left_out_policy in buffer_based_names:
        with open(f'{root_dir}subset_EMDs/{left_out_policy}/sim_buff_{bf_C[left_out_policy]}.pkl', 'rb') as f:
            sim_dict = pickle.load(f)
    
    sim_EMDs.extend([sim_dict[source][left_out_policy] for source in policy_names if source != left_out_policy])
    # 相当于，对于 target 是 linear_bba 来说，从 那应用了不同 policy 计算过来的 simulation，然后和ground truth去比EMD
    
    sim_EMDs = np.sort(sim_EMDs)
    # plt.plot(expert_EMDs, [i/12*100 for i in range(1, 13)], label='ExpertSim')
    # plt.plot(sim_EMDs, [i/12*100 for i in range(1, 13)], label='CausalSim')
    plt.plot(sim_EMDs, [i/4*100 for i in range(1, 5)], label=str(label_dic[root_dir])+' months')
plt.legend()
plt.ylabel('CDF %')
plt.xlabel('EMD')

fig_path = f'{args.dir}plots'
os.makedirs(fig_path, exist_ok=True)
# plt.savefig(f'{fig_path}/fig7a.pdf', format='pdf')
plt.savefig(f'{fig_path}/fig7a-conbine.png')