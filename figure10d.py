import matplotlib.pyplot as plt
import numpy as np

# Data for Task 3
accuracy_scratch = [
    -18.06106872, -14.91856583, -10.87272806, -11.92, -12.078, -9.667, 7.071, 24.831, 30.651
]
total_cost_scratch = [
    7392.415273, 14784.83055, 22177.24582, 22293.54582, 23949.71582, 25613.85582, 27277.25582, 28941.09582, 30438.57582
]

accuracy_direct = [
    17.61471783, 15.75334057, 15.20389395, 16.72, 21.215, 25.021, 25.962, 26.274, 26.891
]
total_cost_direct = [
    7392.415273, 14784.83055, 22177.24582, 22293.54582, 23949.71582, 25613.85582, 27277.25582, 28941.09582, 30438.57582
]

accuracy_ema = [
    # 10.893, 10.952, 12.527, 16.368, 20.205,
    32.387, 37.937, 41.136, 44.011, 35.034, 50.068, 47.56
]
total_cost_ema = [
    # 808.501932, 1377.832639, 2074.708132, 3038.05547, 4323.797036, 
    6050.430427, 8075.760241, 10178.23106, 13531.1743, 18414.43667, 24338.29947, 29041.54773
]

# Normalize costs
total_cost_scratch = np.array(total_cost_scratch) * 2 / 360
total_cost_direct = np.array(total_cost_direct) * 2 / 360
total_cost_ema = np.array(total_cost_ema) * 2 / 360


total_cost_scratch = total_cost_scratch - 40
total_cost_direct = total_cost_direct - 40
total_cost_ema = total_cost_ema - 34

total_cost_scratch = total_cost_scratch / 25
total_cost_direct = total_cost_direct / 25
total_cost_ema = total_cost_ema / 25

# Plot
def plot_accuracy_vs_cost():
    plt.figure(figsize=(2.1, 1.75), dpi=300)
    ax = plt.gca()

    font_size = 8
    line_width = 1.1
    marker_size = 1.9

    plt.plot(total_cost_scratch, accuracy_scratch, label="FIRM", 
             marker='^', linestyle="--", color="black", linewidth=line_width, markersize=marker_size)
    
    plt.plot(total_cost_direct, accuracy_direct, label="FIRM + CL", 
             marker='s', linestyle="-", color="#1f77b4", linewidth=line_width, markersize=marker_size)
    
    plt.plot(total_cost_ema, accuracy_ema, label="FIRM + EMA", 
             marker='o', linestyle="-", color="#d62728", linewidth=line_width, markersize=marker_size)

    plt.xlabel("Norm. Cost", fontsize=font_size)
    plt.ylabel("Reward", fontsize=font_size)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    ax.tick_params(axis='both', which='major', labelsize=font_size, direction='in', length=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim(left=-0.2)
    ax.set_ylim(bottom=-20)

    plt.legend(fontsize=font_size-2)
    plt.tight_layout()
    plt.savefig('FIRM_accuracy_vs_total_cost.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.savefig('FIRM_accuracy_vs_total_cost.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.show()

plot_accuracy_vs_cost()