import matplotlib.pyplot as plt
import numpy as np

# Data for Task 3
accuracy_direct = [
    17.61471783, 15.75334057, 15.20389395, 16.72, 21.215, 25.021, 25.962, 26.274, 26.891
]
total_cost_direct = [
    7392.415273, 14784.83055, 22177.24582, 22293.54582, 23949.71582, 25613.85582, 27277.25582, 28941.09582, 30438.57582
]

accuracy_ema = [
    32.387, 37.937, 41.136, 44.011, 35.034, 50.068, 47.56
]
total_cost_ema = [
    6050.430427, 8075.760241, 10178.23106, 13531.1743, 18414.43667, 24338.29947, 29041.54773
]

accuracy_tca = [
    19.58127402, 22.66486717, 26.2887236, 28.249, 28.334, 31.678, 32.189, 32.751, 30.84
]
total_cost_tca = [
    7392.415273, 14784.83055, 22177.24582, 22293.54582, 23949.71582, 25613.85582, 27277.25582, 28941.09582, 30438.57582
]

accuracy_no_tca = [
    # 10.969, 24.38, 23.307, 
    22.971, 22.02, 22.312, 22.575, 23.337, 23.985
]
total_cost_no_tca = [
    # 998.9509537, 1845.805092, 5469.553954, 
    7531.809821, 10513.46893, 13019.95624, 16397.9867, 20803.8504, 24252.15009
]

# Normalize costs
total_cost_direct = np.array(total_cost_direct) / 100
total_cost_ema = np.array(total_cost_ema) / 100
total_cost_tca = np.array(total_cost_tca) / 100
total_cost_no_tca = np.array(total_cost_no_tca) / 100

total_cost_direct = total_cost_direct / 50
total_cost_ema = total_cost_ema / 50
total_cost_tca = total_cost_tca / 50
total_cost_no_tca = total_cost_no_tca / 50

scale = 1.45
total_cost_direct = total_cost_direct - scale
total_cost_ema = total_cost_ema - scale + 0.25
total_cost_tca = total_cost_tca - scale
total_cost_no_tca = total_cost_no_tca - scale

# Plot
def plot_accuracy_vs_cost():
    plt.figure(figsize=(2.1, 1.75), dpi=300)

    font_size = 8
    line_width = 1.1
    marker_size = 1.9

    plt.plot(total_cost_direct, accuracy_direct, label="CL", color='black', linestyle='-', marker='s',
             linewidth=line_width, markersize=marker_size)
    plt.plot(total_cost_tca, accuracy_tca, label="EMA w/o ITA", color='black', linestyle='-.', marker='^',
             linewidth=line_width, markersize=marker_size)
    plt.plot(total_cost_no_tca, accuracy_no_tca, label="EMA w/o PTA", color="#1f77b4", linestyle='-.', marker='o',
             linewidth=line_width, markersize=marker_size)
    plt.plot(total_cost_ema, accuracy_ema, label="EMA", color='#d62728', linestyle='-', marker='o',
             linewidth=line_width, markersize=marker_size)

    plt.xlabel("Norm. Cost", fontsize=font_size)
    plt.ylabel("Reward", fontsize=font_size)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', direction='in', length=4)

    plt.grid(axis='both', linestyle='-.', alpha=0.6, linewidth=0.5)

    plt.legend(fontsize=font_size - 3)

    plt.xlim(right=5.0)
    plt.xlim(left=-0.2)
    plt.xticks([0, 2, 4])
    plt.tight_layout()

    plt.savefig('FIRM_breakdown.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.savefig('FIRM_breakdown.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)

    plt.show()

plot_accuracy_vs_cost()