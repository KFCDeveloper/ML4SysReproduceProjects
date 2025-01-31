import matplotlib.pyplot as plt
import numpy as np


accuracy_al_tca_v2 = [
    0.6238,
    0.6276,
    0.6291,
    0.6521,
    0.8502,
    0.852,
    0.8521,
    0.8507,
    0.8532,
    0.8539,
    0.8558,
    0.8545
]

accuracy_al_tca_v2 = [ 100 * x for x in accuracy_al_tca_v2]

total_cost_al_tca_v2 = [
    8.75,
    35,
    70,
    86.1236012,
    187.1023571,
    267.7066016,
    354.4122307,
    431.9982316,
    538.8052559,
    614.5866632,
    770.4546711,
    883.4369628
]


# total_cost_direct = [327.096281, 514.192562, 701.288843, 888.385124, 1075.481405]
# accuracy_direct = [0.7843, 0.7846, 0.7868, 0.7878, 0.7879]

total_cost_direct = [0,
109,
218,327.096281, 514.192562, 701.288843, 888.385124, 1075.481405]
accuracy_direct = [0.6463,
0.6493,
0.6509,0.7843, 0.7846, 0.7868, 0.7878, 0.7879]

accuracy_direct = [ 100 * x for x in accuracy_direct]

# total_cost_tca = [292.096281, 500.192562, 687.288843, 902.385124, 1040.481405]
# accuracy_tca = [0.8496, 0.8513, 0.8541, 0.8544, 0.856]

total_cost_tca = [0,
73,
146,
219,292.096281, 500.192562, 687.288843, 902.385124, 1040.481405]
accuracy_tca = [0.5882,0.5937,0.6073,0.6303,0.8496, 0.8513, 0.8541, 0.8544, 0.856]

accuracy_tca = [ 100 * x for x in accuracy_tca]

total_cost_no_tca = [6, 45.5, 84.13523364, 179.8088751, 258.4652276, 334.4727852, 421.7870586, 520.4665997, 621.9079363, 758.5669961]
accuracy_no_tca = [0.6434, 0.6449, 0.6455, 0.7823, 0.7828, 0.7835, 0.784, 0.7874, 0.7884, 0.7864]

accuracy_no_tca = [ 100 * x for x in accuracy_no_tca]



total_cost_al_tca_v2 = np.array(total_cost_al_tca_v2)
total_cost_direct = np.array(total_cost_direct)
total_cost_tca = np.array(total_cost_tca)
# total_cost_scratch = np.array(total_cost_scratch)
total_cost_no_tca = np.array(total_cost_no_tca)

total_cost_al_tca_v2 = total_cost_al_tca_v2 / 100
total_cost_direct = total_cost_direct / 100
total_cost_tca = total_cost_tca / 100
# total_cost_scratch = total_cost_scratch / 100
total_cost_no_tca = total_cost_no_tca / 100



def plot_accuracy_vs_cost():
    plt.figure(figsize=(2.1, 1.75), dpi=300)

    font_size = 8
    line_width = 1.1
    marker_size = 1.9

    plt.plot(
        total_cost_direct, accuracy_direct,
        label="CL", color='#1f77b4', linestyle='-', marker='s',
        linewidth=line_width, markersize=marker_size
    )
    plt.plot(
        total_cost_no_tca, accuracy_no_tca,
        label="EMA w/o PTA", color="black", linestyle='-.', marker='o',
        linewidth=line_width, markersize=marker_size
    )
    plt.plot(
        total_cost_tca, accuracy_tca,
        label="EMA w/o ITA", color='black', linestyle='-.', marker='^',
        linewidth=line_width, markersize=marker_size
    )

    plt.plot(
        total_cost_al_tca_v2, accuracy_al_tca_v2,
        label="EMA", color='#d62728', linestyle='-', marker='o',
        linewidth=line_width, markersize=marker_size
    )

    plt.xlabel("Norm. Cost", fontsize=font_size)
    plt.ylabel("Model Accuracy (%)", fontsize=font_size)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', direction='in', length=4)

    plt.grid(axis='both', linestyle='-.', alpha=0.6, linewidth=0.5)

    plt.legend(fontsize=font_size - 3)

    plt.xlim(right=6.2)
    #xtick is 2 and 4, without decimal
    plt.xticks([0, 2, 4, 6])
    plt.tight_layout()

    plt.savefig('DOTE_breakdown.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.savefig('DOTE_breakdown.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)

    plt.show()

plot_accuracy_vs_cost()