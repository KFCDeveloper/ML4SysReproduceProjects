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

accuracy_al_tca = [
    0.5762,
    0.5781,
    0.5787,
    0.5795,
    0.8495,
    0.8527,
    0.8510,
    0.8519,
    0.8537,
    0.8529
]

accuracy_al_tca = [ 100 * x for x in accuracy_al_tca ]

# Cost values
total_cost_al_tca = [
    8,
    48,
    85.54101729,
    158.8105426,
    257.1222875,
    333.1289575,
    410.5512187,
    493.3946452,
    592.2345114,
    673.886554
]

# total_cost_direct = [327.096281, 514.192562, 701.288843, 888.385124, 1075.481405]
# accuracy_direct = [0.7843, 0.7846, 0.7868, 0.7878, 0.7879]

total_cost_direct = [0,
109,
218,327.096281, 514.192562, 701.288843, 888.385124
# , 1075.481405
]
accuracy_direct = [0.6463,
0.6493,
0.6509,0.7843, 0.7846, 0.7868, 0.7878
# , 0.7879
]

accuracy_direct = [ 100 * x for x in accuracy_direct]

# total_cost_tca = [292.096281, 500.192562, 687.288843, 902.385124, 1040.481405]
# accuracy_tca = [0.8496, 0.8513, 0.8541, 0.8544, 0.856]
# accuracy_tca = [ 100 * x for x in accuracy_tca]

total_cost_tca = [0,
73,
146,
219,292.096281, 500.192562, 687.288843, 902.385124, 1040.481405]
accuracy_tca = [0.5882,0.5937,0.6073,0.6303,0.8496, 0.8513, 0.8541, 0.8544, 0.856]


# total_cost_scratch = [327.096281, 514.192562, 701.288843, 972.385124, 1075.481405]
# accuracy_scratch = [0.7859, 0.7863, 0.7883, 0.7885, 0.7891]

total_cost_scratch = [0, 
    103.88,
    207.76,
    311.64,
    415.52,
    519.4,
    623.28,
    727.16,
    831.04,
    935.481405,
    # 938.481405,
        # 941.481405,
        # 944.481405,
        # 947.481405,
        # 950.481405
    ]
accuracy_scratch = [0,
    0.252,
    0.3689,
    0.3859,
    0.4048,
    0.4262,
    0.5012,
    0.5554,
    0.5738,
    0.5756,
    # 0.6139,
    # 0.7062,
    # 0.7624,
    # 0.7809,
    # 0.7854
]

accuracy_scratch = [ 100 * x for x in accuracy_scratch]

total_cost_al_tca_v2 = np.array(total_cost_al_tca_v2)
total_cost_direct = np.array(total_cost_direct)
total_cost_tca = np.array(total_cost_tca)
total_cost_scratch = np.array(total_cost_scratch)

total_cost_al_tca_v2 = total_cost_al_tca_v2 * 2 / 360
total_cost_direct = total_cost_direct * 2 / 360
total_cost_tca = total_cost_tca * 2 / 360
total_cost_scratch = total_cost_scratch * 2 / 360


def plot_accuracy_vs_cost():
    plt.figure(figsize=(2.1, 1.75), dpi=300)  # Adjusted size for consistency with the bar chart
    ax = plt.gca()  # Get current axes

    font_size = 8
    line_width = 1.1
    marker_size = 1.9

    # Define different line styles and markers for differentiation
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', '^']

    plt.plot(total_cost_scratch, accuracy_scratch, label="DOTE", 
             marker=markers[3], linestyle="-", color="black", linewidth=line_width, markersize=marker_size)
    
    plt.plot(total_cost_direct, accuracy_direct, label="DOTE + CL", 
             marker=markers[1], linestyle=line_styles[1], color="#1f77b4", linewidth=line_width, markersize=marker_size)
    
    # Plot each method with customized styles
    plt.plot(total_cost_al_tca_v2, accuracy_al_tca_v2, label="DOTE + EMA", 
             marker=markers[0], linestyle=line_styles[0], color="#d62728", linewidth=line_width, markersize=marker_size)



    # Label axes
    plt.xlabel("Norm. Cost", fontsize=font_size)
    plt.ylabel("Model Accuracy(%)", fontsize=font_size)

    # Customize ticks
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # Customize ticks and spines
    ax.tick_params(axis='both', which='major', labelsize=font_size, direction='in', length=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim(left=-0.2)
    ax.set_ylim(bottom=-1)

    # Add gridlines
    plt.grid(axis='both', linestyle='--', alpha=0.6, linewidth=0.5)

    # Add legend
    plt.legend(fontsize=font_size-2)

    plt.tight_layout()
    # Save the plot in both PDF and PNG formats
    plt.savefig('DOTE_accuracy_vs_total_cost.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.savefig('DOTE_accuracy_vs_total_cost.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.show()

# Call the function to plot
plot_accuracy_vs_cost()