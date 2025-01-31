import matplotlib.pyplot as plt
import numpy as np
# Data for plotting
# total_cost_al_tca = [140.4282448, 227.0846794, 313.785018, 400.6151555, 487.4763279, 630.9153907, 718.0240021, 805.1097164]
# accuracy_al_tca = [0.8496, 0.8524, 0.8525, 0.8515, 0.8536, 0.8553, 0.8528, 0.8569]

accuracy_al_tca_v2 = [
    0.1259201476,
    0.1249090689,
    0.114579969,
    0.1102304882,
    0.1081110797,
    0.1069054599,
    0.1064983145,
    0.1060266962,
    0.1058850513
]

total_cost_al_tca_v2 = [
    0,
    229.2879974,
    458.5759947,
    1400.229418,
    2377.55472,
    3423.857043,
    4512.847677,
    5338.335593,
    5366.958523
]



total_cost_direct = [
    0,
    36,
    2181,
    3600,
    4968,
    6255.204618,
    # 6389.882248,
    # 6546.333328,
    # 6716.428758
]
accuracy_direct = [
    0.1766499059,
    0.1347010113,
    0.1303362097,
    0.1292295164,
    0.1288875272,
    0.1281103446,
    # 0.1205089608,
    # 0.112984949,
    # 0.1037107742
]

total_cost_tca = [
    0,
    36,
    2181,
    3600,
    4968,
    6255.204618,
    # 6397.470228,
    # 6567.056988,
    # 6744.643748
]
accuracy_tca = [
    0.1715798347,
    0.1279365502,
    0.1275723947,
    0.1273629922,
    0.1271243684,
    0.1269767779,
    # 0.1197054676,
    # 0.1101934442,
    # 0.103272248
]


total_cost_EMA_NO_TCA = [
    0,
    208,
    455.1319947,
    942.0022655,
    1536.99016,
    2220.023582,
    2983.257066,
    4359.094742,
    5278.447833,
]
accuracy_EMA_NO_TCA = [
    0.1345979014,
    0.1303080815,
    0.1272212161,
    0.122506054,
    0.1148453414,
    0.1086585772,
    0.1069323082,
    0.1063228143,
    0.1059012028    
]


total_cost_al_tca_v2 = np.array(total_cost_al_tca_v2)
total_cost_direct = np.array(total_cost_direct)
total_cost_tca = np.array(total_cost_tca)
# total_cost_scratch = np.array(total_cost_scratch)
total_cost_EMA_NO_TCA = np.array(total_cost_EMA_NO_TCA)

total_cost_al_tca_v2 = total_cost_al_tca_v2 / 1000
total_cost_direct = total_cost_direct / 1000
total_cost_tca = total_cost_tca / 1000
# total_cost_scratch = total_cost_scratch / 1000
total_cost_EMA_NO_TCA = total_cost_EMA_NO_TCA / 1000


def plot_accuracy_vs_cost_contrast():
    plt.figure(figsize=(10, 6))  # Adjusted size for consistency with the bar chart
    ax = plt.gca()  # Get current axes

    font_size = 8
    line_width = 2
    marker_size = 5
    base_color = '#FFA500'  # Orange color in HEX

    # Define different line styles and markers for differentiation
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'd']

    # Plot each method with customized styles
    # plt.plot(total_cost_al_tca, accuracy_al_tca, label="AL_tca", 
    #          marker=markers[0], linestyle=line_styles[0], color="Blue", linewidth=line_width, markersize=marker_size)

    
    plt.plot(total_cost_al_tca_v2, accuracy_al_tca_v2, label="AL_tca_v2", 
             marker=markers[0], linestyle=line_styles[0], color=base_color, linewidth=line_width, markersize=marker_size)

    plt.plot(total_cost_direct, accuracy_direct, label="Direct", 
             marker=markers[1], linestyle=line_styles[1], color=base_color, linewidth=line_width, markersize=marker_size)

    plt.plot(total_cost_tca, accuracy_tca, label="TCA", 
             marker=markers[2], linestyle=line_styles[2], color="black", linewidth=line_width, markersize=marker_size)

    # plt.plot(total_cost_scratch, accuracy_scratch, label="Scratch", 
    #          marker=markers[3], linestyle=line_styles[3], color="black", linewidth=line_width, markersize=marker_size)

    # Label axes
    plt.xlabel("Total Cost", fontsize=font_size)
    plt.ylabel("Loss", fontsize=font_size)

    # Customize ticks
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # Customize ticks and spines
    ax.tick_params(axis='both', which='major', labelsize=font_size, direction='in', length=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add gridlines
    plt.grid(axis='both', linestyle='--', alpha=0.6, linewidth=0.5)

    # Add legend
    plt.legend(fontsize=font_size-2)

    plt.tight_layout()
    # Save the plot in both PDF and PNG formats
    plt.savefig('accuracy_vs_total_cost_contrast.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.savefig('accuracy_vs_total_cost_contrast.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.show()


def plot_accuracy_vs_cost():
    plt.figure(figsize=(2, 1.6))  # Adjusted size for consistency with the bar chart
    ax = plt.gca()  # Get current axes

    font_size = 8
    line_width = 0.7
    marker_size = 1.4
    base_color = '#FFA500'  # Orange color in HEX

    # Define different line styles and markers for differentiation
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'd']



    plt.plot(total_cost_direct , accuracy_direct, label="CL", 
             marker='s', linestyle='--', color='#1f77b4', linewidth=line_width, markersize=marker_size)

    plt.plot(total_cost_EMA_NO_TCA , accuracy_EMA_NO_TCA, label="EMA w/o PTA", 
             marker='o', linestyle='-.', color="black", linewidth=line_width, markersize=marker_size)

    plt.plot(total_cost_tca , accuracy_tca, label="EMA w/o ITA", 
             marker='^', linestyle='-', color="black", linewidth=line_width, markersize=marker_size)

    # Plot each method with customized styles
    plt.plot(total_cost_al_tca_v2 , accuracy_al_tca_v2, label="EMA", 
             marker='o', linestyle='-', color='#d62728', linewidth=line_width, markersize=marker_size)

    # Label axes
    plt.xlabel("Norm. Cost", fontsize=font_size)
    plt.ylabel("DropLatency Loss", fontsize=font_size)

    # Customize ticks
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # Customize ticks and spines
    ax.tick_params(axis='both', which='major', labelsize=font_size, direction='in', length=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add gridlines
    plt.grid(axis='both', linestyle='--', alpha=0.6, linewidth=0.5)

    plt.ylim(bottom=0.0)
    plt.xticks([0, 2, 4, 6])
    # Add legend
    plt.legend(fontsize=font_size-3)

    plt.tight_layout()
    # Save the plot in both PDF and PNG formats
    plt.savefig('BREAK_DOWN.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.savefig('BREAK_DOWN.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.show()

# Call the function to plot
plot_accuracy_vs_cost()