import matplotlib.pyplot as plt
import numpy as np
# Data for plotting
# total_cost_al_tca = [140.4282448, 227.0846794, 313.785018, 400.6151555, 487.4763279, 630.9153907, 718.0240021, 805.1097164]
# accuracy_al_tca = [0.8496, 0.8524, 0.8525, 0.8515, 0.8536, 0.8553, 0.8528, 0.8569]

accuracy_al_tca_v2 = [
    # 0.675351033,
    0.746040372,
    0.764645958,
    0.776946252,
    0.78571453,
    0.789757003,
    0.79263705,
    0.795022208,
    0.798072406,
    0.799986021
]

total_cost_al_tca_v2 = [
    # 481.7843122,
    963.7143715,
    1446.608732,
    1930.066849,
    2414.343889,
    2898.664743,
    3383.050119,
    3868.128111,
    4353.853857,
    4420.998271
]



total_cost_direct = [
    981.4403204,
    1962.880641,
    2944.320961,
    3925.761282,
    4907.201602
]
accuracy_direct = [
    0.087662562,
    0.12612238,
    0.141238439,
    0.14588919,
    0.151914152
]

total_cost_tca = [
    981.4403204,
    1962.880641,
    2944.320961,
    3925.761282,
    4907.201602
]
accuracy_tca = [
    0.665054536,
    0.748795908,
    0.763162748,
    0.771427844,
    0.776907449
]

total_cost_scratch = [
    981.4403204,
    1962.880641,
    2944.320961,
    3925.761282,
    4907.201602
]
accuracy_scratch = [
    0.21594619,
    0.27098489,
    0.28096608,
    0.28611655,
    0.2901934
]

total_cost_EMA_NO_TCA = [
    481.7843122,
    963.7143715,
    1446.608732,
    1930.066849,
    2414.343889,
    2898.664743,
    3383.050119,
    3868.128111,
    4353.853857,
    4420.998271
]
accuracy_EMA_NO_TCA = [
    0.078402585,
    0.097904139,
    0.108240189,
    0.122150517,
    0.128419901,
    0.12546019,
    0.128787761,
    0.134648108,
    0.131383221,
    0.135458767
]


total_cost_al_tca_v2 = np.array(total_cost_al_tca_v2)
total_cost_direct = np.array(total_cost_direct)
total_cost_tca = np.array(total_cost_tca)
total_cost_scratch = np.array(total_cost_scratch)
total_cost_EMA_NO_TCA = np.array(total_cost_EMA_NO_TCA)

total_cost_al_tca_v2 = total_cost_al_tca_v2 / 1000
total_cost_direct = total_cost_direct / 1000
total_cost_tca = total_cost_tca / 1000
total_cost_scratch = total_cost_scratch / 1000
total_cost_EMA_NO_TCA = total_cost_EMA_NO_TCA / 1000

bias = 0.95
total_cost_al_tca_v2 = total_cost_al_tca_v2 - bias
total_cost_direct = total_cost_direct - bias
total_cost_tca = total_cost_tca - bias
total_cost_scratch = total_cost_scratch - bias
total_cost_EMA_NO_TCA = total_cost_EMA_NO_TCA - bias


# total_cost_al_tca_v2 = total_cost_al_tca_v2 / 10
# total_cost_direct = total_cost_direct / 10
# total_cost_tca = total_cost_tca / 10
# total_cost_scratch = total_cost_scratch / 10
# total_cost_EMA_NO_TCA = total_cost_EMA_NO_TCA / 10


# def plot_accuracy_vs_cost_old():
#     plt.figure(figsize=(2.1, 1.75))  # Adjusted size for consistency with the bar chart
#     ax = plt.gca()  # Get current axes

#     font_size = 8
#     line_width = 1.1
#     marker_size = 1.9
#     base_color = '#FFA500'  # Orange color in HEX

#     # Define different line styles and markers for differentiation
#     line_styles = ['-', '--', '-.', ':']
#     markers = ['o', 's', '^', 'd']


    
#     # Plot each method with customized styles
#     plt.plot(total_cost_al_tca_v2 , accuracy_al_tca_v2, label="Flux + EMA", 
#              marker=markers[0], linestyle=line_styles[0], color=base_color, linewidth=line_width, markersize=marker_size)

#     plt.plot(total_cost_direct , accuracy_direct, label="Flux + CL", 
#              marker=markers[1], linestyle=line_styles[1], color=base_color, linewidth=line_width, markersize=marker_size)
#     plt.plot(total_cost_scratch , accuracy_scratch, label="TCA", marker=markers[2], linestyle=line_styles[2], color="black", linewidth=line_width, markersize=marker_size)

#     plt.plot(total_cost_tca , accuracy_tca, label="Flux + PTA", 
#              marker=markers[2], linestyle=line_styles[2], color="black", linewidth=line_width, markersize=marker_size)

#     plt.plot(total_cost_EMA_NO_TCA , accuracy_EMA_NO_TCA, label="Flux + ITA", 
#              marker=markers[3], linestyle=line_styles[3], color="black", linewidth=line_width, markersize=marker_size)

#     # Label axes
#     plt.xlabel("Total Cost(s)", fontsize=font_size)
#     plt.ylabel("Model Accuracy (%)", fontsize=font_size)

#     # Customize ticks
#     plt.xticks(fontsize=font_size)
#     plt.yticks(fontsize=font_size)

#     # Customize ticks and spines
#     ax.tick_params(axis='both', which='major', labelsize=font_size, direction='in', length=4)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

#     # Add gridlines
#     plt.grid(axis='both', linestyle='--', alpha=0.6, linewidth=0.5)

#     # Add legend
#     plt.legend(fontsize=font_size-3)

#     plt.tight_layout()
#     # Save the plot in both PDF and PNG formats
#     plt.savefig('BREAK_DOWN.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
#     plt.savefig('BREAK_DOWN.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
#     plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # Data
# accuracy_al_tca_v2 = [
#     0.675351033, 0.746040372, 0.764645958, 0.776946252, 0.78571453,
#     0.789757003, 0.79263705, 0.795022208, 0.798072406, 0.799986021
# ]
# total_cost_al_tca_v2 = [
#     481.7843122, 963.7143715, 1446.608732, 1930.066849, 2414.343889,
#     2898.664743, 3383.050119, 3868.128111, 4353.853857, 4420.998271
# ]

# accuracy_direct = [
#     0.087662562, 0.12612238, 0.141238439, 0.14588919, 0.151914152
# ]
# total_cost_direct = [
#     981.4403204, 1962.880641, 2944.320961, 3925.761282, 4907.201602
# ]

# accuracy_scratch = [
#     0.21594619, 0.27098489, 0.28096608, 0.28611655, 0.2901934
# ]
# total_cost_scratch = [
#     981.4403204, 1962.880641, 2944.320961, 3925.761282, 4907.201602
# ]

import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_vs_cost_old():
    plt.figure(figsize=(2.1, 1.75), dpi=300)  # 调整图表大小
    ax = plt.gca()  # 获取当前坐标轴

    font_size = 8
    line_width = 1.1
    marker_size = 1.9

    # 定义不同的线条样式和标记
    base_color = '#FFA500'  # 橙色
    # 保持统一的颜色和样式设置
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'd']

    # 绘制每种方法，并根据模板设置颜色、线条和标记样式


    plt.plot(total_cost_direct, accuracy_direct, label="Flux + CL", 
             marker=markers[1], linestyle=line_styles[1], color='#1f77b4', linewidth=line_width, markersize=marker_size)

    # plt.plot(total_cost_scratch, accuracy_scratch, label="TCA", 
    #          marker=markers[2], linestyle=line_styles[2], color="black", linewidth=line_width, markersize=marker_size)

    plt.plot(total_cost_EMA_NO_TCA, accuracy_EMA_NO_TCA, label="Flux + ITA", 
             marker=markers[3], linestyle=line_styles[3], color="black", linewidth=line_width, markersize=marker_size)

    plt.plot(total_cost_tca, accuracy_tca, label="Flux + PTA", 
             marker=markers[2], linestyle=line_styles[2], color="black", linewidth=line_width, markersize=marker_size)

    plt.plot(total_cost_al_tca_v2, accuracy_al_tca_v2, label="Flux + EMA", 
             marker=markers[0], linestyle=line_styles[0], color='#d62728', linewidth=line_width, markersize=marker_size)

    # 设置轴标签
    plt.xlabel("Norm. Cost", fontsize=font_size)
    plt.ylabel(f'$\\mathrm{{R}}^2$ Score', fontsize=font_size)

    # 设置刻度字体大小
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # 自定义坐标轴
    ax.tick_params(axis='both', which='major', labelsize=font_size, direction='in', length=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 添加网格线
    plt.grid(axis='both', linestyle='--', alpha=0.6, linewidth=0.5)

    plt.xlim(-0.2, 5)

    # 添加图例
    plt.legend(fontsize=font_size - 2)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig('FLUX_BREAK_DOWN.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.savefig('FLUX_BREAK_DOWN.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)

    # 显示图表
    plt.show()

# 调用绘图函数
plot_accuracy_vs_cost_old()

# Plotting
def plot_accuracy_vs_cost():
    plt.figure(figsize=(2.1, 1.75), dpi=300)
    font_size = 8
    line_width = 1.1
    marker_size = 1.9

    # Plot Scratch (black dashed line)
    plt.plot(
        total_cost_scratch, accuracy_scratch,
        label="Flux", color='black', linestyle='--', marker='^',
        linewidth=line_width, markersize=marker_size
    )

    # Plot Direct (blue solid line)
    plt.plot(
        total_cost_direct, accuracy_direct,
        label="Flux + CL", color='#1f77b4', linestyle='-', marker='s',
        linewidth=line_width, markersize=marker_size
    )
    
    # Plot EMA (red solid line)
    plt.plot(
        total_cost_al_tca_v2, accuracy_al_tca_v2,
        label="Flux + EMA", color='#d62728', linestyle='-', marker='o',
        linewidth=line_width, markersize=marker_size
    )



    # Labels
    plt.xlabel("Norm. Cost", fontsize=font_size)
    plt.ylabel('R-Squared Score', fontsize=font_size, labelpad=0)

    # Ticks
    plt.xticks(fontsize=font_size)
    # plt.yticks(fontsize=font_size, [0, 0.2, 0.4, 0.6, 0.8])
    # y轴上刻度是0 0.2 0.4 。。。
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8], fontsize=font_size)




    # Customize axes
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', direction='in', length=4)

    # Grid and legend
    plt.grid(axis='both', linestyle='--', alpha=0.6, linewidth=0.5)
    plt.legend(fontsize=font_size-2, loc='center right')

    # legend move up a little
    plt.legend(fontsize=font_size-2, loc='center right', bbox_to_anchor=(1, 0.6))

    plt.xlim(-0.2, 4.1)

    # plt.subplots_adjust(left=0.2)  # 试着把 0.1 改成 0.2 或更大
    plt.tight_layout()

    # Save the plot
    plt.savefig('FLUX_accuracy_vs_cost.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.savefig('FLUX_accuracy_vs_cost.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)

    # Show the plot
    plt.show()

# Call the function
# plot_accuracy_vs_cost()
# # Call the function to plot
plot_accuracy_vs_cost()
# plot_accuracy_vs_cost_old()
