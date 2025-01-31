
import json
import matplotlib.pyplot as plt

# 从文件中读取数据
with open('cost_data.json', 'r') as f:
    data = json.load(f)

# 只使用前 6 个数据点
qoe = data["qoe"][:6]
mean_direct_cost = data["mean_direct_cost"][:6]
mean_tca_cost = data["mean_tca_cost"][:6]
mean_tca_al_cost = data["mean_tca_al_cost"][:6]
mean_new_ema_cost = data["mean_new_ema_cost"][:6]

scaled_data_cost = 0.733

scratch_cost = [0, 2.798, 3.881, None, None, None, None, None]
scaled_scratch_cost = [c + scaled_data_cost if c is not None else None for c in scratch_cost][:6]
scaled_scratch_cost[0] = 0

# 绘制图表
# 绘制图表
plt.figure(figsize=(2.1, 1.75), dpi=300)  # Adjusted size for consistency with the bar chart

# Scratch
plt.plot([c for c in scaled_scratch_cost if c is not None], [qoe[i] for i in range(len(qoe)) if scaled_scratch_cost[i] is not None], 
         color='black', linestyle='--', marker='^', label='NetLLM', linewidth=1.1, markersize=1.9)

# Mean Direct
plt.plot([c for c in mean_direct_cost if c is not None], [qoe[i] for i in range(len(qoe)) if mean_direct_cost[i] is not None], 
         color='#1f77b4', linestyle='-', marker='s', label='NetLLM + CL', linewidth=1.1, markersize=1.9)

# Mean TCA + AL
plt.plot([c for c in mean_new_ema_cost if c is not None], [qoe[i] for i in range(len(qoe)) if mean_new_ema_cost[i] is not None], 
         color='#d62728', linestyle='-', marker='o', label='NetLLM + EMA', linewidth=1.1, markersize=1.9)

# # 设置 x 轴限制
# plt.xlim(-0.2, 5.0)
# plt.ylim(0.6199, 0.79)

# 设置图例
plt.legend(prop={'size': 6})

# 设置轴标签
plt.xlabel('Norm. Cost', fontsize=8)
plt.ylabel('QoE', fontsize=8)

# 设置坐标轴刻度和标签字体大小
# x轴只要0, 2, 4
plt.xticks(fontsize=8, ticks=[0, 2, 4])
plt.yticks(fontsize=8, ticks=[0.6, 0.7, 0.8])

# 稀疏纵坐标和横坐标
# plt.gca().yaxis.set_major_locator(plt.MaxNLocator(4))
# plt.gca().xaxis.set_major_locator(plt.MaxNLocator(4))

# 只保留左边和下边的坐标轴线
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)

# tight_layout
plt.tight_layout()
# 调整图表边距以确保标签完全显示
# plt.subplots_adjust(bottom=0.23)
# plt.subplots_adjust(left=0.255)
# plt.subplots_adjust(top=1.0)
# plt.subplots_adjust(right=1.0)

# 显示网格

plt.grid(axis='both', linestyle='--', alpha=0.6, linewidth=0.5)

#调整最后图片大小resize


# 保存图表
plt.savefig(f'NetLLM_qoe_vs_cost_chunk.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
plt.savefig(f'NetLLM_qoe_vs_cost_chunk.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)

# 显示图表
plt.show()