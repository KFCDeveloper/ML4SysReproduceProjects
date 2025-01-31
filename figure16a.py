
import matplotlib.pyplot as plt
import numpy as np

# 定义数据
bandwidths = ['0', '0.5', '1', '3']
x = np.arange(len(bandwidths))  # x 轴位置

# Accuracy 数据
avg_loss_dote = [0.8542, 0.8539, 0.8517, 0.8523]
avg_loss_dote = [x * 100 for x in avg_loss_dote]

# Speedup 数据
speedup_dote = [2, 2, 2, 2]

# 设置样式
font_size = 14
bar_width = 0.3
line_width = 2
marker_size = 2.5
bar_label_font_size = 11

# 颜色设置
color_acc = '#d62728'  # 红色 (Accuracy)
color_speedup = 'black'  # 蓝色 (Speedup)

# 创建图像
fig, ax1 = plt.subplots(figsize=(3.3, 2.4), dpi=300)

# 调整 x 轴，确保折线点正对柱状图中心
x_shifted = x  # 让折线和柱状图对齐

# 左 y 轴 (Accuracy) —— DOTE 柱状图
bars1 = ax1.bar(x_shifted - bar_width/2, avg_loss_dote, width=bar_width, color=color_acc, label='Accuracy')

# 右 y 轴 (Speedup) —— DOTE 柱状图
ax2 = ax1.twinx()
bars2 = ax2.bar(x_shifted + bar_width/2, speedup_dote, width=bar_width, color=color_speedup, label='Speedup')

# 设置 x 轴
ax1.set_xticks(x_shifted)
ax1.set_xticklabels(bandwidths, fontsize=font_size)
ax1.set_xlabel('Noise Scale', fontsize=font_size)

# 设置 y 轴标签
ax1.set_ylabel('Accuracy (%)', fontsize=font_size, color=color_acc)
ax2.set_ylabel('Speedup', fontsize=font_size, color=color_speedup)

# 调整 y 轴范围和刻度
ax1.set_ylim(0, 130)
ax1.set_yticks([0, 25, 50, 75, 100])
ax2.set_ylim(0, 3.9)

# 调整刻度样式
ax1.tick_params(axis='y', labelsize=font_size, colors=color_acc)
ax2.tick_params(axis='y', labelsize=font_size, colors=color_speedup)
ax1.tick_params(axis='x', which='major', labelsize=font_size, direction='in', length=4)

# 隐藏上、右边框
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 添加网格线
ax1.grid(axis='both', linestyle='--', alpha=0.6, linewidth=0.5)

# 分开 Legend，竖着叠放
ax1.legend(loc='upper left', fontsize=12, frameon=False, ncol=1)  # Accuracy 的 legend
ax2.legend(loc='upper left', fontsize=12, frameon=False, bbox_to_anchor=(0, 0.85), ncol=1)  # Speedup 的 legend

# 调整布局并保存
plt.tight_layout()
plt.savefig('dote_privacy_combined_fixed.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
plt.savefig('dote_privacy_combined_fixed.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
plt.show()