# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

# 数据
x = np.array([112, 202, 294,372,486, 1790])
y = np.logspace(-2, 3, 6)

# 绘图
plt.plot(x, y, marker='o')

# 设置坐标轴的对数刻度
plt.xscale('linear')
plt.yscale('log')

# 设置坐标轴标签
plt.xlabel('X轴')
plt.ylabel('Y轴')

# 设置标题
plt.title('折线图')

# 显示网格
plt.grid(True)

# 保存图片
plt.savefig('./line_plot.pdf')

# 显示图形
plt.show()
