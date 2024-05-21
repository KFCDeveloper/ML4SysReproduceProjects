import matplotlib.pyplot as plt
import numpy as np

# 读取文件
file_path = "/mydata/DOTE/real_traffic/Brain/data_collecting_stat/collecting_stat copy.txt"
with open(file_path, "r") as file:
    data = [float(line.strip()) for line in file]

# 将数据排序
data_sorted = np.sort(data)

# 计算CDF
cdf = np.linspace(0, 1, len(data_sorted))

# 绘制CDF图
plt.figure(figsize=(10, 6))
plt.plot(data_sorted, cdf, marker='o', linestyle='-')
plt.title('Topo Brain Labeling CDF')
plt.xlabel('Time(s)')
plt.ylabel('Probability')
plt.grid(True)

# 保存图形
plt.savefig('/mydata/DOTE/real_traffic/Brain/data_collecting_stat/cdf_plot.png')

# 显示图形
plt.show()
