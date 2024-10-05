import os
import numpy as np
import matplotlib.pyplot as plt

trainX = []

# 创建保存图片的文件夹
output_dir = "/mydata/flow-prediction/myfile/RF/cdf_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历矩阵的每一列，绘制CDF图并保存
for i in range(trainX.shape[1]):
    column_data = trainX[:, i].flatten()
    
    # 对数据进行排序
    data_sorted = np.sort(column_data)
    
    # 计算CDF
    cdf = np.arange(1, len(data_sorted)+1) / len(data_sorted)
    
    # 绘制CDF图
    plt.figure(figsize=(8, 6))
    plt.plot(data_sorted, cdf, marker='.', linestyle='none')
    plt.title(f"CDF of column {i+1}")
    plt.xlabel("Data points")
    plt.ylabel("CDF")
    
    # 保存图片
    plt.savefig(os.path.join(output_dir, f"cdf_column_{i+1}.png"))
    plt.close()

print(f"CDF plots saved in {output_dir}")