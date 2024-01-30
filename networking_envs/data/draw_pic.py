# coding=utf-8
import re
import matplotlib.pyplot as plt

# 存储提取的数值
values = []

# 循环处理每个文件
for i in range(200):
    file_path = f'/mydata/DOTE/networking_envs/data/Cogentco/so_stats_{i}.txt'

    # 打开文件并读取第五行
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if len(lines) >= 5:
            fifth_line = lines[4]
            # 使用正则表达式提取小数部分
            match = re.search(r'90TH: (\d+\.\d+)', fifth_line)
            if match:
                value = float(match.group(1))
                values.append(value)

# 绘制折线图
plt.plot(range(1, 201), values, marker='o', linestyle='-')
plt.xlabel('File Index')
plt.ylabel('Value')
plt.title('90TH Value from so_stat Files')

# 保存图形
plt.savefig('90th_values_plot.png')

# 显示图形（可选）
plt.show()
