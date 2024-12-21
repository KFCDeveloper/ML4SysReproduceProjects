import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# print(fm.findSystemFonts(fontpaths=None, fontext='ttf'))

# 文件路径
train_file_path = "/mydata/DOTE/networking_envs/save_models/24-12-prog-direct-tran/log_dir/prog_train_test_losses.txt"
test_file_path = "/mydata/DOTE/networking_envs/save_models/24-12-prog-direct-tran/log_dir/test_losses.txt"

# 设置全局字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 25

# 读取数据
def read_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(float(line.strip()))
    return data

train_data = read_data(train_file_path)
test_data = read_data(test_file_path)

# 检查数据数量
assert len(train_data) == 6000, f"Expected 6000 train data points, got {len(train_data)}."
assert len(test_data) == 6000, f"Expected 6000 test data points, got {len(test_data)}."

# 定义窗口大小
window_size = 1500

# 计算累计平均
def calculate_cumulative_averages(data, window_size):
    cumulative_averages = []
    cumulative_sum = 0
    count = 0

    for i, value in enumerate(data):
        cumulative_sum += value
        count += 1
        cumulative_averages.append(cumulative_sum / count)  # 当前点的累计平均
        
        # 每到达窗口大小，重置累计变量
        if (i + 1) % window_size == 0:
            cumulative_sum = 0
            count = 0
    return cumulative_averages

train_cumulative_averages = calculate_cumulative_averages(train_data, window_size)
test_cumulative_averages = calculate_cumulative_averages(test_data, window_size)

# 绘制折线图
plt.figure(figsize=(12, 6))

# 绘制训练和测试的累计平均曲线
plt.plot(range(len(train_data)), train_cumulative_averages, label="Direct Transfer", color="#E74C3C",linewidth=3) 
plt.plot(range(len(test_data)), test_cumulative_averages, label="No Transfer", color="#4EA1D3",linewidth=3)

# 添加竖直虚线
for x in range(window_size, len(train_data), window_size):
    plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.8)

# for x in range(window_size, len(train_data), window_size):
#     plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.8, label="Interval Line" if x == window_size else "")

# 图例和轴标签
plt.xlabel("Timeline")
plt.ylabel("Accuracy (Cumulative Average)")
plt.title("Cumulative accuracy of each distribution")
plt.legend()
plt.grid(alpha=0.3)

# 保存并显示
plt.savefig("train_test_cumulative_average_accuracy_plot.pdf", bbox_inches="tight", pad_inches=0.01)  # 可选：保存图片
plt.show()


# cd /mydata/DOTE/networking_envs/save_models/24-12-prog-direct-tran/draw_pic
# python /mydata/DOTE/networking_envs/save_models/24-12-prog-direct-tran/draw_pic/draw_prog.py