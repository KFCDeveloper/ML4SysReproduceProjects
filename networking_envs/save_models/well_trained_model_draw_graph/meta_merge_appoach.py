import matplotlib.pyplot as plt

topo_name = "GEANT"  # "Arnes-1-('7', '23')" "Abilene0-1-('1', '10')"    "Abilene-2-('7', '8')-('9', '10')" "GEANT"
# 读取 direct_test.txt 文件的前25行数据
with open("../meta_test_"+ topo_name +"_attention.txt", "r") as file:
    meta_test_attention = [float(line.strip()) for line in file.readlines()[:50]]

# 读取 meta_test.txt 文件的前25行数据
with open("../meta_test_"+ topo_name +"_bilinear.txt", "r") as file:
    meta_test_bilinear = [float(line.strip()) for line in file.readlines()[:50]]

# 读取 meta_test.txt 文件的前25行数据
with open("../meta_test_"+ topo_name +"_concatenate.txt", "r") as file:
    meta_test_concatenate = [float(line.strip()) for line in file.readlines()[:50]]

# 读取 direct_test.txt 文件的前25行数据 !! 注意，并不是一个任务上
with open("../well_trained_test_GEANT.txt", "r") as file:
    well_trained = [float(line.strip()) for line in file.readlines()[:50]]

# 横坐标：num of batch
x = range(1, 51)

# 创建画布和子图
fig, ax = plt.subplots()

# 绘制 direct_test 折线图
ax.plot(x, meta_test_attention, label='attention')

# 绘制 meta_test 折线图
ax.plot(x, meta_test_bilinear, label='bilinear')

# 绘制 meta_test 折线图
ax.plot(x, meta_test_concatenate, label="concatenate")

# # 绘制 meta_test 折线图
ax.plot(x, well_trained, label="well_trained")



# 设置图例
ax.legend()

# 设置标题和轴标签
ax.set_title('Test Meta Learning', fontsize=14)
ax.set_xlabel('num of batch', fontsize=14)
ax.set_ylabel(r"$Avg(\frac{\hat{ MLU }}{MLU}) $ of a batch", fontsize=14)

# 保存图形到当前目录
plt.savefig('test_well_trained.jpg')

# 显示图形
plt.show()