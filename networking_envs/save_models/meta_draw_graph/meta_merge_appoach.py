import matplotlib.pyplot as plt

topo_name = "Arnes-1-('7', '23')"  # "Arnes-1-('7', '23')" "Abilene0-1-('1', '10')"    "Abilene-2-('7', '8')-('9', '10')"
# 读取 direct_test.txt 文件的前25行数据
with open("../meta_test_"+ topo_name +"_attention.txt", "r") as file:
    meta_test_attention = [float(line.strip()) for line in file.readlines()[:25]]

# 读取 meta_test.txt 文件的前25行数据
with open("../meta_test_"+ topo_name +"_bilinear.txt", "r") as file:
    meta_test_bilinear = [float(line.strip()) for line in file.readlines()[:25]]

# 读取 meta_test.txt 文件的前25行数据
with open("../meta_test_"+ topo_name +".txt", "r") as file:
    meta_test_concatenate = [float(line.strip()) for line in file.readlines()[:25]]

# 横坐标：num of batch
x = range(1, 26)

# 创建画布和子图
fig, ax = plt.subplots()

# 绘制 direct_test 折线图
ax.plot(x, meta_test_attention, label='attention')

# 绘制 meta_test 折线图
ax.plot(x, meta_test_bilinear, label='bilinear')

# 绘制 meta_test 折线图
ax.plot(x, meta_test_concatenate, label="concatenate")

# # 绘制 meta_test 折线图
# ax.plot(x, meta_test_Arnes_lines, label="meta_test_Arnes")



# 设置图例
ax.legend()

# 设置标题和轴标签
ax.set_title('Test Meta Learning', fontsize=14)
ax.set_xlabel('num of batch', fontsize=14)
ax.set_ylabel(r"$Avg(\frac{\hat{ MLU }}{MLU}) $ of a batch", fontsize=14)

# 保存图形到当前目录
plt.savefig('test_merge_vector_arne.jpg')

# 显示图形
plt.show()