# import matplotlib.pyplot as plt

# # 读取 direct_test.txt 文件的前25行数据
# with open("direct_test.txt", "r") as file:
#     direct_test_lines = [float(line.strip()) for line in file.readlines()[:25]]

# # 读取 meta_test.txt 文件的前25行数据
# with open("meta_test.txt", "r") as file:
#     meta_test_lines = [float(line.strip()) for line in file.readlines()[:25]]

# # 横坐标：num of batch
# x = range(1, 26)

# # 创建画布和子图
# fig, ax = plt.subplots()

# # 绘制 direct_test 折线图
# ax.plot(x, direct_test_lines, label='direct_test')

# # 绘制 meta_test 折线图
# ax.plot(x, meta_test_lines, label='meta_test')

# # 设置图例
# ax.legend()

# # 设置标题和轴标签
# ax.set_title('Test Meta Learning')
# ax.set_xlabel('num of batch')
# ax.set_ylabel('Loss')

# # 保存图形到当前目录
# plt.savefig('test_loss_plot.pdf')

# # 显示图形
# plt.show()
import matplotlib.pyplot as plt

# 读取 direct_test.txt 文件的前25行数据
with open("direct_test.txt", "r") as file:
    direct_test_lines = [float(line.strip()) for line in file.readlines()[:25]]

# 读取 meta_test.txt 文件的前25行数据
with open("meta_test.txt", "r") as file:
    meta_test_lines = [float(line.strip()) for line in file.readlines()[:25]]

# 读取 meta_test.txt 文件的前25行数据
with open("meta_test_Arnes.txt", "r") as file:
    meta_test_Arnes_lines = [float(line.strip()) for line in file.readlines()[:25]]

# 读取 meta_test.txt 文件的前25行数据
with open("meta_test_Abilene0.txt", "r") as file:
    meta_test_Abilene0_lines = [float(line.strip()) for line in file.readlines()[:25]]

# 横坐标：num of batch
x = range(1, 26)

# 创建画布和子图
fig, ax = plt.subplots()

# 绘制 direct_test 折线图
ax.plot(x, direct_test_lines, label='direct_test')

# 绘制 meta_test 折线图
ax.plot(x, meta_test_lines, label='meta_test_Abilene')

# 绘制 meta_test 折线图
ax.plot(x, meta_test_Abilene0_lines, label="meta_test_Abilene'")

# 绘制 meta_test 折线图
ax.plot(x, meta_test_Arnes_lines, label="meta_test_Arnes")



# 设置图例
ax.legend()

# 设置标题和轴标签
ax.set_title('Test Meta Learning', fontsize=14)
ax.set_xlabel('num of epochs', fontsize=14)
ax.set_ylabel(r"$Avg(\frac{\hat{ MLU }}{MLU}) $ of a batch", fontsize=14)

# 保存图形到当前目录
plt.savefig('test_loss_plot_mix.pdf')

# 显示图形
plt.show()