file_path = '/mydata/ns.py/2.txt'

# 读取文件的前 10000 行数据
with open(file_path, 'r') as file:
    lines = [next(file) for _ in range(10000)]

# 打印前 10000 行数据
for line in lines:
    print(line, end='')