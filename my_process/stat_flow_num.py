import pandas as pd

# 读取 CSV 文件并创建 DataFrame
file_path = "/mydata/DQN/data/4-port switch/FIFO/_traces/_train/4port8link7/4port8link7_0.1_trace1.csv"
df = pd.read_csv(file_path)

# 根据 src 和 dst 列进行分组
grouped_df = df.groupby(['pkt len (byte)']) # grouped_df = df.groupby(['src', 'dst'])

# 查看分组的数量
num_groups = len(grouped_df)
print("Number of groups:", num_groups)
