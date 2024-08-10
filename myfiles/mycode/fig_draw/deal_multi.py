import os
import re
from multiprocessing import Pool, cpu_count
import bisect
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import glob





def find_index(sorted_list, value):
    index = bisect.bisect_left(sorted_list, value)
    if index < len(sorted_list) and sorted_list[index] == value:
        return index  # 找到了精确匹配的值
    else:
        # 返回最接近的索引
        if index == 0:
            return index  # 如果在最前面
        elif index == len(sorted_list):
            return index - 1  # 如果在最后面
        else:
            # 比较前后两个元素哪个更接近
            if value - sorted_list[index - 1] <= sorted_list[index] - value:
                return index - 1
            else:
                return index

def read_target_times(file_path):
    with open(file_path, 'r') as file:
        start_times = []
        end_times = []

        for line in file:
            parts = line.split()
            start = float(parts[0])
            duration = float(parts[1])
            start_times.append(start)
            end_times.append(start + duration)
    return start_times, end_times

       

# Function to process each file
def process_file(file_path):
    virtual_time_list = []
    real_time_list = []
    virtual_time_pattern = re.compile(r'\[([0-9]+\.[0-9]+)\]')
    real_time_pattern = re.compile(r'real time: ([0-9]+\.[0-9]+)s')

    try:
        with open(file_path, 'r') as f:
            total_lines = sum(1 for line in f)
            f.seek(0)
            for line in f: # tqdm(f, total=total_lines, desc=f"Processing {file_path}"):
                virtual_match = virtual_time_pattern.search(line)
                real_match = real_time_pattern.search(line)
                if virtual_match and real_match:
                    virtual_time_list.append(float(virtual_match.group(1)))
                    real_time_list.append(float(real_match.group(1)))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return (virtual_time_list, real_time_list)



def draw_a_line(config_path):
    
    # Define the directories to process
    # 匹配目录中的所有符合条件的子目录
    base_dir = f"/mydata/MimicNet/data/{config_path}"
    directories = glob.glob(os.path.join(base_dir, 'edges*')) + \
                glob.glob(os.path.join(base_dir, 'eval*')) + \
                glob.glob(os.path.join(base_dir, 'hosts*'))

    # Gather all files to process
    files_to_process = []
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.pdmp'):
                    file_path = os.path.join(root, file)
                    files_to_process.append(file_path)

    # Number of processes to use (defaulting to number of CPU cores)
    num_processes = cpu_count()

    # Initialize a pool of workers
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.map(process_file, files_to_process), total=len(files_to_process), desc="Processing files"))

    # Collect results from all processes
    virtual_time_list = []
    real_time_list = []
    for virtual_times, real_times in results:
        virtual_time_list.extend(virtual_times)
        real_time_list.extend(real_times)

    virtual_time_list.sort()
    real_time_list.sort()


    # 读取 out_target_time.dat 文件
    start_times, end_times = read_target_times("/mydata/MimicNet/data/" + config_path + "/out_target_time.dat")

    indices = []
    target_values = []

    time_in_sys = []
    for times_index in tqdm(range(len(start_times)), desc="Processing times"):
        index = find_index(virtual_time_list, start_times[times_index])
        real_time_start = real_time_list[index]
        index2 = find_index(virtual_time_list, end_times[times_index])
        real_time_out = real_time_list[index2]
        time_in_sys.append(real_time_out - real_time_start)



    # 对数据进行排序
    sorted_data = np.sort(time_in_sys)
    sorted_data = [x for x in sorted_data if x < 0.7]
    # 计算CDF的值
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data,cdf



config_paths = ["sw2_cl2_sv4_l0.70_L100e6_s0_qDropTailQueue_vTCPNewReno_S20_tcp", "sw2_cl2_sv32_l0.70_L100e6_s0_qDropTailQueue_vTCPNewReno_S20_tcp", "sw2_cl2_sv64_l0.70_L100e6_s0_qDropTailQueue_vTCPNewReno_S20_tcp","sw2_cl4_sv8_l0.70_L100e6_s0_qDropTailQueue_vTCPNewReno_S20_tcp","sw4_cl2_sv4_l0.70_L100e6_s0_qDropTailQueue_vTCPNewReno_S20_tcp","sw2_cl2_sv8_l0.70_L100e6_s0_qDropTailQueue_vTCPNewReno_S20_tcpfirst"]  # 替换成实际的配置路径
all_sorted_data = []
all_cdf = []

for config_path in config_paths:
    sorted_data, cdf = draw_a_line(config_path)
    all_sorted_data.append(sorted_data)
    all_cdf.append(cdf)

# 绘制CDF图
plt.figure(figsize=(8, 6))

for sorted_data, cdf, config_path in zip(all_sorted_data, all_cdf, config_paths):
    label = config_path[:12]
    plt.plot(sorted_data, cdf, marker='.', linestyle='none', label=label)

# 添加图例
plt.legend()
plt.xlabel('Value')
plt.ylabel('CDF')
plt.title('CDF Plot')
plt.grid(True)
plt.savefig(os.path.join('/mydata/MimicNet/myfiles/myfigs', 'multi_cdf_plot.jpg'))

# config_path = "sw2_cl4_sv4_l0.70_L100e6_s0_qDropTailQueue_vTCPNewReno_S20_tcp"
# config_paths = []
# for con_p in config_paths:
#     sorted_data,cdf = draw_a_line(config_path)
# # 绘制CDF图
# plt.figure(figsize=(8, 6))
# plt.plot(sorted_data, cdf, marker='.', linestyle='none')
# plt.xlabel('Value')
# plt.ylabel('CDF')
# plt.title('CDF Plot')
# plt.grid(True)
# plt.savefig(os.path.join('/mydata/MimicNet/myfiles/myfigs', 'sw2_cl4_sv4cdf_plot.jpg'))