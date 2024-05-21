import os
import pathlib
import math
import urllib.request
import re

# one_year (time slot 1 hour) data https://sndlib.put.poznan.pl/download/directed-brain-1h-over-375day-native.tgz
# one week (time slot 1 min) data https://sndlib.put.poznan.pl/download/directed-brain-1min-over-7days-native.tgz
# topology_file = "http://informatique.umons.ac.be/networks/igen/downloads/geant-20041125.gml"
topology_file = "/mydata/DOTE/networking_envs/data/zoo_topologies/Brain.txt"
demand_matrices_dir = "/mydata/DOTE/real_traffic/download_data/Brain/one_year/SNDLIBroundedNative"
output_dir = "Brain-obj1"

train_fraction = 0.75 #fraction of DMs for the training dataset
num_of_dms_per_file = 2016
first_file_id = 1

#read nodes
#demand matrices
dms_input_files = os.listdir(demand_matrices_dir)
dms_input_files.sort()

nodes = {}
edges = {}
topology_name_to_dm_name = {}
# 读取拓扑，我后面的代码貌似只用读cap就行了，所以，我下面的代码写重了，其实不用写，但是不想改了
# with open(demand_matrices_dir + '/' + dms_input_files[0]) as f:
#     line = f.readline().strip()
#     while not line.startswith("NODES ("): line = f.readline().strip()
#     line = f.readline().strip()
#     while not line.startswith(")"):
#         node_name = line.split()[0]
#         nodes[node_name] = len(nodes)
#         #topology_name = "SL" if node_name == 'si1.si' else node_name.split('.')[1].upper()
#         topology_name = node_name.split('.')[0]
#         topology_name_to_dm_name[topology_name] = node_name
#         if topology_name == "de1": topology_name_to_dm_name["de2"] = node_name
#         line = f.readline().strip()

topology_id_to_dm_id = {}
if topology_file.startswith('http'):
    f = urllib.request.urlopen(topology_file)
    lines = f.read().splitlines()
    f.close()
    lines = [line.decode('utf-8').strip() for line in lines]
else:
    f = open(topology_file)
    lines = f.read().splitlines()
    f.close()
    lines = [line.strip() for line in lines]

i = 0
while i < len(lines):
    if lines[i].startswith("NODES ("):  # 开始读取 NODES
        i += 1
        temp_id = 0
        while not lines[i].startswith(")"):   # 逐行遍历 NODES，直到读到 ')'
            node_id = None
            node_name = None
            matches = re.match(r"(\w+)\s*\(\s*([\d.]+)\s*([\d.]+)\s*\)", lines[i])
            # 提取匹配到的三个字符串
            if matches:
                name = matches.group(1)
                latitude = matches.group(2)
                longitude = matches.group(3)
                nodes[name] = temp_id
            i += 1
            temp_id += 1
    elif lines[i].startswith("LINKS ("):  # 开始读取 LINKS
        i += 1
        while not lines[i].startswith(")"):   # 逐行遍历 NODES，直到读到 ')'
            edge_src = None
            edge_dst = None
            edge_cap = None
            # 使用正则表达式解析字符串
            # 使用正则表达式匹配子字符串
            matches = re.findall(r'\b\S+\b', lines[i])
            # 提取匹配到的三个字符串
            if matches:
                # 使用正则表达式匹配子字符串
                edge_src = matches[1]
                edge_dst = matches[2]
                edge_cap = matches[7]
                dm_src = nodes[edge_src]
                dm_dst = nodes[edge_dst]
                if dm_src != dm_dst:
                    edges[(str(min(dm_src, dm_dst)), str(max(dm_src, dm_dst)))] = edge_cap
            i += 1
        break   # 后面没有要读取的东西了
    i += 1
# ——————————————————deal with topology gml file ————————————————————————————————————————————
# i = 0
# while i < len(lines):
#     if lines[i].startswith("node ["):
#         node_id = None
#         node_name = None
#         while not lines[i].startswith("]"):
#             i += 1
#             if lines[i].startswith("id"):
#                 assert node_id == None
#                 node_id = lines[i].split()[1]
#             elif lines[i].startswith("name") or lines[i].startswith("label"):
#                 assert node_name == None
#                 node_name = lines[i].split()[1][1:-1]
#         assert node_id != None and node_name != None
#         if node_name in topology_name_to_dm_name:
#             assert node_id not in topology_id_to_dm_id
#             topology_id_to_dm_id[node_id] = nodes[topology_name_to_dm_name[node_name]]
#     elif lines[i].startswith("edge ["):
#         edge_src = None
#         edge_dst = None
#         edge_cap = None
#         while not lines[i].startswith("]"):
#             i += 1
#             if lines[i].startswith("source"):
#                 assert edge_src == None
#                 edge_src = lines[i].split()[1]
#             elif lines[i].startswith("target"):
#                 assert edge_dst == None
#                 edge_dst = lines[i].split()[1]
#             elif lines[i].startswith("LinkSpeedRaw") or lines[i].startswith("bandwidth"):
#                 assert edge_cap == None
#                 edge_cap = lines[i].split()[1]
#         assert edge_src != None and edge_dst != None and edge_cap != None
#         if edge_src in topology_id_to_dm_id and edge_dst in topology_id_to_dm_id:
#             dm_src = int(topology_id_to_dm_id[edge_src])
#             dm_dst = int(topology_id_to_dm_id[edge_dst])
#             if dm_src != dm_dst:
#                 edges[(str(min(dm_src, dm_dst)), str(max(dm_src, dm_dst)))] = edge_cap

#     i += 1
# ——————————————————————————————————————————————————————————————
edges_list = [(int(e[0]), int(e[1]), edges[e]) for e in edges]
edges_list.sort()
    
assert not os.path.exists(output_dir)
os.mkdir(output_dir)
os.mkdir(output_dir + "/opts")
os.mkdir(output_dir + "/test")
os.mkdir(output_dir + "/train")

with open(output_dir + '/' + pathlib.PurePath(output_dir).name + "_int.pickle.nnet", "w", newline='\n') as f:
    for e in edges_list:
        f.write(str(e[0]) + ',' + str(e[1]) + ',' + e[2] + '\n')

with open(output_dir + '/' + pathlib.PurePath(output_dir).name + "_int.txt", "w", newline='\n') as f:
    capacities = [["0.0"]*len(nodes) for _ in range(len(nodes))]
    for e in edges_list:
        capacities[e[0]][e[1]] = e[2]
        if (str(e[1]), str(e[0])) not in edges:
            capacities[e[1]][e[0]] = e[2]
    
    for i in range(len(nodes)):
        f.write(','.join(capacities[i]) + '\n')

#demand matrices
demands = [["0.0"]*(len(nodes)*len(nodes)) for _ in range(len(dms_input_files))]
count = 0
for i in range(len(dms_input_files)):
    file = dms_input_files[i]
    assert file.endswith('.txt')
    is_empty = True
    with open(demand_matrices_dir + '/' + file) as f:
        line = f.readline().strip()
        while not line.startswith("DEMANDS ("): line = f.readline().strip()
        line = f.readline().strip()
        while not line.startswith(")"):
            is_empty = False
            demand_info = line.split()
            src = demand_info[2]
            dst = demand_info[3]
            demand = demand_info[6]
            assert src in nodes and dst in nodes and demand_info[0] == src + '_' + dst and float(demand) >= 0.0
            demands[count][nodes[src]*len(nodes) + nodes[dst]] = repr(float(demand)*10)    # 1e6 现在改成这个了，不然结果有问题
            line = f.readline().strip()
    if is_empty:
        continue
    count += 1

#print("Number of demand matrices: " + str(count))

f_dm_idx = first_file_id
n_train_dms = int(math.ceil(count * train_fraction))
f = None
n_dms_in_f = 0
for i in range(count):
    if (n_dms_in_f % num_of_dms_per_file) == 0 or i == n_train_dms:
        if f != None: f.close()
        train_test = 'train' if i < n_train_dms else 'test'
        f = open(output_dir + '/' + train_test + '/' + str(f_dm_idx) + '.hist', 'w')
        f_dm_idx += 1
        n_dms_in_f = 0
        
    f.write(' '.join(demands[i]) + '\n')
    n_dms_in_f += 1
