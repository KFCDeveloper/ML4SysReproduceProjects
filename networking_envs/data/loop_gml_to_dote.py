# Convert GML format to DOTE format
# ydy: use this file to gen a large number of task
import os
import numpy as np
from numpy import random
import sys

src_dir = "zoo_topologies"
topo_name = sys.argv[1]
first_edge_str = sys.argv[2]
second_edge_str = sys.argv[3]
network_name = topo_name    # "Abilene"
# remember to modify three places; second_edge_str| "-2-" | second_edge_tuple
dest_dir = network_name + "-obj2-2-" + first_edge_str + "-" + second_edge_str # remember to modify three places;  "-squeeze-links-more1"  || + "-" + second_edge_str

# additional configuration variables
default_capacity = "10000.0"
demands_factor = 10.0
n_train_matrices = 3
n_test_martices = 1

input_file_name = src_dir + '/' + network_name + ".gml"

nodes = set()
edges = {}
assert os.path.exists(input_file_name)
with open(input_file_name) as f:
    while True:
        line = f.readline().strip()
        if not line: break
        if line.startswith("node ["):  # read node info
            node_id = None
            while not line.startswith("]"):
                if line.startswith("id "):
                    assert node_id == None  # single id per node
                    node_id = line[len("id "):]
                line = f.readline().strip()

            assert node_id != None and node_id not in nodes
            nodes.add(node_id)
            continue

        elif line.startswith("edge ["):
            edge_src = None
            edge_dst = None
            capacity = default_capacity
            while not line.startswith("]"):
                if line.startswith("source "):
                    assert edge_src == None
                    edge_src = line[len("source "):]
                elif line.startswith("target "):
                    assert edge_dst == None
                    edge_dst = line[len("target "):]
                elif line.startswith("LinkSpeedRaw "):
                    capacity = line[len("LinkSpeedRaw "):]
                line = f.readline().strip()

            assert edge_src != None and edge_dst != None
            edges[(edge_src, edge_dst)] = capacity
            continue

# change topo, cut some links  Cap (4,5) Cap(7,10)
# del edges[('4','5')]
# del edges[('7','10')]
# edges[('4','5')] = edges[('7','10')] = '1.0'
# change topo, cut a node  cut node '3'
# del edges[('1','10')]
# del edges[('7','10')]
# del edges[('9','10')]
# nodes.remove('10')
# edges[('1','10')] = edges[('7','10')] = edges[('9','10')] = '1.0'

# verification:
# 1. nodes are numbered 0 to len(nodes)
# 2. edges src and targets are existing nodes
for i in range(len(nodes)): assert (str(i)) in nodes
for e in edges: assert e[0] in nodes and e[1] in nodes

# Convert to DOTE format
assert not os.path.exists(dest_dir)
os.mkdir(dest_dir)
os.mkdir(dest_dir + "/opts")
os.mkdir(dest_dir + "/test")
os.mkdir(dest_dir + "/train")

# **** save the cap and link ****
# edges_list = [(int(e[0]), int(e[1]), edges[e]) for e in edges]
# edges_list.sort()

# with open(dest_dir + '/' + dest_dir + "_int.pickle.nnet", "w", newline='\n') as f:
#     for e in edges_list:
#         f.write(str(e[0]) + ',' + str(e[1]) + ',' + e[2] + '\n')

# with open(dest_dir + '/' + dest_dir + "_int.txt", "w", newline='\n') as f:
#     capacities = [["0.0"]*len(nodes) for x in range(len(nodes))]
#     for e in edges_list:
#         capacities[e[0]][e[1]] = e[2]
#         if (str(e[1]), str(e[0])) not in edges:
#             capacities[e[1]][e[0]] = e[2]

#     for i in range(len(nodes)):
#         f.write(','.join(capacities[i]) + '\n')

# generate random traffic matrices
node_to_n_edges = {}
total_edges_cap = 0.0  # just use it to gen proper traffic matrix
for n in nodes: node_to_n_edges[n] = 0
for e in edges:
    total_edges_cap += float(edges[e])
    if edges[e] == '1.0':  # if edge cap is 0.0
        continue
    node_to_n_edges[e[0]] += 1
    if (e[1], e[0]) not in edges:
        node_to_n_edges[e[1]] += 1
        total_edges_cap += float(edges[e])

total_edges = sum(node_to_n_edges.values())

# print("#nodes = {0}, #edges = {1}, total capacity {2}".format(str(len(nodes)), str(len(edges)), str(total_edges_cap)))

total_demands = total_edges_cap / demands_factor
frac_dict = {}
# for u in range(len(nodes)):
#     for v in range(len(nodes)):
#         if nodes[u]=='10' or nodes[v]=='10':
#             frac_dict[(u, v)] = 0.0

#         if u == v:
#             frac_dict[(u, v)] = 0.0
#         else:
#             u_str = str(u)
#             v_str = str(v)
#             frac_dict[(u, v)] = (node_to_n_edges[u_str] * node_to_n_edges[v_str]) / (total_edges * (total_edges - node_to_n_edges[u_str]))

node_list = list(nodes)
for u in range(len(node_list)):
    for v in range(len(node_list)):
        if u == v:
            frac_dict[(u, v)] = 0.0
        else:
            u_str = str(u)
            v_str = str(v)
            frac_dict[(u, v)] = (node_to_n_edges[u_str] * node_to_n_edges[v_str]) / (
                        total_edges * (total_edges - node_to_n_edges[u_str]))

        # if node_list[u]=='10' or node_list[v]=='10':    # remove node 10
        #     frac_dict[(u, v)] = 0.0
        # if (node_list[u],node_list[v])==('4','5') or (node_list[u],node_list[v])==('5','4') or (node_list[u],node_list[v])==('7','10') or (node_list[u],node_list[v])==('10','7'):
        #     frac_dict[(u, v)] = 0.0

# i'd like to change bandwidth of some paths
# double_bw_path = {('4','3'),('5','6'),('8','9')}
# double_bw_path = {('8', '5'), ('9', '12'), ('9', '11'), ('10', '9'), ('10', '12'), ('8', '6'), ('0', '12'), ('0', '11'), ('6', '7'), ('1', '9'), ('1', '7'), ('6', '9'), ('12', '7'), ('1', '12'), ('2', '8'), ('1', '3'), ('5', '8'), ('7', '10'), ('4', '12'), ('6', '10'), ('11', '10'), ('9', '8'), ('10', '5'), ('0', '8'), ('0', '1'), ('0', '5'), ('3', '4'), ('8', '10'), ('2', '9'), ('2', '7'), ('6', '5'), ('7', '6'), ('5', '9'), ('11', '1'), ('11', '5'), ('2', '11'), ('3', '5'), ('5', '11'), ('5', '3'), ('6', '0'), ('0', '2'), ('4', '1'), ('12', '6')}
# gen train and test DMs
for m_idx in range(1, n_train_matrices + n_test_martices + 1):
    if m_idx <= n_train_matrices:
        fname = dest_dir + '/train/' + str(m_idx) + '.hist'
    else:
        fname = dest_dir + '/test/' + str(m_idx) + '.hist'
    f = open(fname, 'w')
    for dm_idx in range(2016):
        demands = ["0.0"] * (len(nodes) * len(nodes))
        for u in range(len(nodes)):
            for v in range(len(nodes)):
                if u == v: continue
                frac = frac_dict[(u, v)]
                if frac == 0.0: continue  # remove links
                # sample from gaussian with mean = frac, stddev = frac / 4   # ydy: to just use half of bandwidth  * 0.5; frac / 2
                demands[u * len(nodes) + v] = f"{(total_demands * max(np.random.normal(frac, frac / 4), 0.0)):.6g}"
                # change bandwidth of some paths
                # if (list(nodes)[u],list(nodes)[v]) in double_bw_path:
                #     demands[u*len(nodes) + v] = f"{(total_demands *2 *max(np.random.normal(frac, frac / 4), 0.0)):.6g}"
        f.write(' '.join(demands) + '\n')
    f.close()

# save the cap and link
# change the cap
# edges[('1','10')]
first_edge_tuple = tuple(first_edge_str.strip("()").replace("'", "").split(", "))
second_edge_tuple = tuple(second_edge_str.strip("()").replace("'", "").split(", "))  # edges[second_edge_tuple] = 
edges[first_edge_tuple] = str(float(edges[first_edge_tuple])/2) # '5000.0'
edges[second_edge_tuple] = str(float(edges[second_edge_tuple])/2) # '5000.0'

edges_list = [(int(e[0]), int(e[1]), edges[e]) for e in edges]
edges_list.sort()

with open(dest_dir + '/' + dest_dir + "_int.pickle.nnet", "w", newline='\n') as f:
    for e in edges_list:
        f.write(str(e[0]) + ',' + str(e[1]) + ',' + e[2] + '\n')

with open(dest_dir + '/' + dest_dir + "_int.txt", "w", newline='\n') as f:
    capacities = [["0.0"] * len(nodes) for x in range(len(nodes))]
    for e in edges_list:
        capacities[e[0]][e[1]] = e[2]
        if (str(e[1]), str(e[0])) not in edges:
            capacities[e[1]][e[0]] = e[2]

    for i in range(len(nodes)):
        f.write(','.join(capacities[i]) + '\n')