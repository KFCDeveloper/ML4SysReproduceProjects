import os
import random

total_nodes = 13
change_set = set()

for i in range(total_nodes):
    for j in range(total_nodes):
        if i == j: 
            continue
        if random.random() <= 0.3:
            change_set.add((str(i),str(j)))
        

print(change_set)
