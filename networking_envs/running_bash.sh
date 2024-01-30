#!/bin/bash
for i in {0..199}; do
    echo $i
done

# 构建命令并执行
cmd="python3 /mydata/DOTE/dote.py --ecmp_topo Cogentco --paths_from sp --so_mode train --so_epochs 200 --so_batch_size 32 --opt_function MAXUTIL"

# 打印并执行命令
echo "Running command for i=$i:"
echo $cmd
$cmd

# 休眠一秒（可选，用于控制速度）
sleep 1


for i in {0..199}; do
    # 构建命令并执行
    cmd="python3 /mydata/DOTE/dote.py --ecmp_topo Cogentco --paths_from sp --so_mode test --so_epochs $i --so_batch_size 32 --opt_function MAXUTIL"

    # 打印并执行命令
    echo "Running command for i=$i:"
    echo $cmd
    $cmd

    # 休眠一秒（可选，用于控制速度）
    sleep 1
done
