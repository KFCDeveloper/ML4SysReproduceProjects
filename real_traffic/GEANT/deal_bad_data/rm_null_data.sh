#!/bin/bash
# 切换到目标目录
cd /mydata/software/GEANT/directed-geant-uhlig-15min-over-4months-ALL-native

# 循环遍历目录中的所有文件
for file in *; do
    # 检查是否是文件
    if [ -f "$file" ]; then
        # 计算文件的行数
        line_count=$(wc -l < "$file")
        # 如果行数少于 100 行，则删除文件
        if [ "$line_count" -lt 100 ]; then
            echo "Deleting $file with $line_count lines."
            rm "$file"
        fi
    fi
done
