#!/bin/bash

# 记录开始时间
start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Start Time: $start_time"

# 运行 Python 脚本
python3 data_preparation/create_dataset.py --dir CAUSALSIM_DIR-21-5-27/

# 记录结束时间
end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "End Time: $end_time"

# 计算运行时间
start_timestamp=$(date -d "$start_time" +%s)
end_timestamp=$(date -d "$end_time" +%s)
duration=$((end_timestamp - start_timestamp))

echo "Duration: $duration seconds"
