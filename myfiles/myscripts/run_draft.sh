

./run_2_generate.sh "tcp" > /mydata/MimicNet/myfiles/mylogs/cl8_sv8_deg8.txt 2>&1

./run_2_generate.sh "tcp" > /mydata/MimicNet/myfiles/mylogs/cl2_sv32_deg2.txt 2>&1

./run_2_generate.sh "tcp" > /mydata/MimicNet/myfiles/mylogs/cl2_sv4_deg4.txt 2>&1



prepare/run_prepare.sh data/sw2_cl4_sv4_l0.70_L100e6_s0_qDropTailQueue_vTCPNewReno_S20_tcp 0 2 4

# to debug
# ./run_3_train.sh ${VARIANT} train/lstm/train_lstm_${VARIANT}.py ${RESULTS_DIR}
./run_3_train.sh tcp train/lstm/train_lstm_tcp.py ${RESULTS_DIR}


# 下面在debug sw8_cl8_sv8，因为最后run_prepare 基本没有运行，最后发现是有三个sort因为tmp是在系统盘上的，系统盘太小了导致失败。
# 给所有的 sort 加上 TMPDIR=/mydata/tmp,
# 在run_prepare 的时候，需要显示 Handled: 1, unmatched: 0, failed: 0 才是对的

prepare/parse_pdmps.sh data/sw8_cl8_sv8_l0.70_L100e6_s0_qDropTailQueue_vTCPNewReno_S20_tcp 8

prepare/run_prepare.sh data/sw8_cl8_sv8_l0.70_L100e6_s0_qDropTailQueue_vTCPNewReno_S20_tcp 0 8 8

python3 prepare/extract_features_tcp.py data/sw8_cl8_sv8_l0.70_L100e6_s0_qDropTailQueue_vTCPNewReno_S20_tcp 1.1. 0 8 --num_clusters 8


prepare/run_prepare.sh data/sw4_cl4_sv8_l0.70_L100e6_s0_qDropTailQueue_vTCPNewReno_S20_tcp 0 4 4