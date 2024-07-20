#!/bin/bash
# cd /mydata/MimicNet
BASE_DIR=`pwd`
RESULTS_DIR=results/sw4_cl2_sv4_l0.70_L100e6_s0_qDropTailQueue_vTCPNewReno_S20_tcp
RESULTS_FILE=${RESULTS_DIR##*/}
echo ${RESULTS_FILE}
rsync -a ${RESULTS_DIR}/ ${BASE_DIR}/data/${RESULTS_FILE}
rm -r ${RESULTS_DIR}

if [[ ${RESULTS_DIR} =~ /sw([0-9]+)_ ]]; then
    NUM_SWITCHES=${BASH_REMATCH[1]}
else
    echo "ERROR: unable to extract switch count from ${RESULTS_DIR}"
    exit 1
fi
if [[ ${RESULTS_DIR} =~ _s([0-9]+)_ ]]; then
    SEED=${BASH_REMATCH[1]}
else
    echo "ERROR: unable to extract seed from ${RESULTS_DIR}"
    exit 1
fi

if [[ ${RESULTS_DIR} =~ _cl([0-9]+)_ ]]; then # ydy: to match cluster number
    NUM_CLUSTER=${BASH_REMATCH[1]}
else
    echo "ERROR: unable to extract seed from ${RESULTS_DIR}"
    exit 1
fi

cd ${BASE_DIR}
echo "prepare/run_prepare.sh data/${RESULTS_FILE} ${SEED} ${NUM_SWITCHES} ${NUM_CLUSTER}"
prepare/run_prepare.sh data/${RESULTS_FILE} ${SEED} ${NUM_SWITCHES} ${NUM_CLUSTER}

echo "Generate complete! Results are in the following directory:"
echo "data/${RESULTS_FILE}"