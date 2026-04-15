#!/bin/bash
DATASET_NAME=$1
INDEX=$2
CHECKPOINT=$3
TOPK=4

# path to lerf_ovs/label
DATASET_ROOT_PATH=../../data/Mip-NeRF360
gt_folder=../../data/Mip-NeRF360-OVS

ROOT_PATH="."

python eval_mip_nerf360.py \
    -s ${DATASET_ROOT_PATH}/${DATASET_NAME} \
    -m "${ROOT_PATH}/output/${DATASET_NAME}_${INDEX}_1" \
    --dataset_name ${DATASET_NAME} \
    --index ${INDEX} \
    --ckpt_root_path ${ROOT_PATH}/output \
    --output_dir ${ROOT_PATH}/eval_result \
    --mask_thresh 0.4 \
    --json_folder ${gt_folder} \
    --checkpoint ${CHECKPOINT} \
    --include_feature \
    --topk ${TOPK} \
    --quick_render