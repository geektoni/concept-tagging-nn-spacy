#!/bin/bash

export model_type=${1}
export hidden=${2}
export epochs=${3}
export batch_size=${4}
export lr=${5}
export drop_rate=${6}
export emb_norm=${7}
export embedder=${8}
export more_features=${9}
export train_file=${10}
export test_file=${11}
export embedding_files=${12}
export result_file_name=${13}

export result_name=${model_type}-${hidden}-${epochs}-${batch_size}-${lr}-${drop_rate}-${emb_norm}-${embedder}-${mf}

qsub -V -N "$result_name" -q common_cpuQ train_models.sh
