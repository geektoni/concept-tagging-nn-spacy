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
export seed=${14}
export max_iters=${15}
export char_emb=${16}
export char_emb_file=${17}

export mf=False
if [ ${more_features} == "--more-features" ]; then
  export mf=True
fi

export cemb=False
if [ ${char_emb} == "--c2v" ]; then
  export cemb=True
fi

embedder_rep=$embedder

if [ ${embedder} == "glove" ]; then
  export embedder="none"
  embedder_rep="glove"
fi

if [ ${embedder} == "conceptnet" ]; then
  export embedder="none"
  embedder_rep="conceptnet"
fi

if [ ${embedder} == "elmo-combined" ]; then
  export embedder="elmo-combined"
  embedder_rep="elmo_combined"
fi

export result_name=${model_type}-${hidden}-${epochs}-${batch_size}-${lr}-${drop_rate}-${emb_norm}-${embedder_rep}-${mf}-${cemb}-${seed}

qsub -V -N "$result_name" -q common_cpuQ train_models.sh
