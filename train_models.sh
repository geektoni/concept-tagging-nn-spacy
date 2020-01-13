#!/bin/bash

# Strict bash mode
set -euo pipefail
IFS=$'\n\t'

echo ${13}

model_type=${1}
hidden=${2}
epochs=${3}
batch_size=${4}
lr=${5}
drop_rate=${6}
emb_norm=${7}
embedder=${8}
more_features=${9}
train_file=${10}
test_file=${11}
embedding_files=${12}
result_file_name=${13}

mf=False
if [ ${more_features} == "--more-features" ]; then
  mf=True
fi

result_name=${model_type}-${hidden}-${epochs}-${batch_size}-${lr}-${drop_rate}-${emb_norm}-${embedder}-${mf}
result_dir="results/$result_name"
mkdir -p ${result_dir}

cd ./concept-tagging-with-neural-networks/src

if [ ${more_features} == "--more-features" ]; then
  python run_model.py \
      --train ${train_file} \
      --test ${test_file} \
      --w2v ${embedding_files} \
      --model ${model_type} \
      --epochs ${epochs} \
      --write_results=$result_dir/${result_file_name}.res \
      --dev \
      --bidirectional \
      --more-features \
      --embedder ${embedder} \
      --batch ${batch_size} \
      --lr ${lr} \
      --hidden_size ${hidden} \
      --embedding_norm ${emb_norm} \
      --drop ${drop_rate}
      --unfreeze
else
  python run_model.py \
      --train ${train_file} \
      --test ${test_file} \
      --w2v ${embedding_files} \
      --model ${model_type} \
      --epochs ${epochs} \
      --write_results=$result_dir/${result_file_name}.res \
      --dev \
      --bidirectional \
      --embedder ${embedder} \
      --batch ${batch_size} \
      --lr ${lr} \
      --hidden_size ${hidden} \
      --embedding_norm ${emb_norm} \
      --drop ${drop_rate}
      --unfreeze
fi
