#!/bin/bash
#PBS -l select=3:ncpus=10:mem=5GB
#PBS -l walltime=48:0:0
#PBS -q common_cpuQ
#PBS -M giovanni.detoni@studenti.unitn.it
#PBS -V
#PBS -m be

# Strict bash mode
set -euo pipefail
IFS=$'\n\t'

mf=False
if [ ${more_features} == "--more-features" ]; then
  mf=True
fi

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