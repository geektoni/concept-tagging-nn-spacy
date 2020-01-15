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

cd concept-tagging-nn-spacy

result_dir="./results/$result_name"
mkdir -p ${result_dir}

cd ./concept-tagging-with-neural-networks/src

for i in $(seq 1 $max_iters)
do
  seed_used=$((seed+i))
  if [ ${char_emb} == "--c2v" ]; then
    if [ ${more_features} == "--more-features" ]; then
    python run_model.py \
      --train ${train_file} \
      --test ${test_file} \
      --w2v ${embedding_files} \
      --model ${model_type} \
      --epochs ${epochs} \
      --write_results=../../$result_dir/${result_file_name}_${i}.txt \
      --bidirectional \
      --more-features \
      --embedder ${embedder} \
      --batch ${batch_size} \
      --lr ${lr} \
      --hidden_size ${hidden} \
      --embedding_norm ${emb_norm} \
      --drop ${drop_rate} \
      --unfreeze \
      --seed ${seed_used} \
      --c2v ${char_emb_file}
  else
    python run_model.py \
      --train ${train_file} \
      --test ${test_file} \
      --w2v ${embedding_files} \
      --model ${model_type} \
      --epochs ${epochs} \
      --write_results=../../$result_dir/${result_file_name}_${i}.txt \
      --bidirectional \
      --embedder ${embedder} \
      --batch ${batch_size} \
      --lr ${lr} \
      --hidden_size ${hidden} \
      --embedding_norm ${emb_norm} \
      --drop ${drop_rate} \
      --unfreeze \
      --seed ${seed_used} \
      --c2v ${char_emb_file}
  fi
  else
    if [ ${more_features} == "--more-features" ]; then
      python run_model.py \
        --train ${train_file} \
        --test ${test_file} \
        --w2v ${embedding_files} \
        --model ${model_type} \
        --epochs ${epochs} \
        --write_results=../../$result_dir/${result_file_name}_${i}.txt \
        --bidirectional \
        --more-features \
        --embedder ${embedder} \
        --batch ${batch_size} \
        --lr ${lr} \
        --hidden_size ${hidden} \
        --embedding_norm ${emb_norm} \
        --drop ${drop_rate} \
        --unfreeze \
        --seed ${seed_used}
    else
      python run_model.py \
        --train ${train_file} \
        --test ${test_file} \
        --w2v ${embedding_files} \
        --model ${model_type} \
        --epochs ${epochs} \
        --write_results=../../$result_dir/${result_file_name}_${i}.txt \
        --bidirectional \
        --embedder ${embedder} \
        --batch ${batch_size} \
        --lr ${lr} \
        --hidden_size ${hidden} \
        --embedding_norm ${emb_norm} \
        --drop ${drop_rate} \
        --unfreeze \
        --seed ${seed_used}
    fi
  fi
done
