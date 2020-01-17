#!/bin/bash

for f in `ls ./results/`;
do
  if [ ! -z "$(ls -A ./results/${f})" ]; then
    cd concept-tagging-with-neural-networks/src/
    python collect_results.py ../../results/${f}
    result=`cat ../../results/${f}/f1.scores | head -n 1`
    echo $f, $result >> ../../complete_results.txt
    cd ../../
  fi
done