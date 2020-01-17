#!/bin/bash

for f in `ls ./results/`;
do
  python concept-tagging-with-neural-networks/src/collect_results.py ./results/${f}
  result=`cat ./results/${f}/f1.scores | head -n 1`
  echo $f, $result >> complete_results.txt
done