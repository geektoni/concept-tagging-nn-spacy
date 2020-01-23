#!/bin/bash
#
# This software is distributed under MIT license (see LICENSE file).
#
# Authors: Giovanni De Toni
#

# Collect all the results inside the results directory and generate a complete
# file with all the results.
for f in `ls ./results/`;
do
  if [ ! -z "$(ls -A ./results/${f})" ]; then
    cd concept-tagging-with-neural-networks/src/
    python collect_results.py ../../results/${f}
    mean=`cat ../../results/${f}/f1.scores | head -n 1 | cut -d"," -f 1 -- | cut -d":" -f 2 | awk '{$1=$1;print}'`
    std=`cat ../../results/${f}/f1.scores | head -n 1 | cut -d"," -f 2 -- |  cut -d":" -f 2 | awk '{$1=$1;print}'`
    max=`cat ../../results/${f}/f1.scores | head -n 1 | cut -d"," -f 3 -- |  cut -d":" -f 2 | awk '{$1=$1;print}'`
    min=`cat ../../results/${f}/f1.scores | head -n 1 | cut -d"," -f 4 -- |  cut -d":" -f 2 | awk '{$1=$1;print}'`
    echo $f, $mean, $std, $max, $min >> ../../complete_results.txt
    cd ../../
  fi
done