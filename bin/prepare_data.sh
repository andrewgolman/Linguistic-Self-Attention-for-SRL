#!/usr/bin/env bash
# USAGE: ./prepare_data.sh "directory with preprocessed files"

cp $1/conll2012-test.txt.bio data/
cp $1/conll2012-dev.txt.bio data/
cp $1/conll2012-train.txt.bio data/

python bin/adjust_fields.py --in_file_name data/conll2012-test.txt.bio
python bin/adjust_fields.py --in_file_name data/conll2012-dev.txt.bio
python bin/adjust_fields.py --in_file_name data/conll2012-train.txt.bio

python $1/bin/compute_transition_probs.py --in_file_name data/conll2012-train.txt.lisa > data/transition_probs.tsv

rm data/conll2012-test.txt.bio
rm data/conll2012-dev.txt.bio
rm data/conll2012-train.txt.bio
