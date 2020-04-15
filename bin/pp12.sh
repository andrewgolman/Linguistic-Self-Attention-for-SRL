#!/usr/bin/env bash

# This can be run in the CONLL2012-PREPROCESS repo.

# ./bin/preprocess_conll2012.sh ../conll-2012/v4/data/test/data/english
# ./bin/preprocess_conll2012.sh ../conll-2012/v4/data/dev/data/english
# ./bin/preprocess_conll2012.sh ../conll-2012/v4/data/train/data/english

# for f in `find ../conll-2012/v4/data/test/data/english -type f -name "*\.parse\.dep\.combined"`; do cat $f >> conll2012-test.txt; done
# for f in `find ../conll-2012/v4/data/dev/data/english -type f -name "*\.parse\.dep\.combined"`; do cat $f >> conll2012-dev.txt; done
# for f in `find ../conll-2012/v4/data/train/data/english -type f -name "*\.parse\.dep\.combined"`; do cat $f >> conll2012-train.txt; done

# ./bin/convert-bio.sh conll2012-test.txt
# ./bin/convert-bio.sh conll2012-dev.txt
# ./bin/convert-bio.sh conll2012-train.txt

# python ./bin/compute_transition_probs.py --in_file_name conll2012-train.txt.bio > transition_probs.tsv
