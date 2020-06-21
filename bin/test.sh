#!/usr/bin/env bash

python src/test.py \
--data config/conll2012_path.json \
--config config/transformer_config.json \
--save_dir albert_notokens \
--eval_every 10 \
--save_every 40 \
--checkpoint glove_params/checkpoints/epoch_160 \
