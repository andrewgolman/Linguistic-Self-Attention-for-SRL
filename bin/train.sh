#!/usr/bin/env bash

python src/train.py \
--train_files data/conll2012-train.txt.lisa \
--dev_files data/conll2012-dev.txt.lisa,data/conll2012-train-eval.txt.lisa \
--transition_stats data/transition_probs.tsv \
--data_config config/data_configs/conll05.json \
--model_configs config/model_configs/glove_basic.json,config/model_configs/lisa2_embeddings.json \
--task_configs config/task_configs/joint_pos_predicate.json,config/task_configs/parse_heads.json,config/task_configs/parse_labels.json,config/task_configs/srl.json \
--layer_configs config/layer_configs/lisa2_layers.json \
--attention_configs config/attention_configs/parse_attention.json,config/attention_configs/parse_label_attention.json,config/attention_configs/pos_attention.json \
--best_eval_key srl_f1 \
--save_dir model
