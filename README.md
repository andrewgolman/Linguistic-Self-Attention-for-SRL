# Linguistic-Self-Attention-for-SRL


#### Data setup:

- Clone [this repo](https://github.com/iesl/conll2012-preprocess-parsing/tree/master/bin).

- Run bin/pp12.sh in the repo.

- Run bin/prepare_data.sh [conll12 preprocessed files directory] 

- Download GloVe embeddings
```
wget -P embeddings http://nlp.stanford.edu/data/glove.6B.zip
unzip -j embeddings/glove.6B.zip glove.6B.100d.txt -d embeddings
```

#### Requirements:
- python >= 3.6.10


#### Environment setup:

```
pip install -r requirements.txt
```

#### Running (for now):

```
python src/train.py \
--train_files data/conll2012-train.txt.lisa \
--dev_files data/conll2012-test.txt.lisa \
--transition_stats data/transition_probs.tsv \
--data_config config/data_configs/conll05.json \
--model_configs config/model_configs/glove_reduced.json \
--task_configs config/task_configs/joint_pos_predicate.json,config/task_configs/parse_heads.json,config/task_configs/parse_labels.json,config/task_configs/srl.json \
--layer_configs config/layer_configs/lisa_layers.json \
--attention_configs config/attention_configs/parse_attention.json \
--best_eval_key srl_f1 \
--save_dir model 
```
