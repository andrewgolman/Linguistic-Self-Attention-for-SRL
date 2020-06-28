# Linguistic-Self-Attention-for-SRL

#### This work is based on the work by @strubell and [this repo](https://github.com/strubell/LISA). 

[Original paper](https://arxiv.org/abs/1804.08199)

#### Requirements:

- python >= 3.6.10


#### Environment setup:

```
pip install -r requirements.txt
```

#### CoNLL-2012 data setup:

- Get CONLL-2012 dataset.

- Clone [this repo](https://github.com/iesl/conll2012-preprocess-parsing/tree/master/bin).

- Run bin/pp12.sh in the IESL repo directory.

- Run `bin/prepare_data.sh [conll12 preprocessed files directory]` 

- If you want to run the model on English GloVe embeddings, you may download them:
```
wget -P embeddings http://nlp.stanford.edu/data/glove.6B.zip
unzip -j embeddings/glove.6B.zip glove.6B.100d.txt -d embeddings
```

#### Russian Framebank data setup:

- Follow preprocessing from [isanlp_srl_framebank](https://github.com/IINemo/isanlp_srl_framebank/tree/master/src/training)

- Put generated files [features.pckl](https://github.com/IINemo/isanlp_srl_framebank/blob/d6e2af3c5015ac31070bce076835d4db1f7593ae/src/training/run_extract_features.py)
and [ling_data.pckl](https://github.com/IINemo/isanlp_srl_framebank/blob/ff0c7b5cf3303f290143d87d9c5f1af46abcf3e5/src/training/run_ling_parse.py)
into data directory.

- Run bin/framebank_preprocess.py.

- If you want to run the model on FastText embeddings, you may download them at 
https://fasttext.cc/docs/en/crawl-vectors.html

#### Running:

```
python src/train.py \
--data [your data path config]
--config [your global config]
--save_dir [directory to save models, vocabs and metrics]
--eval_every [int]
--save_every [int]
```

Data path config is a following .json file:
```
{
    "train": ["file1", "file2", ...],
    "dev": ["file1", "file2", ...],
    "test": ["file1", "file2", ...],  # not required
    "transition_stats": "file"  # required if crf/viterbi decoding is used
}
```

Global config:
```
{
  "data_configs": ["config_file1", "config_file2", ...],
  "model_configs": ["config_file1", "config_file2", ...],
  "task_configs": ["config_file1", "config_file2", ...],
  "layer_configs": ["config_file1", "config_file2", ...],
  "attention_configs": ["config_file1", "config_file2", ...],
}
```

Find out more about config files in config/README.md.

#### Code reading starter pack
1. model.py has main logic, as usual
2. Due to keras limitations for multi-output models, all metrics and
losses are computed inside the model.
On inference, loss and metrics are returned empty (didn’t test it yet).
3. transormer_layer.py adjusts Multi-Head Attention Encoder
implementation by OpenNMT
4. output_fns.py contains all output layers. All current model
outputs are passed into the layer, then BaseFunction proxies
parameters into layer-specific format.
5. attention_fns.py implements outputs that will be used in
further transformer layers

#### Notes

- We use OpenNMT implementation of the multi-head self-attention encoder
- While training, long sentences (more than 100 tokens) are ignored to save GPU memory
- Training an English model on Tesla-1080 takes about up to 6 hours.
- Training a Russian model in Colab takes up to 2 hours.
