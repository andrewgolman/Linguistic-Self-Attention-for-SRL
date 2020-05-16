# Linguistic-Self-Attention-for-SRL

#### This work is based on the work by @strubell and [this repo](https://github.com/strubell/LISA). 

[Original article](https://arxiv.org/abs/1804.08199)


#### Data setup:

- Clone [this repo](https://github.com/iesl/conll2012-preprocess-parsing/tree/master/bin).

- Get CONLL-12 data. (some additional steps? - check out a readme there) Run bin/pp12.sh in the repo directory.

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

#### Running:

[change configs inside the scripts if needed]
```
bin/train.sh
bin/test.sh [checkpoint (todo)]
```

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


#### Differences from the original article
- used transformer implementation by OpenNMT. This implementation has
2 dense layers instead of 3, so the size of hidden layer has been
increased. 
- attention is normalized when being copied from another layer
- on inference SRL loss is calculated over correctly predicted predicates.
This might be used on the late stages of training

#### Current todos:
- add masks for training on Russian Framebank
- generate features with large models and save them prior to training phase
- evaluate within the graph

#### Notes:
- long sentences are also deleted from training dataset to save some gpu memory...