### Task config and attention config

Task config is formed by merging all task configs, specified in the global config.
Same is applied to attention config. 

```
{
    'task1_name': TASK_DATA,
    'task2_name': TASK_DATA,
    ...
}
```

Task data:
```
{
    "penalty": FLOAT,  # contribution of this task to the total loss
    "non_fixed_last_dim": BOOL,  # in case output of the task has shape SEQ_LEN * SEQ_LEN
    "output_fn/attention_fn": FUNCTION_DATA,  # which output layer or attention function to use
    "eval_fns": {  # which metrics to use
        "name1": FUNCTION_DATA,
        "name2": FUNCTION_DATA,
        ...
    }
}

```

Function data. Describes params that will be passed into the function.
```
{
    "name": STRING, # names are specified in dispatcher dics in output_fns.py, metrics.py, attention_fns.py
    "params": {
        "param_name1": PARAM_DATA,
        "param_name2": PARAM_DATA,
        ...
    }
}    
```

Param data:

Label of one of the tasks:
```
{"label": "task_name"}
```

OR predictions of one of the previous output layers:
```
{
    "predictions": {
        "layer": "task_name",
        "output": "key_in_an_output_dict_for_this_task"
    }
}
```

OR embeddings
```
{"embeddings": "embeddings_name"}
```

OR vocab mappings.

### Data config

Each row of dataset consists of several fields. Data config tells the model, how to 
use every field.

Param {```conll_idx```} specifies the column number for the field.

Fields with {```feature: true```} will be used as features. 

Fields with {```label: true```} will be used as labels for tasks with this name.

Fields with {```multifeature: true```} will be replaced with multiple fields for the corresponding features.

Name `word_type` should be used for the field with the string-word.
 
### Layer config

For every special attention, specify the layer where it will be applied.

If you use several special attentions of the same type,
you may name them as "NAME_#2", "NAME_#3", etc.

For every output layer, specify, after which transformer layer it will be placed.

Numeration of transformer layers starts with 0.

### Model config

```
"first_layer": "embeddings" OR name of the model from preprocessor_maps.py
"other hyperparams names": their values,
"layers": {params of the transformer layers},
"embeddings": {
    "embeddings_name1": {
      "embedding_dim": DIM1
    },
    "embeddings_name2": {
      "embedding_dim": DIM2
    },
    ...
},
"inputs": [full list of params that will be passed into the model]
```
