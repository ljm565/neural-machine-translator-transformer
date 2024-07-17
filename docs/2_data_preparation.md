# Data Preparation
Here, we will proceed with a Transformer translator model training tutorial using [WMT'14](https://nlp.stanford.edu/projects/nmt/) and [IWSLT'14](https://github.com/facebookresearch/fairseq) datasets.
Please refer to the following instructions to utilize custom datasets.


### 1. IWSLT'14
If you want to train on the IWSLT'14 dataset, simply set the `training_data` value in the `config/config.yaml` file to `iwslt14` as follows.
```yaml
training_data: iwslt14   # You can choose amoge [iwslt14, wmt14, custom].
iwslt14:
    ende: True           # If True, en-de translation will be used. If False, de-en translation will be used.
    path: data/
wmt14:
    ende: True           # If True, en-de translation will be used. If False, de-en translation will be used.
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
<br>

### 2. WMT'14
If you want to train on the WMT'14 dataset, simply set the `training_data` value in the `config/config.yaml` file to `iwslt14` as follows.
```yaml
training_data: wmt14     # You can choose amoge [iwslt14, wmt14, custom].
iwslt14:
    ende: True           # If True, en-de translation will be used. If False, de-en translation will be used.
    path: data/
wmt14:
    ende: True           # If True, en-de translation will be used. If False, de-en translation will be used.
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
<br>

### 3. Custom Data
If you want to train on the custom dataset, simply set the `training_data` value in the `config/config.yaml` file to `custom` as follows.
You may require to implement your custom dataloader codes in `src/utils/data_utils.py`.
You have to set your custom training/validation/test datasets.
```yaml
training_data: custom    # You can choose amoge [iwslt14, wmt14, custom].
iwslt14:
    ende: True           # If True, en-de translation will be used. If False, de-en translation will be used.
    path: data/
wmt14:
    ende: True           # If True, en-de translation will be used. If False, de-en translation will be used.
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
<br>