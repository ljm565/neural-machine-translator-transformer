# Data Preparation
여기서는 기본적으로 [WMT'14](https://nlp.stanford.edu/projects/nmt/)와 [IWSLT'14](https://github.com/facebookresearch/fairseq) 데이터셋을 활용하여 Transformer 기계 번역 모델 학습 튜토리얼을 진행합니다.
Custom 데이터를 이용하기 위해서는 아래 설명을 참고하시기 바랍니다.

### 1. IWSLT'14
IWSLT'14 데이터를 학습하고싶다면 아래처럼 `config/config.yaml`의 `training_data`을 `iwslt14` 설정하면 됩니다.
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

### 2. WMT'14
WMT'14 데이터를 학습하고싶다면 아래처럼 `config/config.yaml`의 `training_data`을 `wmt14` 설정하면 됩니다.
```yaml
training_data: wmt14   # You can choose amoge [iwslt14, wmt14, custom].
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
만약 custom 데이터를 학습하고 싶다면 아래처럼 `config/config.yaml`의 `training_data`을 `custom`로 설정하면 됩니다.
다만 `src/utils/data_utils.py`에 custom dataloader를 구현해야할 수 있습니다.
Custom data 사용을 위해 train/validation/test 데이터셋 경로를 입력해주어야 합니다.
```yaml
training_data: custom   # You can choose amoge [iwslt14, wmt14, custom].
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