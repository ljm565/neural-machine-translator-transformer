# Training Transformer Translator
Here, we provide guides for training a Transformer translator model.

### 1. Configuration Preparation
To train a Transformer translator model, you need to create a configuration.
Detailed descriptions and examples of the configuration options are as follows.

```yaml
# base
seed: 0
deterministic: True

# environment config
device: cpu     # examples: [0], [0,1], [1,2,3], cpu, mps... 

# project config
project: outputs/transformer
name: iwslt14-ende

# model config
vocab_size: 10000
max_len: 32
hidden_dim: 1024
ffn_dim: 4096
enc_num_layers: 6
dec_num_layers: 6
num_heads: 16
bias: 0
dropout: 0.1
layernorm_eps: 1e-6
pos_encoding: False     # If False, positional positional embedding will be used. If True, positional encoding will be used.

# data config
workers: 0               # Don't worry to set worker. The number of workers will be set automatically according to the batch size.
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

# train config
batch_size: 128
steps: 2000000
warmup_steps: 1000
lr0: 0.0001
lrf: 0.001                            # last_lr = lr0 * lrf
scheduler_type: 'cosine'              # ['linear', 'cosine']
patience: 5                           # Early stopping epochs.
prediction_print_n: 10                # Number of examples to show during inference.

# logging config
common: ['train_loss', 'validation_loss', 'lr']
metrics: ['ppl', 'bleu2', 'bleu4', 'nist2', 'nist4']   # You can add more metrics after implements metric validation codes
```


### 2. Training
#### 2.1 Arguments
There are several arguments for running `src/run/train.py`:
* [`-c`, `--config`]: Path to the config file for training.
* [`-m`, `--mode`]: Choose one of [`train`, `resume`].
* [`-r`, `--resume_model_dir`]: Path to the model directory when the mode is resume. Provide the path up to `${project}/${name}`, and it will automatically select the model from `${project}/${name}/weights/` to resume.
* [`-l`, `--load_model_type`]: Choose one of [`metric`, `loss`, `last`].
    * `metric` (default): Resume the model with the best validation set's metrics.
    * `loss`: Resume the model with the minimum validation loss.
    * `last`: Resume the model saved at the last epoch.
* [`-p`, `--port`]: (default: `10001`) NCCL port for DDP training.


#### 2.2 Command
`src/run/train.py` file is used to train the model with the following command:
```bash
# training from scratch
python3 src/run/train.py --config configs/config.yaml --mode train

# training from resumed model
python3 src/run/train.py --config config/config.yaml --mode resume --resume_model_dir ${project}/${name}
```

When training started, the learning rate curve will be saved in `${project}/${name}/vis_outputs/lr_schedule.png` automatically based on the values set in `config/config.yaml`.
When the model training is complete, the checkpoint is saved in `${project}/${name}/weights` and the training config is saved at `${project}/${name}/args.yaml`.