# Calculate BLEU Scores
Here, we provide guides for calculating BLEU scores through `multi_bleu.perl` the trained Transformer translator model.


### 1. Calculate BLEU Scores
#### 1.1 Arguments
There are several arguments for running `src/run/multi_bleu_perl.py`:
* [`-r`, `--resume_model_dir`]: Directory to the model to calculate BLEU scores. Provide the path up to `${project}/${name}`, and it will automatically select the model from `${project}/${name}/weights/` to evaluate.
* [`-l`, `--load_model_type`]: Choose one of [`metric`, `loss`, `last`].
    * `metric` (default): Resume the model with the best validation set's metric such as BLEU, NIST, etc.
    * `loss`: Resume the model with the minimum validation loss.
    * `last`: Resume the model saved at the last epoch.
* [`-d`, `--dataset_type`]: (default: `test`) Choose one of [`train`, `validation`, `test`].


#### 1.2 Command
`src/run/multi_bleu_perl.py` file is used to evaluate the model with the following command:
```bash
python3 src/run/multi_bleu_perl.py --resume_model_dir ${project}/${name}
```