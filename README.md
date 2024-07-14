# Neural Machine Translator Transformer (WMT'14, IWSLT'14 En-De)
한국어 버전의 설명은 [여기](./docs/README_ko.md)를 참고하시기 바랍니다.

## Introduction
Using the English-German sentence pair data from WMT'14 and IWSLT'14, create a machine translation model based on the [Transformer](https://arxiv.org/pdf/1706.03762.pdf) architecture.
For a detailed explanation of the Transformer-based machine translation model, refer to [Transformer를 이용한 WMT'14, IWSLT'14 (En-De) 기계 번역](https://ljm565.github.io/contents/transformer2.html).
Additionally, the model should allow the choice between using positional encoding, as used in the vanilla transformer, and positional embedding.
Finally, you can calculate the benchmark score for the trained model using [multi_bleu.perl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl), which is commonly used for this purpose.
<br><br><br>

## Supported Models
### Transformer
* Transformer
<br><br><br>

## Supported Tokenizer
### Wordpiece Tokenizer
Use a subword tokenizer that performs BPE (Byte Pair Encoding) based on likelihood.
Additionally, construct a shared vocabulary by integrating both the English and German data.

* IWSLT'14: When using IWSLT'14 data, tokenizer will be trained automatically through [make_vocab.sh](src/tools/tokenizers/build/make_vocab.sh) file. You can set vocabulary size in the `config/config.yaml` (Default: 10,000).
* WMT'14: Pre-trained vocabulary will be loaded automatically from Hugging Face. 
<br><br><br>

## Base Dataset
The `data_sample` mentioned here is only a subset of the entire dataset.
You can download the full dataset from the link below.
If you want to check if the code runs properly, you need to execute the following command first to rename the data folder.
```bash
mv data_sample data
```
* WMT'14 (En-De): [Stanford WMT'14](https://nlp.stanford.edu/projects/nmt/)
* IWSLT'14 (En-De): [fairseq](https://github.com/facebookresearch/fairseq)
<br><br><br>


## Supported Devices
* CPU, GPU, multi-GPU (DDP), MPS (for Mac and torch>=1.12.0)
<br><br><br>

## Quick Start
```bash
python3 src/run/train.py --config config/config.yaml --mode train
```
<br><br>


## Project Tree
This repository is structured as follows.
```
├── configs                           <- Folder for storing config files
│   └── *.yaml
│
├── etc                               
│   └── multi_bleu.perl               <- Perl script for calculating BLEU
│
└── src      
    ├── models
    |   ├── embeddings.py             <- Transformer embedding layers
    |   ├── modules.py                <- Attention, FFN, etc. layers
    |   └── transformer.py            <- Whole encoder, decoder and transformer model
    |
    ├── run                   
    |   ├── multi_bleu_perl.py        <- Codes for calculating BLEU scores using multi_bleu.perl script
    |   ├── train.py                  <- Training execution file
    |   └── validation.py             <- Trained model evaulation execution file
    |
    ├── tools                   
    |   ├── tokenizers
    |   |   ├── build                 <- Using for constructing custom IWSLT's tokenizer
    |   |   |   ├── make_vocab.sh
    |   |   |   └── vocab_trainer.py
    |   |   └── tokenizer.py          <- Tokenizer classes of IWSLT and WMT
    |   |
    |   ├── early_stopper.py          <- Early stopper class file
    |   ├── evaluator.py              <- Metric evaluator class file
    |   ├── model_manager.py          
    |   └── training_logger.py        <- Training logger class file
    |
    ├── trainer                 
    |   ├── build.py                  <- Codes for initializing dataset, dataloader, etc.
    |   └── trainer.py                <- Class for training, evaluating, and calculating accuracy
    |
    └── uitls                   
        ├── __init__.py               <- File for initializing the logger, versioning, etc.
        ├── data_utils.py             <- File defining the dataset's dataloader
        ├── filesys_utils.py       
        ├── func_utils.py       
        └── training_utils.py     
```
<br><br>


## Tutorials & Documentations
Please follow the steps below to train a Transformer translator model.
1. [Getting Started](./docs/1_getting_started.md)
2. [Data Preparation](./docs/2_data_preparation.md)
3. [Training](./docs/3_trainig.md)
4. ETC
   * [Evaluation](./docs/4_model_evaluation.md)
   * [Calculate BLEU via `perl` Script](./docs/5_cal_multi_bleu_perl.md)

<br><br><br>



## Training Results
### Each Result of Transformer Neural Machine Translator
The scores below are the results obtained from inference on the validation set.
The results shown below are the test set BLEU-4 scores achieved when the highest BLEU-4 on the validation set was obtained.
Therefore, there may be a difference between the best score seen during training in the graph and the scores listed below.
The BLEU-4 scores were calculated using both NLTK and `multi_bleu.perl`.

* WMT'14 (En-De) Validation Set BLEU History<br>
    <img src="docs/figs/wmt_bleu.png" width="80%"><br>
    * Test set BLEU-4: 0.2803 (NLTK)
    * Test set BLEU-4: 0.2803 (multi_bleu.perl)<br><br>

* WMT'14 (En-De) Validation Set NIST History<br>
    <img src="docs/figs/wmt_nist.png" width="80%"><br><br>

* IWSLT'14 (En-De) Validation Set BLEU History<br>
    <img src="docs/figs/iwslt_bleu.png" width="80%"><br>
    * Test set BLEU-4: 0.2579 (NLTK)
    * Test set BLEU-4: 0.2580 (multi_bleu.perl)<br><br>

* IWSLT'14 (En-De) Validation Set NIST History<br>
    <img src="docs/figs/iwslt_nist.png" width="80%"><br><br>


### Translated Samples and Attentions of Each Transformer Model
* WMT'14 Trained Model
    ```
    # Sample 1
    gt  : „ erwachsene sollten in der lage sein , eigene entscheidungen uber ihr rechtliche ##s geschlecht zu treffen “ , erklarte sie .
    pred: " erwachsene sollten in der lage sein , ihre eigenen entscheidungen uber das legal ##e geschlecht zu treffen " , sagte sie .


    # Sample 2
    gt  : insgesamt seien vier verkehrs ##schauen durchgefuhrt worden , auch ein kreis ##verkehr wurde ange ##dacht , allerdings wegen der enge in dem kreuzung ##s ##bereich sulz ##bach ##weg / kirchimportant .
    pred: insgesamt wurden vier sicherheits ##kontrollen im straßenverkehr durchgefuhrt , und auch ein kreis ##verkehr wurde berucksichtigt , jedoch wurde dieser gedanke aufgrund der engen linien abgelehnt .


    # Sample 3
    gt  : austral ##ische flug ##pass ##agi ##e ##re mussen auch weiterhin tablets und smartphones bei start und landung abschalten , trotz bemuhungen in den usa , die regelungen fur derartige gerate
    pred: flug ##gaste austral ##ischer fluggesellschaft mussen trotz der bemuhungen in den usa , die bestimmungen uber die flug ##pass ##agi ##e ##re zu locker ##n , ihre flug ##table ##tten

    ```


* IWSLT'14 Trained Model
    ```
    # Sample 1
    gt  : ein sehr konk ##rete ##r wunsch , dass wir diese technologie erfinden .
    pred: es ist ein sehr konk ##ret wunsch , dass wir diese technologie erfinden .


    # Sample 2
    gt  : wir durch ##laufen initi ##ations ##ri ##t ##ual ##e .
    pred: wir durch ##laufen den initi ##ations ##ri ##k .


    # Sample 3
    gt  : vac ##la ##v have ##l , der große ts ##che ##ch ##ische politiker , hat einmal gesagt :
    pred: v ##la ##v haben ##l den großen c ##ze ##ch - anfü ##hrer , darüber gesprochen .
    ```



<br><br><br>
