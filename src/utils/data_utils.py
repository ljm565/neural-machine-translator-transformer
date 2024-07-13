import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset

from utils import LOGGER, colorstr
from utils.filesys_utils import txt_read, write_dataset



def preprocess_data(config):
    if config.training_data.lower() == 'iwslt14':
        splits = ['train', 'val', 'test']
        processed_data_dir = os.path.join(config.iwslt14.path, 'iwslt14-en-de/processed')
        os.makedirs(processed_data_dir, exist_ok=True)
        
        # check whether training, validation and test data files are exsisted
        if not all([os.path.isfile(os.path.join(processed_data_dir, f'en-de.{split}')) for split in splits]):
            LOGGER.info(colorstr('Processing the IWSLT14 raw en-de data'))
            for split in splits:
                en_raw_path = os.path.join(config.iwslt14.path, f'iwslt14-en-de/raw/{split}.en')
                de_raw_path = os.path.join(config.iwslt14.path, f'iwslt14-en-de/raw/{split}.de')
                
                en, de = txt_read(en_raw_path), txt_read(de_raw_path)
                dataset = [(e, d) for e, d in zip(en, de)]
                dataset_path = os.path.join(processed_data_dir, f'en-de.{split}')
                write_dataset(dataset_path, dataset)
        
        dataset_paths = {split: os.path.join(processed_data_dir, f'en-de.{split}') for split in splits}
        dataset_paths['validation'] = dataset_paths.pop('val')  # to change 'val' to 'validation'

    elif config.training_data.lower() == 'wmt14':
        splits = ['train', 'val', 'test']
        processed_data_dir = os.path.join(config.wmt14.path, 'wmt14-en-de/processed')
        os.makedirs(processed_data_dir, exist_ok=True)
        
        # check whether training, validation and test data files are exsisted
        if not all([os.path.isfile(os.path.join(processed_data_dir, f'en-de.{split}')) for split in splits]):
            LOGGER.info(colorstr('Processing the WMT4 raw en-de data'))
            for split in splits:
                en_raw_path = os.path.join(config.iwslt14.path, f'wmt14-en-de/raw/{split}.en')
                de_raw_path = os.path.join(config.iwslt14.path, f'wmt14-en-de/raw/{split}.de')
                
                en, de = txt_read(en_raw_path), txt_read(de_raw_path)
                dataset = [(e, d) for e, d in zip(en, de)]
                dataset_path = os.path.join(processed_data_dir, f'en-de.{split}') 
                write_dataset(dataset_path, dataset)

        dataset_paths = {split: os.path.join(processed_data_dir, f'en-de.{split}') for split in splits}
        dataset_paths['validation'] = dataset_paths.pop('val')  # to change 'val' to 'validation'
    
    else:
        LOGGER.warning(colorstr('yellow', 'You must implement data preprocessing codes..'))
        raise NotImplementedError
        
    return dataset_paths
        


def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



class DLoader_iwslt(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = config.max_len
        self.ende = config.iwslt14.ende
        self.length = len(self.data)

    
    def add_special_token(self, s, tokenizer):
        s = [tokenizer.bos_token_id] + tokenizer.encode(s)[:self.max_len-2] + [tokenizer.eos_token_id]
        s = s + [tokenizer.pad_token_id] * (self.max_len - len(s))
        return s


    def __getitem__(self, idx):
        if self.ende:
            src, trg = self.add_special_token(self.data[idx][0], self.tokenizer), self.add_special_token(self.data[idx][1], self.tokenizer)
        else:
            src, trg = self.add_special_token(self.data[idx][1], self.tokenizer), self.add_special_token(self.data[idx][0], self.tokenizer)
        return torch.tensor(src, dtype=torch.long), torch.tensor(trg, dtype=torch.long)

    
    def __len__(self):
        return self.length



class DLoader_wmt(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = config.max_len
        self.ende = config.wmt14.ende
        self.length = len(self.data)

    
    def add_special_token(self, s, tokenizer):
        s = [tokenizer.bos_token_id] + tokenizer.encode(s)[:self.max_len-2] + [tokenizer.eos_token_id]
        s = s + [tokenizer.pad_token_id] * (self.max_len - len(s))
        return s


    def __getitem__(self, idx):
        if self.ende:
            src, trg = self.add_special_token(self.data[idx][0], self.tokenizer), self.add_special_token(self.data[idx][1], self.tokenizer)
        else:
            src, trg = self.add_special_token(self.data[idx][1], self.tokenizer), self.add_special_token(self.data[idx][0], self.tokenizer)
        return torch.tensor(src, dtype=torch.long), torch.tensor(trg, dytpe=torch.long)

    
    def __len__(self):
        return self.length