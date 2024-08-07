
import os
import subprocess

import torch
from torch.utils.data import DataLoader, distributed
from torch import distributed as dist

from models import Transformer
from tools.tokenizers import IWSLTTokenizer_EnDe, WMTTokenizer_EnDe
from utils import LOGGER, RANK, colorstr
from utils.filesys_utils import read_dataset
from utils.data_utils import DLoader_iwslt, DLoader_wmt, preprocess_data, seed_worker

PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders



def get_tokenizers(config, is_ddp=False):
    if config.training_data.lower() == 'iwslt14':
        vocab_size = str(config.vocab_size)
        data_path = os.path.join(config.iwslt14.path, 'iwslt14-en-de') 
        
        if not os.path.isdir(os.path.join(data_path, f'tokenizer/vocab_{vocab_size}')) and config.is_rank_zero:
            main_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
            vocab_sh = os.path.join(main_dir, 'src/tools/tokenizers/build/make_vocab.sh')
            vocab_py = os.path.join(main_dir, 'src/tools/tokenizers/build/vocab_trainer.py')
            
            raw_data_path = os.path.join(main_dir, config.iwslt14.path, 'iwslt14-en-de/raw')
            tokenizer_path = os.path.join(main_dir, config.iwslt14.path, 'iwslt14-en-de/tokenizer')
            
            LOGGER.info((colorstr("Making vocab file for custom tokenizer..")))
            runs = subprocess.run([vocab_sh, raw_data_path, tokenizer_path, vocab_size, vocab_py], capture_output=True, text=True)
            LOGGER.info((colorstr(runs.stdout)))
            LOGGER.error(colorstr('red', runs.stderr))
        
        if is_ddp:
            dist.barrier()      # wait all gpus to finish vocab file creation

        config.tokenizer_path = os.path.join(data_path, f'tokenizer/vocab_{vocab_size}/vocab.txt')
        tokenizer = IWSLTTokenizer_EnDe(config)
            
    elif config.training_data.lower() == 'wmt14':
        tokenizer = WMTTokenizer_EnDe()
            
    else:
        LOGGER.warning(colorstr('yellow', 'You must implement your custom tokenizer loading codes..'))
        raise NotImplementedError
    
    return tokenizer


def get_model(config, tokenizer, device):
    model = Transformer(config, tokenizer, device).to(device)
    return model


def build_dataset(config, tokenizer, modes):
    if config.training_data.lower() in ['iwslt14', 'wmt14']:
        dataset_paths = preprocess_data(config)
        dloader = DLoader_iwslt if config.training_data.lower() == 'iwslt14' else DLoader_wmt
        tmp_dsets = {s: dloader(read_dataset(p), tokenizer, config) for s, p in dataset_paths.items()}
        dataset_dict = {mode: tmp_dsets[mode] for mode in modes}
    else:
        LOGGER.warning(colorstr('yellow', 'You have to implement data pre-processing code..'))
        # dataset_dict = {mode: CustomDLoader(config.CUSTOM.get(f'{mode}_data_path')) for mode in modes}
        raise NotImplementedError
    return dataset_dict


def build_dataloader(dataset, batch, workers, shuffle=True, is_ddp=False):
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers
    sampler = None if not is_ddp else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return DataLoader(dataset=dataset,
                              batch_size=batch,
                              shuffle=shuffle and sampler is None,
                              num_workers=nw,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY,
                              collate_fn=getattr(dataset, 'collate_fn', None),
                              worker_init_fn=seed_worker,
                              generator=generator)


def get_data_loader(config, tokenizer, modes, is_ddp=False):
    datasets = build_dataset(config, tokenizer, modes)
    dataloaders = {m: build_dataloader(datasets[m], 
                                       config.batch_size, 
                                       config.workers, 
                                       shuffle=(m == 'train'), 
                                       is_ddp=is_ddp) for m in modes}

    return dataloaders