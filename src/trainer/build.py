
import os
import subprocess

import torch
from torch.utils.data import DataLoader, distributed

from models import Transformer
from utils import LOGGER, RANK, colorstr
# from utils.data_utils import DLoader, CustomDLoader, seed_worker, get_tatoeba

PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders



def get_tokenizers(config):
    if config.training_data.lower() == 'iwslt14':
        vocab_size = str(config.vocab_size)
        data_path = os.path.join(config.iwslt14.path, 'iwslt14-en-de') 
        
        if not os.path.isdir(os.path.join(data_path, f'tokenizer/vocab_{vocab_size}')):
            main_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
            vocab_sh = os.path.join(main_dir, 'src/tools/tokenizers/build/make_vocab.sh')
            vocab_py = os.path.join(main_dir, 'src/tools/tokenizers/build/vocab_trainer.py')
            
            data_path = os.path.join(main_dir, config.iwslt14.path, 'iwslt14-en-de/raw')
            tokenizer_path = os.path.join(main_dir, config.iwslt14.path, 'iwslt14-en-de/tokenizer')
            
            runs = subprocess.run([vocab_sh, data_path, tokenizer_path, vocab_size, vocab_py], capture_output=True, text=True)
            
            LOGGER.info((colorstr("Making vocab file for custom tokenizer..")))
            LOGGER.info((colorstr(runs.stdout)))
            LOGGER.error(colorstr('red', runs.stderr))
            
    else:
        # NOTE: You need train data to build custom word tokenizer
        trainset_path = config.CUSTOM.train_data_path
        LOGGER.info(colorstr('red', 'You need train data to build custom word tokenizer..'))
        raise NotImplementedError
    return tokenizers


def get_model(config, tokenizers, device):
    src_tokenizer, trg_tokenizer = tokenizers
    encoder = Encoder(config, src_tokenizer, device).to(device)
    decoder = Decoder(config, trg_tokenizer).to(device)
    return encoder, decoder


def build_dataset(config, tokenizers, modes):
    if config.tatoeba_train:
        trainset, testset = get_tatoeba(config)
        tmp_dsets = {'train': trainset, 'validation': testset}
        dataset_dict = {mode: DLoader(config, tmp_dsets[mode], tokenizers) for mode in modes}
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


def get_data_loader(config, tokenizers, modes, is_ddp=False):
    datasets = build_dataset(config, tokenizers, modes)
    dataloaders = {m: build_dataloader(datasets[m], 
                                       config.batch_size, 
                                       config.workers, 
                                       shuffle=(m == 'train'), 
                                       is_ddp=is_ddp) for m in modes}

    return dataloaders