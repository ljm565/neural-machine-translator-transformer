import torch
import os
import pickle
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.nist_score import corpus_nist


"""
common utils
"""
def load_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def txt_read(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines


def save_data(base_path, data_type):
    if data_type == 'iwslt14-ende':
        if not (os.path.isfile(base_path+'data/iwslt14-en-de/processed/en-de.train') and os.path.isfile(base_path+'data/iwslt14-en-de/processed/en-de.val') and os.path.isfile(base_path+'data/iwslt14-en-de/processed/en-de.test')):
            print('Processing the IWSLT14 raw en-de data')
            for split in ['train', 'val', 'test']:
                en_raw_path, de_raw_path = base_path+'data/iwslt14-en-de/raw/'+split+'.en', base_path+'data/iwslt14-en-de/raw/'+split+'.de'
                en, de = txt_read(en_raw_path), txt_read(de_raw_path)
                dataset = [(e, d) for e, d in zip(en, de)]
                with open(base_path+'data/iwslt14-en-de/processed/en-de.'+split, 'wb') as f:
                    pickle.dump(dataset, f)

    elif data_type == 'wmt14-ende':
        if not (os.path.isfile(base_path+'data/wmt14-en-de/processed/en-de.train') and os.path.isfile(base_path+'data/wmt14-en-de/processed/en-de.val') and os.path.isfile(base_path+'data/wmt14-en-de/processed/en-de.test')):
            print('Processing the WMT14 raw en-de data')
            for split in ['train', 'val', 'test']:
                en_raw_path, de_raw_path = base_path+'data/wmt14-en-de/raw/'+split+'.en', base_path+'data/wmt14-en-de/raw/'+split+'.de'
                en, de = txt_read(en_raw_path), txt_read(de_raw_path)
                dataset = [(e, d) for e, d in zip(en, de)]
                with open(base_path+'data/wmt14-en-de/processed/en-de.'+split, 'wb') as f:
                    pickle.dump(dataset, f)
                

def make_dataset_path(base_path, data_type):
    dataset_path = {}
    if data_type == 'iwslt14-ende':
        for split in ['train', 'val', 'test']:
            dataset_path[split] = base_path+'data/iwslt14-en-de/processed/en-de.'+split
    elif data_type == 'wmt14-ende':
        for split in ['train', 'val', 'test']:
            dataset_path[split] = base_path+'data/wmt14-en-de/processed/en-de.'+split
    return dataset_path


def save_checkpoint(file, model, optimizer):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, file)
    print('model pt file is being saved\n')
    

def bleu_score(ref, pred, weights):
    smoothing = SmoothingFunction().method3
    return corpus_bleu(ref, pred, weights, smoothing)


def nist_score(ref, pred, n):
    return corpus_nist(ref, pred, n)


def cal_scores(ref, pred, type, n_gram):
    assert type in ['bleu', 'nist']
    if type == 'bleu':
        wts = tuple([1/n_gram]*n_gram)
        return bleu_score(ref, pred, wts)
    return nist_score(ref, pred, n_gram)


def tensor2list(ref, pred, tokenizer):
    ref, pred = torch.cat(ref, dim=0)[:, 1:], torch.cat(pred, dim=0)[:, :-1]
    # ref = [[tokenizer.decode(ref[i].tolist()).split()] for i in range(ref.size(0))]
    # pred = [tokenizer.decode(pred[i].tolist()).split() for i in range(pred.size(0))]
    ref = [[tokenizer.tokenize(tokenizer.decode(ref[i].tolist()))] for i in range(ref.size(0))]
    pred = [tokenizer.tokenize(tokenizer.decode(pred[i].tolist())) for i in range(pred.size(0))]
    return ref, pred


def print_samples(ref, pred, ids):
    print('-'*50)
    for i in ids:
        r, p = ' '.join(ref[i][0]), ' '.join(pred[i])
        print('gt  : {}'.format(r))
        print('pred: {}\n'.format(p))
    print('-'*50 + '\n')