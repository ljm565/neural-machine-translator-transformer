import torch
import re
import pickle
import random
import matplotlib.pyplot as plt
import unicodedata
import random
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.nist_score import corpus_nist



def save_checkpoint(file, model, optimizer):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, file)
    print('model pt file is being saved\n')


def preprocessing(s):
    s = unicodeToAscii(s)
    for punc in '!?.,"':
        s = s.replace(punc, ' '+punc)
    s = re.sub('[#$%&()*+\-/:;<=>@\[\]^_`{|}~]', '', s).lower()
    s = ' '.join(s.split())
    return s


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def load_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def print_samples(src, trg, output, tokenizers, show_n=3, idx=None):
    src, trg, output = src.detach().cpu(), trg.detach().cpu(), output.detach().cpu()
    if idx == None:
        idx = random.sample(list(range(trg.size(0))), show_n)
        print('-'*50)
        for i in idx:
            s, t, o = src[i, :].tolist(), trg[i, 1:].tolist(), torch.argmax(output[i, :-1], dim=1).tolist()
            s, t, o = tokenizers[0].decode(s), tokenizers[1].decode(t), tokenizers[1].decode(o)
            print('src : {}'.format(' '.join(s.split()[1:-1])))
            print('gt  : {}'.format(' '.join(t.split()[:-1])))
            print('pred: {}\n'.format(' '.join(o.split()[:-1])))
        print('-'*50 + '\n')
    else:
        print('-'*50)
        for i in idx:
            s, t, o = src[i, :].tolist(), trg[i, 1:].tolist(), output[i, :-1].tolist()
            s, t, o = tokenizers[0].decode(s), tokenizers[1].decode(t), tokenizers[1].decode(o)
            print('src : {}'.format(' '.join(s.split()[1:])))
            print('gt  : {}'.format(' '.join(t.split())))
            print('pred: {}\n'.format(' '.join(o.split())))
        print('-'*50 + '\n')



def bleu_score(ref, pred, weights):
    return corpus_bleu(ref, pred, weights)


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
    ref = [[tokenizer.decode(ref[i].tolist()).split()] for i in range(ref.size(0))]
    pred = [tokenizer.decode(torch.argmax(pred[i], dim=1).tolist()).split() for i in range(pred.size(0))]
    return ref, pred
    

def visualize_attn(score, src, trg, pred, tokenizers, result_num, save_path):
    src_tokenizer, trg_tokenizer = tokenizers[0], tokenizers[1]
    ids = random.sample(list(range(score.size(0))), result_num)

    for num, i in enumerate(ids):
        src_tok = src_tokenizer.tokenize(src_tokenizer.decode(src[i].tolist()))
        pred_tok = trg_tokenizer.tokenize(trg_tokenizer.decode(pred[i].tolist()))
        
        src_st, src_tr = 1, len(src_tok) - 1
        pred_st, pred_tr = 0, len(pred_tok) - 1

        score_i = score[i, src_st:src_tr, pred_st:pred_tr]
        src_tok = src_tok[src_st:src_tr]
        pred_tok = pred_tok[pred_st:pred_tr]

        plt.figure(figsize=(8, 8))
        plt.title('Neural Machine Translator Attention', fontsize=20)
        plt.imshow(score_i, cmap='gray')
        plt.yticks(list(range(len(src_tok))), src_tok)
        plt.xticks(list(range(len(pred_tok))), pred_tok, rotation=90)
        plt.colorbar()
        plt.savefig(save_path + '_attention' + str(num)+'.jpg')

    print_samples(src, trg, pred, tokenizers, result_num, ids)


def make_inference_data(query, tokenizer, max_len):
    query = [tokenizer.sos_token_id] + tokenizer.encode(preprocessing(query)) + [tokenizer.eos_token_id]
    query = query + [tokenizer.pad_token_id] * (max_len - len(query))
    return torch.LongTensor(query).unsqueeze(0)