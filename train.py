import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
from tokenizer import Tokenizer
import numpy as np
import time
import sys

from config import Config
from utils_func import *
from utils_data import DLoader
from model import Transformer



class Trainer:
    def __init__(self, config:Config, device:torch.device, mode:str, continuous:int):
        self.config = config
        self.device = device
        self.mode = mode
        self.continuous = continuous
        self.dataloaders = {}

        # if continuous, load previous training info
        if self.continuous:
            with open(self.config.loss_data_path, 'rb') as f:
                self.loss_data = pickle.load(f)

        # path, data params
        self.base_path = self.config.base_path
        self.model_path = self.config.model_path
        self.data_path = self.config.dataset_path
 
        # train params
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.lr = self.config.lr
        self.max_len = self.config.max_len

        # define tokenizer
        self.src_tokenizer = Tokenizer(self.config, self.data_path['train'], src=True)
        self.trg_tokenizer = Tokenizer(self.config, self.data_path['train'], src=False)
        self.tokenizers = [self.src_tokenizer, self.trg_tokenizer]

        # dataloader
        torch.manual_seed(999)  # for reproducibility
        if self.mode == 'train':
            self.dataset = {s: DLoader(load_dataset(p), self.tokenizers, self.config) for s, p in self.data_path.items()}
            self.dataloaders = {
                s: DataLoader(d, self.batch_size, shuffle=True) if s == 'train' else DataLoader(d, self.batch_size, shuffle=False)
                for s, d in self.dataset.items()}
        else:
            self.dataset = {s: DLoader(load_dataset(p), self.tokenizers, self.config) for s, p in self.data_path.items() if s == 'test'}
            self.dataloaders = {s: DataLoader(d, self.batch_size, shuffle=False) for s, d in self.dataset.items() if s == 'test'}

        # model, optimizer, loss
        self.model = Transformer(self.config, self.tokenizers, self.device).to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.trg_tokenizer.pad_token_id)
        if self.mode == 'train':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            if self.continuous:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(self.check_point['model'])
                self.optimizer.load_state_dict(self.check_point['optimizer'])
                del self.check_point
                torch.cuda.empty_cache()
        elif self.mode == 'test' or self.mode == 'inference':
            self.check_point = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(self.check_point['model'])    
            self.model.eval()
            del self.check_point
            torch.cuda.empty_cache()

        
    def train(self):
        early_stop = 0
        best_val_bleu = 0 if not self.continuous else self.loss_data['best_val_bleu']
        train_loss_history = [] if not self.continuous else self.loss_data['train_loss_history']
        val_loss_history = [] if not self.continuous else self.loss_data['val_loss_history']
        val_score_history = {'bleu2': [], 'bleu4': [], 'nist2': [], 'nist4': []} if not self.continuous else self.loss_data['val_score_history']
        best_epoch_info = 0 if not self.continuous else self.loss_data['best_epoch']

        for epoch in range(self.epochs):
            start = time.time()
            print(epoch+1, '/', self.epochs)
            print('-'*10)
            for phase in ['train', 'test']:
                print('Phase: {}'.format(phase))
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                total_loss = 0
                all_val_trg, all_val_output = [], []
                for i, (src, trg) in enumerate(self.dataloaders[phase]):
                    batch = src.size(0)
                    src, trg = src.to(self.device), trg.to(self.device)
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase=='train'):
                        _, output = self.model(src, trg)
                        loss = self.criterion(output[:, :-1, :].reshape(-1, output.size(-1)), trg[:, 1:].reshape(-1))
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                        else:
                            all_val_trg.append(trg.detach().cpu())
                            all_val_output.append(output.detach().cpu())

                    total_loss += loss.item()*batch

                    if i % 100 == 0:
                        print('Epoch {}: {}/{} step loss: {}'.format(epoch+1, i, len(self.dataloaders[phase]), loss.item()))
                epoch_loss = total_loss/len(self.dataloaders[phase].dataset)
                print('{} loss: {:4f}\n'.format(phase, epoch_loss))

                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                if phase == 'test':
                    val_loss_history.append(epoch_loss)

                    # print examples
                    print_samples(src, trg, output, self.tokenizers)

                    # calculate scores
                    all_val_trg, all_val_output = tensor2list(all_val_trg, all_val_output, self.trg_tokenizer)
                    val_score_history['bleu2'].append(cal_scores(all_val_trg, all_val_output, 'bleu', 2))
                    val_score_history['bleu4'].append(cal_scores(all_val_trg, all_val_output, 'bleu', 4))
                    val_score_history['nist2'].append(cal_scores(all_val_trg, all_val_output, 'nist', 2))
                    val_score_history['nist4'].append(cal_scores(all_val_trg, all_val_output, 'nist', 4))
                    print('bleu2: {}, bleu4: {}, nist2: {}, nist4: {}'.format(val_score_history['bleu2'][-1], val_score_history['bleu4'][-1], val_score_history['nist2'][-1], val_score_history['nist4'][-1]))
                    
                    # save best model
                    early_stop += 1
                    if val_score_history['bleu4'][-1] > best_val_bleu:
                        early_stop = 0
                        best_val_bleu = val_score_history['bleu4'][-1]
                        best_epoch = best_epoch_info + epoch + 1
                        save_checkpoint(self.model_path, self.model, self.optimizer)

            print("time: {} s\n".format(time.time() - start))
            print('\n'*2)

            # early stopping
            if early_stop == self.config.early_stop_criterion:
                break

        print('best val bleu: {:4f}, best epoch: {:d}\n'.format(best_val_bleu, best_epoch))
        self.loss_data = {'best_epoch': best_epoch, 'best_val_bleu': best_val_bleu, 'train_loss_history': train_loss_history, 'val_loss_history': val_loss_history, 'val_score_history': val_score_history}
        return self.loss_data
    

    def test(self, result_num):
        if result_num > len(self.dataloaders['test'].dataset):
            print('The number of results that you want to see are larger than total test set')
            sys.exit()
        
        # statistics of the test set
        phase = 'test'
        total_loss = 0
        all_val_src, all_val_trg, all_val_output = [], [], []

        with torch.no_grad():
            self.model.eval()

            for src, trg in self.dataloaders[phase]:
                batch = src.size(0)
                src, trg = src.to(self.device), trg.to(self.device)
                
                _, output = self.model(src, trg)
                loss = self.criterion(output[:, :-1, :].reshape(-1, output.size(-1)), trg[:, 1:].reshape(-1))
                total_loss += loss.item()*batch
                
                all_val_src.append(src.detach().cpu())
                all_val_trg.append(trg.detach().cpu())
                all_val_output.append(output.detach().cpu())

        # calculate loss and ppl
        total_loss = total_loss / len(self.dataloaders[phase].dataset)
        print('loss: {}, ppl: {}'.format(total_loss, np.exp(total_loss)))

        # calculate scores
        all_val_trg_l, all_val_output_l = tensor2list(all_val_trg, all_val_output, self.trg_tokenizer)
        bleu2 = cal_scores(all_val_trg_l, all_val_output_l, 'bleu', 2)
        bleu4 = cal_scores(all_val_trg_l, all_val_output_l, 'bleu', 4)
        nist2 = cal_scores(all_val_trg_l, all_val_output_l, 'nist', 2)
        nist4 = cal_scores(all_val_trg_l, all_val_output_l, 'nist', 4)
        print('bleu2: {}, bleu4: {}, nist2: {}, nist4: {}'.format(bleu2, bleu4, nist2, nist4))

        # print samples
        all_val_src = torch.cat(all_val_src, dim=0)
        all_val_trg = torch.cat(all_val_trg, dim=0)
        all_val_output = torch.argmax(torch.cat(all_val_output, dim=0), dim=2)
        ids = random.sample(list(range(all_val_trg.size(0))), result_num)
        print_samples(all_val_src, all_val_trg, all_val_output, self.tokenizers, result_num, ids)

    
    def inference(self, query):
        query = make_inference_data(query, self.tokenizers[0], self.max_len)

        with torch.no_grad():
            query = query.to(self.device)
            self.model.eval()

            trg_word = torch.LongTensor([[self.tokenizers[1].sos_token_id]]).to(self.device)
            for j in range(self.max_len):
                if j == 0:
                    _, dec_output = self.model(query, trg_word)
                    trg_word = torch.cat((trg_word, torch.argmax(dec_output[:, -1], dim=-1).unsqueeze(1)), dim=1)
                else:
                    _, dec_output = self.model(query, trg_word)
                    trg_word = torch.cat((trg_word, torch.argmax(dec_output[:, -1], dim=-1).unsqueeze(1)), dim=1)
            
            output = self.tokenizers[1].decode(trg_word[0, 1:].tolist())
        
        return output


    def inference_score(self, result_num):
        phase = 'test'
        total_loss = 0
        all_val_src, all_val_trg, all_val_output = [], [], []

        with torch.no_grad():
            self.model.eval()

            for src, trg in self.dataloaders[phase]:
                batch = src.size(0)
                src, trg = src.to(self.device), trg.to(self.device)
            
                decoder_all_output = []
                for j in range(self.max_len):
                    if j == 0:
                        trg_word = trg[:, j].unsqueeze(1)
                        _, dec_output = self.model(src, trg_word)
                        trg_word = torch.cat((trg_word, torch.argmax(dec_output[:, -1], dim=-1).unsqueeze(1)), dim=1)
                    else:
                        _, dec_output = self.model(src, trg_word)
                        trg_word = torch.cat((trg_word, torch.argmax(dec_output[:, -1], dim=-1).unsqueeze(1)), dim=1)
                    decoder_all_output.append(dec_output[:, -1].unsqueeze(1))
                        
                decoder_all_output = torch.cat(decoder_all_output, dim=1)
                loss = self.criterion(decoder_all_output[:, :-1, :].reshape(-1, decoder_all_output.size(-1)), trg[:, 1:].reshape(-1))
                total_loss += loss.item()*batch
            
                all_val_src.append(src.detach().cpu())
                all_val_trg.append(trg.detach().cpu())
                all_val_output.append(decoder_all_output.detach().cpu())

        # calculate loss and ppl
        total_loss = total_loss / len(self.dataloaders[phase].dataset)
        print('Inference Score')
        print('loss: {}, ppl: {}'.format(total_loss, np.exp(total_loss)))

        # calculate scores
        all_val_trg_l, all_val_output_l = tensor2list(all_val_trg, all_val_output, self.trg_tokenizer)
        bleu2 = cal_scores(all_val_trg_l, all_val_output_l, 'bleu', 2)
        bleu4 = cal_scores(all_val_trg_l, all_val_output_l, 'bleu', 4)
        nist2 = cal_scores(all_val_trg_l, all_val_output_l, 'nist', 2)
        nist4 = cal_scores(all_val_trg_l, all_val_output_l, 'nist', 4)
        print('\nInference Score')
        print('bleu2: {}, bleu4: {}, nist2: {}, nist4: {}'.format(bleu2, bleu4, nist2, nist4))

        # visualize the attention score
        all_val_src = torch.cat(all_val_src, dim=0)
        all_val_trg = torch.cat(all_val_trg, dim=0)
        all_val_output = torch.argmax(torch.cat(all_val_output, dim=0), dim=2)
        ids = random.sample(list(range(all_val_trg.size(0))), result_num)
        print_samples(all_val_src, all_val_trg, all_val_output, self.tokenizers, result_num, ids)