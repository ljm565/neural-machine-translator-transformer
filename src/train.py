import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
import pickle
from tokenizer import Tokenizer_iwslt_ende, Tokenizer_wmt_ende
import time
import random
from tqdm import tqdm

from utils.config import Config
from utils.utils_func import *
from utils.utils_data import DLoader_wmt, DLoader_iwslt
from models.model import Transformer as TransformerNMT



class Trainer_iwslt_ende:
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
        self.result_num = self.config.result_num

        # define tokenizer (WMT uses shared tokenizer)
        self.src_tokenizer = Tokenizer_iwslt_ende(self.config)
        self.trg_tokenizer = Tokenizer_iwslt_ende(self.config)
        self.tokenizers = [self.src_tokenizer, self.trg_tokenizer]

        # dataloader
        torch.manual_seed(999)  # for reproducibility
        if self.mode == 'train':
            self.dataset = {s: DLoader_iwslt(load_dataset(p), self.tokenizers, self.config) for s, p in self.data_path.items()}
            self.dataloaders = {
                s: DataLoader(d, self.batch_size, shuffle=True, num_workers=4) if s == 'train' else DataLoader(d, self.batch_size, shuffle=False, num_workers=4)
                for s, d in self.dataset.items()}
        else:
            self.dataset = {s: DLoader_iwslt(load_dataset(p), self.tokenizers, self.config) for s, p in self.data_path.items() if s == 'test'}
            self.dataloaders = {s: DataLoader(d, self.batch_size, shuffle=False, num_workers=4) for s, d in self.dataset.items() if s == 'test'}

        # model, optimizer, loss
        milestones = list(range(10, self.epochs, 10))
        self.model = TransformerNMT(self.config, self.tokenizers, self.device).to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.trg_tokenizer.pad_token_id)

        if self.mode == 'train':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=0.8)
            if self.continuous:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(self.check_point['model'])
                self.optimizer.load_state_dict(self.check_point['optimizer'])
                del self.check_point
                torch.cuda.empty_cache()
        else:
            self.check_point = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(self.check_point['model'])    
            self.model.eval()
            del self.check_point
            torch.cuda.empty_cache()

        
    def training(self):
        early_stop = 0
        best_val_bleu = 0 if not self.continuous else self.loss_data['best_val_bleu']
        train_loss_history = [] if not self.continuous else self.loss_data['train_loss_history']
        val_score_history = {'bleu2': [], 'bleu4': [], 'nist2': [], 'nist4': []} if not self.continuous else self.loss_data['val_score_history']
        best_epoch_info = 0 if not self.continuous else self.loss_data['best_epoch']

        for epoch in range(self.epochs):
            start = time.time()
            print(epoch+1, '/', self.epochs)
            print('-'*10)
            for phase in ['train', 'val', 'test']:
                print('Phase: {}'.format(phase))
                if phase == 'train':
                    epoch_loss = self.train(phase, epoch)
                    train_loss_history.append(epoch_loss)
                    self.scheduler.step()
                else:
                    bleu2, bleu4, nist2, nist4 = self.inference(phase, self.result_num)
                    if phase == 'val':
                        val_score_history['bleu2'].append(bleu2)
                        val_score_history['bleu4'].append(bleu4)
                        val_score_history['nist2'].append(nist2)
                        val_score_history['nist4'].append(nist4)

                        # save best model
                        early_stop += 1
                        if  val_score_history['bleu4'][-1] > best_val_bleu:
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
        self.loss_data = {'best_epoch': best_epoch, 'best_val_bleu': best_val_bleu, 'train_loss_history': train_loss_history, 'val_score_history': val_score_history}
        return self.loss_data


    def train(self, phase, epoch):
        self.model.train()
        total_loss = 0

        for i, (src, trg) in enumerate(self.dataloaders[phase]):
            batch = src.size(0)
            src, trg = src.to(self.device), trg.to(self.device)
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase=='train'):
                _, output = self.model(src, trg)
                loss = self.criterion(output[:, :-1, :].reshape(-1, output.size(-1)), trg[:, 1:].reshape(-1))
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()*batch

            if i % 100 == 0:
                print('Epoch {}: {}/{} step loss: {}'.format(epoch+1, i, len(self.dataloaders[phase]), loss.item()))

        epoch_loss = total_loss/len(self.dataloaders[phase].dataset)

        print('{} loss: {:4f}\n'.format(phase, epoch_loss))
        return epoch_loss


    def inference(self, phase, result_num=3):
        self.model.eval()
        all_trg, all_output = [], []

        with torch.no_grad():
            for src, trg in tqdm(self.dataloaders[phase], desc=phase+' inferencing..'):
                src, trg = src.to(self.device), trg.to(self.device)
                all_trg.append(trg.detach().cpu())
            
                decoder_all_output = []
                for j in range(self.max_len):
                    if j == 0:
                        trg = trg[:, j].unsqueeze(1)
                        _, output = self.model(src, trg)
                        trg = torch.cat((trg, torch.argmax(output[:, -1], dim=-1).unsqueeze(1)), dim=1)
                    else:
                        _, output = self.model(src, trg)
                        trg = torch.cat((trg, torch.argmax(output[:, -1], dim=-1).unsqueeze(1)), dim=1)
                    decoder_all_output.append(output[:, -1].unsqueeze(1).detach().cpu())
                        
                all_output.append(torch.argmax(torch.cat(decoder_all_output, dim=1), dim=-1))

        # calculate scores
        all_ref, all_pred = tensor2list(all_trg, all_output, self.trg_tokenizer)
        bleu2 = cal_scores(all_ref, all_pred, 'bleu', 2)
        bleu4 = cal_scores(all_ref, all_pred, 'bleu', 4)
        nist2 = cal_scores(all_ref, all_pred, 'nist', 2)
        nist4 = cal_scores(all_ref, all_pred, 'nist', 4)
        print('\nInference Score')
        print('bleu2: {}, bleu4: {}, nist2: {}, nist4: {}'.format(bleu2, bleu4, nist2, nist4))

        # print samples
        ids = random.sample(list(range(len(all_pred))), result_num)
        print_samples(all_ref, all_pred, ids)

        return bleu2, bleu4, nist2, nist4


    def multi_bleu_perl(self, phase):
        self.model.eval()
        all_trg, all_output = [], []

        with torch.no_grad():
            for src, trg in tqdm(self.dataloaders[phase], desc=phase+' inferencing..'):
                src, trg = src.to(self.device), trg.to(self.device)
                all_trg.append(trg.detach().cpu())
            
                decoder_all_output = []
                for j in range(self.max_len):
                    if j == 0:
                        trg = trg[:, j].unsqueeze(1)
                        _, output = self.model(src, trg)
                        trg = torch.cat((trg, torch.argmax(output[:, -1], dim=-1).unsqueeze(1)), dim=1)
                    else:
                        _, output = self.model(src, trg)
                        trg = torch.cat((trg, torch.argmax(output[:, -1], dim=-1).unsqueeze(1)), dim=1)
                    decoder_all_output.append(output[:, -1].unsqueeze(1).detach().cpu())
                        
                all_output.append(torch.argmax(torch.cat(decoder_all_output, dim=1), dim=-1))
            
        # calculate scores
        all_ref, all_pred = tensor2list(all_trg, all_output, self.trg_tokenizer)
        cal_multi_bleu_perl(self.base_path, all_ref, all_pred)



class Trainer_wmt_ende:
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
        self.result_num = self.config.result_num

        # define tokenizer (WMT uses shared tokenizer)
        self.src_tokenizer = Tokenizer_wmt_ende()
        self.trg_tokenizer = Tokenizer_wmt_ende()
        self.tokenizers = [self.src_tokenizer, self.trg_tokenizer]

        # dataloader
        torch.manual_seed(999)  # for reproducibility
        if self.mode == 'train':
            self.dataset = {s: DLoader_wmt(load_dataset(p), self.tokenizers, self.config) for s, p in self.data_path.items()}
            self.dataloaders = {
                s: DataLoader(d, self.batch_size, shuffle=True, num_workers=4) if s == 'train' else DataLoader(d, self.batch_size, shuffle=False, num_workers=4)
                for s, d in self.dataset.items()}
        else:
            self.dataset = {s: DLoader_wmt(load_dataset(p), self.tokenizers, self.config) for s, p in self.data_path.items() if s == 'test'}
            self.dataloaders = {s: DataLoader(d, self.batch_size, shuffle=False, num_workers=4) for s, d in self.dataset.items() if s == 'test'}

        # model, optimizer, loss
        milestones = list(range(8, self.epochs, 8))
        self.model = TransformerNMT(self.config, self.tokenizers, self.device).to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.trg_tokenizer.pad_token_id)

        if self.mode == 'train':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=0.8)
            if self.continuous:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(self.check_point['model'])
                self.optimizer.load_state_dict(self.check_point['optimizer'])
                del self.check_point
                torch.cuda.empty_cache()
        else:
            self.check_point = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(self.check_point['model'])    
            self.model.eval()
            del self.check_point
            torch.cuda.empty_cache()

        
    def training(self):
        early_stop = 0
        best_val_bleu = 0 if not self.continuous else self.loss_data['best_val_bleu']
        train_loss_history = [] if not self.continuous else self.loss_data['train_loss_history']
        val_score_history = {'bleu2': [], 'bleu4': [], 'nist2': [], 'nist4': []} if not self.continuous else self.loss_data['val_score_history']
        best_epoch_info = 0 if not self.continuous else self.loss_data['best_epoch']

        for epoch in range(self.epochs):
            start = time.time()
            print(epoch+1, '/', self.epochs)
            print('-'*10)
            for phase in ['train', 'val', 'test']:
                print('Phase: {}'.format(phase))
                if phase == 'train':
                    epoch_loss = self.train(phase, epoch)
                    train_loss_history.append(epoch_loss)
                    self.scheduler.step()
                else:
                    bleu2, bleu4, nist2, nist4 = self.inference(phase, self.result_num)
                    if phase == 'val':
                        val_score_history['bleu2'].append(bleu2)
                        val_score_history['bleu4'].append(bleu4)
                        val_score_history['nist2'].append(nist2)
                        val_score_history['nist4'].append(nist4)

                        # save best model
                        early_stop += 1
                        if  val_score_history['bleu4'][-1] > best_val_bleu:
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
        self.loss_data = {'best_epoch': best_epoch, 'best_val_bleu': best_val_bleu, 'train_loss_history': train_loss_history, 'val_score_history': val_score_history}
        return self.loss_data


    def train(self, phase, epoch):
        self.model.train()
        total_loss = 0

        for i, (src, trg) in enumerate(self.dataloaders[phase]):
            batch = src.size(0)
            src, trg = src.to(self.device), trg.to(self.device)
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase=='train'):
                _, output = self.model(src, trg)
                loss = self.criterion(output[:, :-1, :].reshape(-1, output.size(-1)), trg[:, 1:].reshape(-1))
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()*batch

            if i % 1000 == 0:
                print('Epoch {}: {}/{} step loss: {}'.format(epoch+1, i, len(self.dataloaders[phase]), loss.item()))

        epoch_loss = total_loss/len(self.dataloaders[phase].dataset)

        print('{} loss: {:4f}\n'.format(phase, epoch_loss))
        return epoch_loss


    def inference(self, phase, result_num=3):
        self.model.eval()
        all_trg, all_output = [], []

        with torch.no_grad():
            for src, trg in tqdm(self.dataloaders[phase], desc=phase+' inferencing..'):
                src, trg = src.to(self.device), trg.to(self.device)
                all_trg.append(trg.detach().cpu())
            
                decoder_all_output = []
                for j in range(self.max_len):
                    if j == 0:
                        trg = trg[:, j].unsqueeze(1)
                        _, output = self.model(src, trg)
                        trg = torch.cat((trg, torch.argmax(output[:, -1], dim=-1).unsqueeze(1)), dim=1)
                    else:
                        _, output = self.model(src, trg)
                        trg = torch.cat((trg, torch.argmax(output[:, -1], dim=-1).unsqueeze(1)), dim=1)
                    decoder_all_output.append(output[:, -1].unsqueeze(1).detach().cpu())
                        
                all_output.append(torch.argmax(torch.cat(decoder_all_output, dim=1), dim=-1))
            
        # calculate scores
        all_ref, all_pred = tensor2list(all_trg, all_output, self.trg_tokenizer)
        bleu2 = cal_scores(all_ref, all_pred, 'bleu', 2)
        bleu4 = cal_scores(all_ref, all_pred, 'bleu', 4)
        nist2 = cal_scores(all_ref, all_pred, 'nist', 2)
        nist4 = cal_scores(all_ref, all_pred, 'nist', 4)
        print('\nInference Score')
        print('bleu2: {}, bleu4: {}, nist2: {}, nist4: {}'.format(bleu2, bleu4, nist2, nist4))

        # print samples
        ids = random.sample(list(range(len(all_pred))), result_num)
        print_samples(all_ref, all_pred, ids)

        return bleu2, bleu4, nist2, nist4


    def multi_bleu_perl(self, phase):
        self.model.eval()
        all_trg, all_output = [], []

        with torch.no_grad():
            for src, trg in tqdm(self.dataloaders[phase], desc=phase+' inferencing..'):
                src, trg = src.to(self.device), trg.to(self.device)
                all_trg.append(trg.detach().cpu())
            
                decoder_all_output = []
                for j in range(self.max_len):
                    if j == 0:
                        trg = trg[:, j].unsqueeze(1)
                        _, output = self.model(src, trg)
                        trg = torch.cat((trg, torch.argmax(output[:, -1], dim=-1).unsqueeze(1)), dim=1)
                    else:
                        _, output = self.model(src, trg)
                        trg = torch.cat((trg, torch.argmax(output[:, -1], dim=-1).unsqueeze(1)), dim=1)
                    decoder_all_output.append(output[:, -1].unsqueeze(1).detach().cpu())
                        
                all_output.append(torch.argmax(torch.cat(decoder_all_output, dim=1), dim=-1))
            
        # calculate scores
        all_ref, all_pred = tensor2list(all_trg, all_output, self.trg_tokenizer)
        cal_multi_bleu_perl(self.base_path, all_ref, all_pred)