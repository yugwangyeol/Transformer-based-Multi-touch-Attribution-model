import time
import random

import torch
import torch.nn as nn
import torch.optim as optim

from utils import epoch_time
from model.transformer_decoder import Transformer_decoder
from model.transformer_encoder import Transformer_encoder
from model.loss import MTA_Loss
from tqdm import tqdm
from utils import Params

from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,log_loss

import wandb
import numpy as np

# Wandb ##!
run = wandb.init(project="2024_capstoen")
params = Params('config/params.json')

wandb.config = {
    'num_epoch': params.num_epoch,
    "batch_size": params.batch_size,
}

random.seed(32)
torch.manual_seed(32)
torch.backends.cudnn.deterministic = True

class Trainer:
    def __init__(self, params, mode, train_iter=None, valid_iter=None, test_iter=None):
        self.params = params

        if mode == 'train':
            self.train_iter = train_iter
            self.valid_iter = valid_iter
        else:
            self.test_iter = test_iter

        self.encoder = Transformer_encoder(self.params)
        self.decoder = Transformer_decoder(self.params)
        self.encoder.to(self.params.device)
        self.decoder.to(self.params.device)

        self.optimizer = optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters()), lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4) # , betas=(0.9, 0.98), weight_decay=1e-4
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50, T_mult=2, eta_min=1e-10)
        
        self.criterion = MTA_Loss()
        self.criterion.to(self.params.device)

    def to(self, device):
        self.device = device
        self.encoder.to(device)
        self.decoder.to(device)
        self.criterion.to(device)

    def _move_batch_to_device(self, batch):
        if isinstance(batch, dict):
            return {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [self._move_batch_to_device(item) if isinstance(item, (dict, list, tuple)) else item.to(self.device) if isinstance(item, torch.Tensor) else item for item in batch]
        else:
            return batch.to(self.device)

    def train(self):
        print(f'The model has {self.encoder.count_params()+self.decoder.count_params():,} trainable parameters')
        #best_valid_loss = float('inf')
        best_valid_f1 = 0

        for epoch in range(self.params.num_epoch):
            self.encoder.train()
            self.decoder.train()
            epoch_loss = 0
            start_time = time.time()
            epoch_acc = 0
            epoch_f1 = 0
            conversion_epoch_loss = 0
            epoch_auc = 0
            epoch_log = 0

            for batch in tqdm(self.train_iter):
                self.optimizer.zero_grad()
                batch = self._move_batch_to_device(batch)

                cam_sequential = torch.stack([item['cam_sequential'] for item in batch])
                cate_sequential = torch.stack([item['cate_sequential'] for item in batch])
                price_sequential = torch.stack([item['price_sequential'] for item in batch])
                segment = torch.stack([item['segment'] for item in batch])

                conversion_label = torch.stack([item['label'] for item in batch])

                encoder_output = self.encoder(cam_sequential, cate_sequential, price_sequential, segment)
                conversion_output, attn_map = self.decoder(cam_sequential, cate_sequential, 
                                                price_sequential, segment,encoder_output)

                output = conversion_output.contiguous().view(-1, conversion_output.shape[-1]).squeeze(1)
                target = conversion_label.contiguous().view(-1)
                loss = self.criterion(output, target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters())+list(self.decoder.parameters()), self.params.clip)
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                output = np.where(output.cpu() > 0.5, 1 , 0)
                epoch_acc += accuracy_score(target.cpu(), output)
                epoch_f1 += f1_score(target.cpu(), output)
                epoch_auc += roc_auc_score(target.cpu(), output)
                epoch_log += log_loss(target.cpu(), output)

            train_loss = epoch_loss / len(self.train_iter)
            train_acc = epoch_acc / len(self.train_iter)
            train_f1 = epoch_f1 / len(self.train_iter)
            train_auc = epoch_auc / len(self.train_iter)
            train_log = epoch_log / len(self.train_iter)

            valid_loss, valid_acc, valid_f1, valid_auc, valid_log = self.evaluate()

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            wandb.log({"train_loss": train_loss}, step=epoch)
            wandb.log({"train_acc": train_acc}, step=epoch)
            wandb.log({"train_f1": train_f1}, step=epoch)
            wandb.log({"train_auc": train_auc}, step=epoch)
            wandb.log({"train_log": train_log}, step=epoch)
            
            wandb.log({"valid_loss": valid_loss}, step=epoch)
            wandb.log({"valid_acc": valid_acc}, step=epoch)
            wandb.log({"valid_f1": valid_f1}, step=epoch)
            wandb.log({"valid_auc": valid_auc}, step=epoch)
            wandb.log({"valid_log": valid_log}, step=epoch)

            if best_valid_f1 < valid_f1:
                best_valid_f1 = valid_f1
                torch.save(self.encoder.state_dict(), self.params.save_encoder)
                torch.save(self.decoder.state_dict(), self.params.save_decoder)

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | train. ACC : {train_acc:.3f} | train. F1 : {train_f1:.3f} | train. AUC : {train_auc:.3f} | train. log : {train_log:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} | Val. ACC : {valid_acc:.3f} | Val. F1 : {valid_f1:.3f} | Val. AUC : {valid_auc:.3f} | Val. log : {valid_log:.3f}')

    def evaluate(self):
        self.encoder.eval()
        self.decoder.eval()
        epoch_loss = 0
        epoch_acc = 0
        epoch_f1 = 0
        epoch_auc = 0
        epoch_log = 0

        with torch.no_grad():
            for batch in self.valid_iter: 
                batch = self._move_batch_to_device(batch)

                cam_sequential = torch.stack([item['cam_sequential'] for item in batch])
                cate_sequential = torch.stack([item['cate_sequential'] for item in batch])
                price_sequential = torch.stack([item['price_sequential'] for item in batch])
                segment = torch.stack([item['segment'] for item in batch])
                conversion_label = torch.stack([item['label'] for item in batch])

                encoder_output = self.encoder(cam_sequential, cate_sequential, price_sequential, segment)
                conversion_output, attn_map = self.decoder(cam_sequential, cate_sequential, 
                                                price_sequential, segment,encoder_output)

                output = conversion_output.contiguous().view(-1, conversion_output.shape[-1]).squeeze(1)
                target = conversion_label.contiguous().view(-1)
                loss = self.criterion(output, target)

                epoch_loss += loss.item()
                output = np.where(output.cpu() > 0.5, 1 , 0)
                epoch_acc += accuracy_score(target.cpu(), output)
                epoch_f1 += f1_score(target.cpu(), output)
                epoch_auc += roc_auc_score(target.cpu(), output)
                epoch_log += log_loss(target.cpu(), output)

        return epoch_loss / len(self.valid_iter), epoch_acc / len(self.valid_iter),  epoch_f1 / len(self.valid_iter) , epoch_auc / len(self.valid_iter), epoch_log / len(self.valid_iter)

    def inference(self):
        self.encoder.load_state_dict(torch.load(self.params.save_encoder))
        self.decoder.load_state_dict(torch.load(self.params.save_decoder))
        self.encoder.eval()
        self.decoder.eval()
        epoch_loss = 0
        epoch_acc = 0
        epoch_f1 = 0
        epoch_auc = 0
        epoch_log = 0

        with torch.no_grad():
            for batch in tqdm(self.test_iter): 
                batch = self._move_batch_to_device(batch)

                cam_sequential = torch.stack([item['cam_sequential'] for item in batch])
                cate_sequential = torch.stack([item['cate_sequential'] for item in batch])
                price_sequential = torch.stack([item['price_sequential'] for item in batch])
                segment = torch.stack([item['segment'] for item in batch])
                conversion_label = torch.stack([item['label'] for item in batch])

                encoder_output = self.encoder(cam_sequential, cate_sequential, price_sequential, segment)
                conversion_output, attn_map = self.decoder(cam_sequential, cate_sequential, 
                                                price_sequential, segment,encoder_output)

                output = conversion_output.contiguous().view(-1, conversion_output.shape[-1]).squeeze(1)
                target = conversion_label.contiguous().view(-1)
                loss = self.criterion(output, target)

                epoch_loss += loss.item()
                output = np.where(output.cpu() > 0.5, 1 , 0)
                epoch_acc += accuracy_score(target.cpu(), output)
                epoch_f1 += f1_score(target.cpu(), output)
                epoch_auc += roc_auc_score(target.cpu(), output)
                epoch_log += log_loss(target.cpu(), output)

        test_loss = epoch_loss / len(self.test_iter)
        test_acc = epoch_acc / len(self.test_iter)
        test_f1 = epoch_f1 / len(self.test_iter)
        test_auc = epoch_auc / len(self.test_iter)
        test_log = epoch_log / len(self.test_iter)

        print(f'Test Loss: {test_loss:.3f} | Test ACC: {test_acc:.3f} | Test F1: {test_f1:.3f} | Test AUC: {test_auc:.3f} | Test log: {test_log:.3f}')
