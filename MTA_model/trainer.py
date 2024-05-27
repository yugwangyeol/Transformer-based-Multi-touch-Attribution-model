import time
import random

import torch
import torch.nn as nn
import torch.optim as optim

from utils import epoch_time
from model.optim import ScheduledAdam
from model.transformer import Transformer
from model.loss import MTA_Loss
from tqdm import tqdm
from utils import Params

from sklearn.metrics import accuracy_score

import wandb

# Wandb ##!
run = wandb.init(project="2024_capstoen")
params = Params('config/params.json')

wandb.config = {
    'num_epoch' : params.num_epoch,
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

        self.model = Transformer(self.params)
        self.model.to(self.params.device)

        self.optimizer = ScheduledAdam(
            optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-9),
            hidden_dim=params.hidden_dim,
            warm_steps=params.warm_steps
        )

        self.criterion = MTA_Loss()
        self.criterion.to(self.params.device)

    def train(self):
        #print(self.model)
        print(f'The model has {self.model.count_params():,} trainable parameters')
        best_valid_loss = float('inf')

        for epoch in tqdm(range(self.params.num_epoch)):
            self.model.train()
            epoch_loss = 0
            conversion_epoch_loss = 0
            start_time = time.time()

            for batch in self.train_iter:
                self.optimizer.zero_grad()

                cam_sequential = torch.stack([item['cam_sequential'] for item in batch])
                cate_sequential = torch.stack([item['cate_sequential'] for item in batch])
                price_sequential = torch.stack([item['price_sequential'] for item in batch])
                segment = torch.stack([item['segment'] for item in batch])
                cms_label = torch.stack([item['cms'] for item in batch])
                gender_label = torch.stack([item['gender'] for item in batch])
                age_label = torch.stack([item['age'] for item in batch])
                pvalue_label = torch.stack([item['pvalue'] for item in batch])
                shopping_label = torch.stack([item['shopping'] for item in batch])
                conversion_label = torch.stack([item['label'] for item in batch])

                cms_output, gender_output, age_output, pvalue_output, shopping_output, conversion_output, attn_map = self.model(
                    cam_sequential, cate_sequential, price_sequential, segment)

                output = conversion_output.contiguous().view(-1, conversion_output.shape[-1]).squeeze(1)
                target = conversion_label.contiguous().view(-1)
                loss, conversion_loss = self.criterion(cms_output, cms_label, gender_output, gender_label, age_output, age_label,
                                        pvalue_output, pvalue_label, shopping_output, shopping_label, output, target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip)
                self.optimizer.step()

                epoch_loss += loss.item()
                conversion_epoch_loss += conversion_loss.item()

            train_loss = epoch_loss / len(self.train_iter)
            train_conversion_loss = conversion_epoch_loss / len(self.train_iter)
            valid_loss,valid_acc,valid_conversion_loss = self.evaluate()

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            wandb.log({"train_loss": train_loss}, step=epoch)
            wandb.log({"train_conversion_loss": train_conversion_loss}, step=epoch)
            wandb.log({"valid_loss": valid_loss}, step=epoch)
            wandb.log({"valid_conversion_loss": valid_conversion_loss}, step=epoch)
            wandb.log({"valid_acc": valid_acc}, step=epoch)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.params.save_model)

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Conversion Loss : {train_conversion_loss:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Conversion Loss : {valid_conversion_loss:.3f} | Val. ACC : {valid_acc:.3f}')

    def evaluate(self):
        self.model.eval()
        epoch_loss = 0 # 에폭 로스 0으로 설정
        epoch_conversion_loss = 0
        epoch_acc = 0

        with torch.no_grad():
            for batch in self.valid_iter: 
                cam_sequential = torch.stack([item['cam_sequential'] for item in batch])
                cate_sequential = torch.stack([item['cate_sequential'] for item in batch])
                price_sequential = torch.stack([item['price_sequential'] for item in batch])
                segment = torch.stack([item['segment'] for item in batch])
                cms_label = torch.stack([item['cms'] for item in batch])
                gender_label = torch.stack([item['gender'] for item in batch])
                age_label = torch.stack([item['age'] for item in batch])
                pvalue_label = torch.stack([item['pvalue'] for item in batch])
                shopping_label = torch.stack([item['shopping'] for item in batch])
                conversion_label = torch.stack([item['label'] for item in batch])

                cms_output, gender_output, age_output, pvalue_output, shopping_output, conversion_output, attn_map = self.model(
                    cam_sequential, cate_sequential, price_sequential, segment)

                output = conversion_output.contiguous().view(-1, conversion_output.shape[-1]).squeeze(1)
                target = conversion_label.contiguous().view(-1)
                loss,conversion_loss = self.criterion(cms_output, cms_label, gender_output, gender_label, age_output, age_label,
                                        pvalue_output, pvalue_label, shopping_output, shopping_label, output, target)

                epoch_loss += loss.item()
                epoch_conversion_loss += conversion_loss.item()
                epoch_acc += accuracy_score(target, output)



        return epoch_loss / len(self.valid_iter), epoch_acc / len(self.valid_iter), epoch_conversion_loss / len(self.valid_iter)

    def inference(self):
        self.model.load_state_dict(torch.load(self.params.save_model))
        self.model.eval()
        epoch_loss = 0 # 에폭 로스 0으로 설정
        epoch_conversion_loss = 0
        epoch_acc = 0

        with torch.no_grad():
            for batch in self.test_iter:
                cam_sequential = torch.stack([item['cam_sequential'] for item in batch])
                cate_sequential = torch.stack([item['cate_sequential'] for item in batch])
                price_sequential = torch.stack([item['price_sequential'] for item in batch])
                segment = torch.stack([item['segment'] for item in batch])
                cms_label = torch.stack([item['cms'] for item in batch])
                gender_label = torch.stack([item['gender'] for item in batch])
                age_label = torch.stack([item['age'] for item in batch])
                pvalue_label = torch.stack([item['pvalue'] for item in batch])
                shopping_label = torch.stack([item['shopping'] for item in batch])
                conversion_label = torch.stack([item['label'] for item in batch])

                cms_output, gender_output, age_output, pvalue_output, shopping_output, conversion_output, attn_map = self.model(
                    cam_sequential, cate_sequential, price_sequential, segment)

                output = conversion_output.contiguous().view(-1, conversion_output.shape[-1]).squeeze(1)
                target = conversion_label.contiguous().view(-1)
                loss,conversion_loss = self.criterion(cms_output, cms_label, gender_output, gender_label, age_output, age_label,
                                        pvalue_output, pvalue_label, shopping_output, shopping_label, output, target)

                epoch_loss += loss.item()
                epoch_conversion_loss += conversion_loss.item()
                epoch_acc += accuracy_score(target, output)
                #print(output.unique())

        test_loss = epoch_loss / len(self.test_iter)
        epoch_conversion_loss = epoch_conversion_loss / len(self.test_iter)
        epoch_acc = epoch_acc / len(self.test_iter)
        print(f'Test Loss: {test_loss:.3f} | Test Conversion Loss: {epoch_acc:.3f}| Test ACC: {epoch_acc:.3f}')
