import os
import re
import json
import pickle
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

from torchtext import data as ttd

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, data, max_sequence_length):
        self.cam_sequential, self.cate_sequential, self.price_sequential, self.segment, self.label, self.cms, self.gender, self.age, self.pvalue, self.shopping = self.pad_embed(data, max_sequence_length)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            'cam_sequential': self.cam_sequential[idx],
            'cate_sequential': self.cate_sequential[idx],
            'price_sequential': self.price_sequential[idx],
            'segment': self.segment[idx],
            'cms': self.cms[idx],
            'gender': self.gender[idx],
            'age': self.age[idx],
            'pvalue': self.pvalue[idx],
            'shopping': self.shopping[idx],
            'label': self.label[idx]
        }

    def pad_sequences(self, sequences, max_length, padding_value=0):
        padded_sequences = []
        for seq in sequences:
            seq = list(map(float, seq.split()))
            length = len(seq)
            seq = torch.tensor(seq)
            if length < max_length:
                padding = torch.full((max_length - length,), padding_value)
                padded_seq = torch.cat((seq, padding), dim=0)
            else:
                padded_seq = seq[:max_length]
            padded_sequences.append(padded_seq)
        return torch.stack(padded_sequences)

    def pad_embed(self, data, max_sequence_length):
        cam_sequential_padded = self.pad_sequences(data['cam_sequential'].values, max_sequence_length)
        cate_sequential_padded = self.pad_sequences(data['cate_sequential'].values, max_sequence_length)
        price_sequential_padded = self.pad_sequences(data['price_sequential'].values, max_sequence_length)
        segment = self.pad_sequences(data['segment'].values, 5)

        label = data['label'].astype(float).values
        cms = data['cms'].astype(float).values
        gender = data['gender'].astype(float).values
        age = data['age'].astype(float).values
        pvalue = data['pvalue'].astype(float).values
        shopping = data['shopping'].astype(float).values

        return cam_sequential_padded, cate_sequential_padded, \
            price_sequential_padded, segment, torch.tensor(label), \
            torch.tensor(cms), torch.tensor(gender), torch.tensor(age), torch.tensor(pvalue), torch.tensor(shopping)

def load_dataset(mode,max_seq):
    data_dir = '../../Data3'
    
    if mode == 'train':
        cam_sequential_path = os.path.join(data_dir, 'camp_train.csv')
        cate_sequential_path = os.path.join(data_dir, 'cate_train.csv')
        price_sequential_path = os.path.join(data_dir, 'price_train.csv')
        segment_path = os.path.join(data_dir, 'train_seg.csv')

        cam_sequential_data = pd.read_csv(cam_sequential_path, encoding='utf-8').iloc[:,1:]
        cate_sequential_data = pd.read_csv(cate_sequential_path, encoding='utf-8').iloc[:,1:]
        price_sequential_data = pd.read_csv(price_sequential_path, encoding='utf-8').iloc[:,1:]
        segment_data = pd.read_csv(segment_path, encoding='utf-8').iloc[:,1:]
        segment_data.columns = ['user_id', "cms_group_id", "gender", "age_level", "pvalue_level", "shopping_level", 'segment']

        data = cam_sequential_data
        data['cate_sequential'] = cate_sequential_data['seq_space_sep']
        data['price_sequential'] = price_sequential_data['seq_space_sep']

        data = pd.merge(data, segment_data, on='user_id')
        data = data.drop(['user_id', 'num_user'], axis=1)
        data.columns = ['cam_sequential', 'label', 'cate_sequential', 'price_sequential', 'cms',
                        'gender', 'age', 'pvalue', 'shopping', 'segment']

        train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42)

        print(f'Number of training examples: {len(train_data)}')
        print(f'Number of validation examples: {len(valid_data)}')

        return CustomDataset(train_data, max_seq), CustomDataset(valid_data, max_seq)

    else:
        cam_sequential_path = os.path.join(data_dir, 'camp_test.csv')
        cate_sequential_path = os.path.join(data_dir, 'cate_test.csv')
        price_sequential_path = os.path.join(data_dir, 'price_test.csv')
        segment_path = os.path.join(data_dir, 'test_seg.csv')

        cam_sequential_data = pd.read_csv(cam_sequential_path, encoding='utf-8').iloc[:,1:]
        cate_sequential_data = pd.read_csv(cate_sequential_path, encoding='utf-8').iloc[:,1:]
        price_sequential_data = pd.read_csv(price_sequential_path, encoding='utf-8').iloc[:,1:]
        segment_data = pd.read_csv(segment_path, encoding='utf-8').iloc[:,1:]
        segment_data.columns = ['user_id', "cms_group_id", "gender", "age_level", "pvalue_level", "shopping_level", 'segment']

        data = cam_sequential_data
        data['cate_sequential'] = cate_sequential_data['seq_space_sep']
        data['price_sequential'] = price_sequential_data['seq_space_sep']

        test_data = pd.merge(data, segment_data, on='user_id')
        test_data = test_data.drop(['user_id', 'num_user'], axis=1)
        test_data.columns = ['cam_sequential', 'label', 'cate_sequential', 'price_sequential', 'cms',
                        'gender', 'age', 'pvalue', 'shopping', 'segment']
        
        print(f'Number of testing examples: {len(test_data)}')

        return CustomDataset(test_data, max_seq)

def make_iter(batch_size, mode, train_data=None, valid_data=None, test_data=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if mode == 'train':
        print(f'Make Iterators for training ....')
        train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
        valid_iter = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
        return train_iter, valid_iter
    else:
        print(f'Make Iterators for testing...')
        test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
        return test_iter

def epoch_time(start_time,end_tiem): # epoch 시간 재는 함수
    elapsed_time = end_tiem - start_time
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time - (elapsed_mins*60))

    return elapsed_mins, elapsed_secs

# utils.py

def load_and_prepare_vocab(train_data, valid_data=None):
    if valid_data is None:
        cam_input_dim = train_data.cam_sequential.max().item()
        cate_input_dim = train_data.cate_sequential.max().item()
        price_input_dim = train_data.price_sequential.max().item()
    else:
        cam_input_dim = max(train_data.cam_sequential.max().item(), valid_data.cam_sequential.max().item())
        cate_input_dim = max(train_data.cate_sequential.max().item(), valid_data.cate_sequential.max().item())
        price_input_dim = max(train_data.price_sequential.max().item(), valid_data.price_sequential.max().item())
    
    return cam_input_dim + 1, cate_input_dim + 1, price_input_dim + 1


def display_attention(condidate, translation, attention):
    """
    Args:
        condidate: (목록) 토큰화된 소스 토큰
        translation: (목록) 예측된 대상 번역 토큰
        attention: 주의 점수를 포함하는 텐서
    """
    # attention = [target length, source length]

    attention = attention.cpu().detach().numpy()

    font_location = 'pickles/NanumSquareR.ttf'
    fontprop = fm.FontProperties(font_location)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    ax.matshow(attention, cmap='bone')
    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + [t.lower() for t in candidate], rotation=45, fontproperties=fontprop)
    ax.set_yticklabels([''] + translation, fontproperties=fontprop)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()

class Params:
    """
    Class that loads hyperparameters from a json file
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)
        #self.load_vocab()

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def load_vocab(self, cam_input_dim, cate_input_dim, price_input_dim):

        # add device information to the the params
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # add <sos> and <eos> tokens' indices used to predict the target sentence
        params = {
            'cam_input_dim': cam_input_dim,
            'cate_input_dim': cate_input_dim,
            'price_input_dim': price_input_dim,
            'output_dim': 512,
            'device': device,
            ''
            'pad_idx': 0
        }

        self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__
