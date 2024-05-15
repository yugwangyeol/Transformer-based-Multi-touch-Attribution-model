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
from torchtext.data import Example, Dataset

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, data, max_sequence_length):
        self.cam_sequential,self.cate_sequential,self.brand_sequential,self.price_sequential, self.segment, self.label, self.cms, self.gender, self.age, self.pvalue, self.shopping = self.pad_embed(data, max_sequence_length)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'cam_sequential': self.cam_sequential[idx],
            'cate_sequential': self.cate_sequential[idx],
            'brand_sequential': self.brand_sequential[idx],
            'price_sequential': self.price_sequential[idx],
            'segment': self.segment[idx],
            'cms': self.cms[idx],
            'gender': self.gender[idx],
            'age': self.age[idx],
            'pvalue': self.pvalue[idx],
            'shopping': self.shopping[idx],
            'label': self.labels[idx]
        }

    def pad_embed(self, data, max_sequence_length):
        # Padding
        cam_sequential_padded = pad_sequence(data['cam_sequential'], maxlen=max_sequence_length, padding='post')
        cate_sequential_padded = pad_sequence(data['cate_sequential'], maxlen=max_sequence_length, padding='post')
        brand_sequential_padded = pad_sequence(data['brand_sequential'], maxlen=max_sequence_length, padding='post')
        price_sequential_padded = pad_sequence(data['price_sequential'], maxlen=max_sequence_length, padding='post')

        return torch.tensor(cam_sequential_padded), torch.tensor(cate_sequential_padded), \
                torch.tensor(brand_sequential_padded), torch.tensor(price_sequential_padded), \
                torch.tensor(data['segment']), torch.tensot(data['label']), \
                torch.tensor(data['cms']), torch.tensor(data['gender']), torch.tensor(data['age']), torch.tensor(data['pvalue']), torch.tensor(data['shopping'])

def load_dataset(mode):
    data_dir = Path().cwd() / 'data' # directory 설정

    if mode == 'train': # mode 설정
        cam_sequential_path = os.path.join(data_dir, 'cam_sequential.csv')
        cate_sequential_path = os.path.join(data_dir, 'cate_sequential.csv')
        brand_sequential_path = os.path.join(data_dir, 'brand_sequential.csv')
        price_sequential_path = os.path.join(data_dir, 'price_sequential.csv')
        segment_path = os.path.join(data_dir, 'segment.csv')

        cam_sequential_data = pd.read_csv(cam_sequential_path, encoding='utf-8')
        cate_sequential_data = pd.read_csv(cate_sequential_path, encoding='utf-8')
        brand_sequential_data = pd.read_csv(brand_sequential_path, encoding='utf-8')
        price_sequential_data = pd.read_csv(price_sequential_path, encoding='utf-8')
        segment_data = pd.read_csv(segment_path, encoding='utf-8')

        data = cam_sequential_data
        data['cate_sequential'] = cate_sequential_data['cate_sequential']
        data['brand_sequential'] = brand_sequential_data['brand_sequential']
        data['price_sequential'] = price_sequential_data['price_sequential']

        data = pd.merge(data, segment_data, on='User_id')

        # train, valid 데이터셋으로 나누기
        train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42)

        print(f'Number of training examples: {len(train_data)}')
        print(f'Number of validation examples: {len(valid_data)}')

        return CustomDataset(train_data), CustomDataset(valid_data)

    else:
        test_sequential_path = os.path.join(data_dir, 'test_sequential.csv')
        test_segment_path = os.path.join(data_dir, 'segment.csv')
        test_sequential_data = pd.read_csv(test_sequential_path, encoding='utf-8')
        test_segment_data = pd.read_csv(test_segment_path, encoding='utf-8')

        test_data = pd.merge(test_sequential_data, test_segment_data, on='User_id')

        print(f'Number of testing examples: {len(test_data)}')

        return CustomDataset(test_data)


def make_iter(batch_size,mode,train_data,valid_data,test_data):
    #Panda DataFrame을 Torchtext Dataset으로 변환하고 모델을 교육하고 테스트하는 데 사용할 반복기를 만듬
    # load text and label field made by build_pickles.py
    # build_message에서 만든 텍스트 및 레이블 필드를 로드

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # device 설정

    if mode=='train':
        train_data = load_dataset(train_data)
        valid_data = load_dataset(valid_data)

        print(f'Make Iterators for training ....')

        '''
        Pytorch의 dataloader와 비슷한 역할을 함
        하지만 dataloader 와 다르게 비슷한 길이의 문장들끼리 batch를 만들기 때문에 padding의 개수를 최소화할 수 있음
        '''
        train_iter, valid_iter = ttd.BucketIterator.splits(
            (train_data,valid_data),
            # 버킷 반복기는 데이터를 그룹화하기 위해 어떤 기능을 사용해야 하는지 알려주어야 함
            # 우리의 경우, 우리는 예제 텍스트를 사용하여 데이터 세트를 정렬
            batch_size=batch_size,
            device=device
        )
        #여기서 BucketIterator.splits 함수는 여러 데이터셋을 사용하여 BucketIterator를 생성
        #이렇게 설정된 train_iter와 valid_iter는 데이터셋을 미니배치로 나누어주는 반복자(iterator) 역

        return train_iter,valid_iter
    else:
        test_data = load_dataset(test_data)
        dummy = list()

        print(f'Make Iterators for testing...')

        test_iter, _ = ttd.BucketIterator.splits(
            (test_data,dummy),
            batch_size=batch_size,
            device=device)
    return test_iter

def epoch_time(start_time,end_tiem): # epoch 시간 재는 함수
    elapsed_time = end_tiem - start_time
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time - (elapsed_mins*60))

    return elapsed_mins, elapsed_secs

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
    json 파일에서 하이퍼파라미터를 로드하는 클래스
    예제:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5 # params의 learning_rate 값 변경
    ```
    """

    def __init__(self,json_path):
        self.update(json_path)
        self.load_vocab()

    def update(self,json_path):
        #Loads parameters from json file
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
    
    def load_vocab(self):
        pickle_kor = open('pickles/kor.pickle', 'rb')
        kor = pickle.load(pickle_kor)

        pickle_eng = open('pickles/eng.pickle', 'rb')
        eng = pickle.load(pickle_eng)

        # add device information to the the params
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # add <sos> and <eos> tokens' indices used to predict the target sentence
        params = {'input_dim': len(kor.vocab), 'output_dim': len(eng.vocab),
                'sos_idx': eng.vocab.stoi['<sos>'], 'eos_idx': eng.vocab.stoi['<eos>'],
                'pad_idx': eng.vocab.stoi['<pad>'], 'device': device}

        self.__dict__.update(params)
    
    @property
    def dict(self):
        #Params 인스턴스에 `params.dict['learning_rate']`로 딕트와 유사한 접근 권한을 부여.
        return self.__dict__ 