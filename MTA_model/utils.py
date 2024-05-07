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

def load_dataset(mode):
    print(f'Loading AI Hub Kor-Eng translation dataset and converting it to pandas DataFrame . . .')

    data_dir = Path().cwd() / 'data' # directory 설정

    if mode == 'train': # mode 설정
        train_file = os.path.join(data_dir, 'train.csv')
        train_data = pd.read_csv(train_file, encoding='utf-8') # csv 불러오기

        valid_file = os.path.join(data_dir, 'valid.csv')
        valid_data = pd.read_csv(valid_file, encoding='utf-8') # csv 불러오기

        print(f'Number of training examples: {len(train_data)}')
        print(f'Number of validation examples: {len(valid_data)}')

        return train_data, valid_data # data return

    else:
        test_file = os.path.join(data_dir, 'test.csv')
        test_data = pd.read_csv(test_file, encoding='utf-8')

        print(f'Number of testing examples: {len(test_data)}')

        return test_data

def clean_text(text):
    """
    입력문에서 특수문자를 삭제하여 정규화하다
    """
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`…》]', '', text) #특수문자를 찾고 변형 
    return text

def convert_to_dataset(data, kor, eng):
    """
    입력 DataFrame을 전처리하고 Panda DataFrame을 Torchtext Dataset으로 변환합니다.
    Args:
        data: (DataFrame) pandas DataFrame to be converted into torchtext Dataset
        kor: torchtext Field containing Korean sentence
        eng: torchtext Field containing English sentence

    Returns:
        (Dataset) torchtext Dataset containing 'kor' and 'eng' Fields
    """
    # drop missing values not containing str value from DataFrame
    # DataFrame에서 str 값을 포함하지 않는 결측값 삭제
    missing_rows = [idx for idx, row in data.iterrows() if type(row.korean) != str or type(row.english) != str]
    data = data.drop(missing_rows)

    # convert each row of DataFrame to torchtext 'Example' containing 'kor' and 'eng' Fields
    # 데이터 프레임의 각 행을 'kor' 및 'eng' 필드가 포함된 '예시' 텍스트로 변환
    list_of_examples = [Example.fromlist(row.apply(lambda x: clean_text(x)).tolist(), # -> 텍스트 전처리 위 함수 ->
                                        fields=[('kor', kor), ('eng', eng)]) for _, row in data.iterrows()]
    
    #Example.fromlist(): Example 클래스의 정적 메서드로, 리스트 형태의 데이터를 받아 Example 객체를 생성, 여기서는 fields 파라미터를 사용하여 어떤 필드에 어떤 데이터를 저장할지를 지정
    #[('kor', kor), ('eng', eng)]: Example 객체에 저장될 필드들을 지정하는데, ('kor', kor)는 'kor'라는 필드에 대응되는 데이터는 kor 변수에서 가져오고, 'eng'라는 필드에 대응되는 데이터는 eng 변수에서 가져온다는 의미

    # construct torchtext 'Dataset' using torchtext 'Example' list
    dataset = Dataset(examples=list_of_examples, fields=[('kor', kor), ('eng', eng)])

    return dataset

def make_iter(batch_size,mode,train_data,valid_data,test_data):
    #Panda DataFrame을 Torchtext Dataset으로 변환하고 모델을 교육하고 테스트하는 데 사용할 반복기를 만듬
    # load text and label field made by build_pickles.py
    # build_message에서 만든 텍스트 및 레이블 필드를 로드

    file_kor = open('pickles/kor.pickle','rb') 
    # 한국 pickle
    #rb는 파일을 바이너리 읽기 모드로 열겠다는 것을 나타냄, read,binary
    kor = pickle.load(file_kor)

    file_eng = open('pickles/eng.pickle','rb')
    eng = pickle.load(file_eng)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # device 설정

    if mode=='train':
        train_data = convert_to_dataset(train_data,kor,eng)
        valid_data = convert_to_dataset(valid_data,kor,eng)

        print(f'Make Iterators for training ....')

        '''
        Pytorch의 dataloader와 비슷한 역할을 함
        하지만 dataloader 와 다르게 비슷한 길이의 문장들끼리 batch를 만들기 때문에 padding의 개수를 최소화할 수 있음
        '''
        train_iter, valid_iter = ttd.BucketIterator.splits(
            (train_data,valid_data),
            # 버킷 반복기는 데이터를 그룹화하기 위해 어떤 기능을 사용해야 하는지 알려주어야 함
            # 우리의 경우, 우리는 예제 텍스트를 사용하여 데이터 세트를 정렬
            sort_key= lambda sent : len(sent.kor),
            sort_within_batch=True,
            batch_size=batch_size,
            device=device
        )
        #여기서 BucketIterator.splits 함수는 여러 데이터셋을 사용하여 BucketIterator를 생성
        #sort_key: 데이터를 정렬할 기준을 지정 -> 이 경우에는 kor 필드의 길이를 기준으로 데이터를 정렬하도록 되어 있음
        #sort_within_batch: 미니배치 내에서도 정렬을 수행할지 여부를 결정
        #이렇게 설정된 train_iter와 valid_iter는 데이터셋을 미니배치로 나누어주는 반복자(iterator) 역

        return train_iter,valid_iter
    else:
        test_data = convert_to_dataset(test_data,kor,eng)
        dummy = list()

        print(f'Make Iterators for testing...')

        test_iter, _ = ttd.BucketIterator.splits(
            (test_data,dummy),
            sort_key=lambda sent:len(sent.kor),
            sort_within_batch=True,
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