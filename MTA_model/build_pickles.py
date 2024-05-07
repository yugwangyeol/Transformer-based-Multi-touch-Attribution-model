import os
import pickle
import argparse

import pandas as pd
from pathlib import Path
from utils import convert_to_dataset

from torchtext import data as ttd
from soynlp.word import WordRxtractor
from soynlp.torkenizer import LTokenizer

def build_torkenizer():
    print(f'Now building soy-nlp torkenizer ....')

    data_dir = Path().cwd() /'data'
    train_file = os.path.join(data_dir, 'courpus.csv')

    df = pd.read_csv(train_file,encoding='utf-8')

    kor_lines = [row.korean for _, row in df.iterrows() if type(row.korean) == str]

    word_extractor = WordRxtractor(min_frequency=5)
    word_extractor.train(kor_lines)

    word_scores = word_extractor.extract()
    #학습된 단어 추출기를 이용하여 단어의 응집도 점수를 추출
    #extract 메서드는 추출된 단어와 각 단어의 응집도 점수를 담은 객체를 반환함 word_scores는 단어와 해당 응집도 점수를 담은 딕셔너리
    cohesion_score = {word: score.cohesion_forward for word, score in word_scores.items()}
    #word_scores에서 단어와 응집도 점수를 추출하여 새로운 딕셔너리 cohesion_score에 저장함
    #이때, 응집도 점수는 score.cohesion_forward로 접근
    #응집도 점수는 해당 단어가 얼마나 응집되어 있는지를 나타내며, 높을수록 해당 단어가 응집도 높은 단어일 가능성이 있

    with open('pickles/torkenizer.pickle','wb') as pickle_out:
        pickle.dump(cohesion_score,pickle_out)

def build_vocab(config):
    #입력 문장을 단어 인덱스로 변환하는 데 사용되는 어휘를 soynlp 및 spacy 토큰화기를 사용하여 빌드

    pickle_torkenizer = open('pickles/torkenizer.pickle','rb')
    cohesion_scores = pickle.load(pickle_torkenizer)
    tokenizer = LTokenizer(cohesion_scores)

    kor = ttd.Field(tokenizer=tokenizer.tokenize,
                    lower=True,
                    batch_first=True)
    
    eng = ttd.Field(tokenizer='spacy',
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True,
                    batch_first=True)
    
    data_dir = Path().cwd()/'data'
    train_file = os.path.join(data_dir,'train.csv')
    train_data = pd.read_csv(train_file,encoding='utf-8')
    train_data = convert_to_dataset(train_data)

    print(f'Build vocabulary using torchtext ...')

    kor.build_vocab(train_data, max_szie=config.kor_vocab)
    eng.build_vocab(train_data,max_size=config.eng_vocab)

    print(f'Unique tokens in Korean vocabulary: {len(kor.vocab)}')
    print(f'Unique tokens in English vocabulary: {len(eng.vocab)}')

    print(f'Most commonly used Korean words are as follows:')
    print(kor.vocab.freqs.most_common(20))

    print(f'Most commonly used English words are as follows:')
    print(eng.vocab.freqs.most_common(20))

    with open('pickles/kor.pickle', 'wb') as kor_file:
        pickle.dump(kor, kor_file)

    with open('pickles/eng.pickle', 'wb') as eng_file:
        pickle.dump(eng, eng_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pickle Builder')

    parser.add_argument('--kor_vocab',type=int,default=55000)
    parser.add_argument('--eng_vocab',type=int,default=30000)

    config = parser.parse_args()

    build_torkenizer()
    build_vocab(config)


