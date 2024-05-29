import csv
import argparse
import pickle
import torch
from trainer import Trainer
from utils import load_dataset, make_iter, Params

def main(config):
    # 파라미터 파일 로드
    params = Params('config/params.json')
    
    # vocab.pkl 파일 로드
    with open('../../Data/vocab.pkl', 'rb') as f:
        vocab_info = pickle.load(f)
    
    # params에 최대 인덱스 값을 로드
    params.load_vocab(vocab_info['cam_input_dim'], vocab_info['cate_input_dim'], vocab_info['price_input_dim'])

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if config.mode == 'train':
        # train 데이터셋과 valid 데이터셋 로드
        train_data, valid_data = load_dataset('train', params.max_seq)

        # train_iter, valid_iter 생성
        train_iter, valid_iter = make_iter(params.batch_size, config.mode,
                                           train_data=train_data, valid_data=valid_data)

        # 트레이너 생성 및 학습
        trainer = Trainer(params, config.mode, train_iter=train_iter, valid_iter=valid_iter)
        trainer.to(device)  # 모델과 데이터 이동
        trainer.train()

    else:
        # test 데이터셋 로드
        test_data = load_dataset(config.mode, params.max_seq)

        # test_iter 생성
        test_iter = make_iter(params.batch_size, config.mode, test_data=test_data)

        # 트레이너 생성 및 추론
        trainer = Trainer(params, config.mode, test_iter=test_iter)
        trainer.to(device)  # 모델과 데이터 이동
        trainer.inference()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer based MTA')  # parser 생성
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])  # argument 추가
    args = parser.parse_args()  # parser 변수 지정
    main(args)  # main 함수 실행
