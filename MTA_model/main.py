import csv
import argparse
from trainer import Trainer
from utils import load_dataset, make_iter, Params

def main(config):
    # 파라미터 파일 로드
    params = Params('config/params.json')

    if config.mode == 'train':
        # train 데이터셋과 valid 데이터셋 로드
        train_data, valid_data = load_dataset('train')
        
        # cam_sequential, cate_sequential, price_sequential의 최대 인덱스 값 추출
        cam_input_dim = max(train_data.cam_sequential.max().item(), valid_data.cam_sequential.max().item())
        cate_input_dim = max(train_data.cate_sequential.max().item(), valid_data.cate_sequential.max().item())
        price_input_dim = max(train_data.price_sequential.max().item(), valid_data.price_sequential.max().item())
        #print(cam_input_dim, cate_input_dim, price_input_dim)
        # params에 최대 인덱스 값을 로드
        params.load_vocab(cam_input_dim+1, cate_input_dim+1, price_input_dim+1)

        # train_iter, valid_iter 생성
        train_iter, valid_iter = make_iter(params.batch_size, config.mode,
                                        train_data=train_data, valid_data=valid_data)

        # 트레이너 생성 및 학습
        trainer = Trainer(params, config.mode, train_iter=train_iter, valid_iter=valid_iter)
        trainer.train()

    else:
        # test 데이터셋 로드
        test_data = load_dataset(config.mode)
        
        train_data, valid_data = load_dataset('train')
        
        # cam_sequential, cate_sequential, price_sequential의 최대 인덱스 값 추출
        cam_input_dim = max(train_data.cam_sequential.max().item(), valid_data.cam_sequential.max().item())
        cate_input_dim = max(train_data.cate_sequential.max().item(), valid_data.cate_sequential.max().item())
        price_input_dim = max(train_data.price_sequential.max().item(), valid_data.price_sequential.max().item())
        #print(cam_input_dim, cate_input_dim, price_input_dim)
        # params에 최대 인덱스 값을 로드
        params.load_vocab(cam_input_dim+1, cate_input_dim+1, price_input_dim+1)

        # test_iter 생성
        test_iter = make_iter(params.batch_size, config.mode, test_data=test_data)

        # 트레이너 생성 및 추론
        trainer = Trainer(params, config.mode, test_iter=test_iter)
        trainer.inference()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer based MTA') # parser 생성
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test']) # argument 추가
    args = parser.parse_args() # parser 변수 지정
    main(args) # main 함수 실행 
