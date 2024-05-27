# prepare_vocab.py
import pickle
from utils import load_dataset, load_and_prepare_vocab

def main():
    # train 데이터셋과 valid 데이터셋 로드
    train_data, valid_data = load_dataset('train')
    
    # vocab 준비
    cam_input_dim, cate_input_dim, price_input_dim = load_and_prepare_vocab(train_data, valid_data)
    
    # vocab 정보를 딕셔너리로 저장
    vocab_info = {
        'cam_input_dim': cam_input_dim,
        'cate_input_dim': cate_input_dim,
        'price_input_dim': price_input_dim
    }
    
    # pkl 파일로 저장
    with open('../../Data/vocab.pkl', 'wb') as f:
        pickle.dump(vocab_info, f)

if __name__ == '__main__':
    main()