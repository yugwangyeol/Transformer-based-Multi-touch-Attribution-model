import pickle
import numpy as np
import torch
import torch.nn as nn

pickle_eng = open('pickles/eng.pickle','rb')
eng = pickle.load(pickle_eng)
pad_idx = eng.vocab.stoi['<pad>']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weigth(layer): #가중치 초기회
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.constant_(layer.bias,0) #편향을 0으로 초기화

def create_positional_encoding(max_len, hidden_dim):
    # PE(pos, 2i)     = sin(pos/10000 ** (2*i / hidden_dim))
    # PE(pos, 2i + 1) = cos(pos/10000 ** (2*i / hidden_dim))

    sinusoid_table = np.array([pos/np.power(10000,2*i/hidden_dim) for pos in range(max_len) for i in range(hidden_dim)])

    sinusoid_table = sinusoid_table.reshape(max_len,-1)

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # calculate pe for even dimension 짝수 차원에 대한 pe 계산
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # calculate pe for odd dimension 홀수 차원에 대한 pe 계산

    sinusoid_table = torch.FloatTensor(sinusoid_table).to(device)
    sinusoid_table[0] = 0.

    return sinusoid_table

def create_source_mask(source):
    #인코더의 자체 주의를 위한 마스킹 텐서 생성

    source_length = source.length

    source_mask = (source == pad_idx)# pad가 있으면 마스크 처리

    source_mask = source_mask.unsqueeze(1).repeat(1,source_length,1)
    return source_mask

def create_position_vector(sentencce):
    #위치 정보를 포함하는 위치 벡터 생성
    #패드 인덱스에 0번째 위치가 사용

    batch_size = sentencce.size()
    pos_vec = np.array([(pos+1) if word != pad_idx else 0 for row in range(batch_size) for pos,word in enumerate (sentencce[row])])  # pad를 0으로 나머지 1씩 미룸
    pos_vec - pos_vec.reshape(batch_size,-1)
    pos_vec = torch.LongTensor(pos_vec).to(device)
    return pos_vec

def create_subsequent_mask(target):
    
    """
    if target length is 5 and diagonal is 1, this function returns
        [[0, 1, 1, 1, 1],
         [0, 0, 1, 1, 1],
         [0, 0, 0, 1, 1],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0]]
    :param target: [batch size, target length]
    :return:
    """
    
    batch_size, target_length = target.size() #batch,target_length 지정

    subsequent_mask = torch.triu(torch.ones(target_length,target_length),diagonal=1).bool().to(device) # 삼각 행렬 생성
    # Torch.triu는 사용자 정의 대각선을 기준으로 행렬의 위쪽 삼각형 부분을 반환

    subsequent_mask = subsequent_mask.unsqueeze(0).repeat(batch_size,1,1)
    # subsequent_mask 'batch size'를 반복하여 배치의 모든 데이터 인스턴스를 포함

    return subsequent_mask

def create_target_mask(source,target):
    """
    인코더 출력에 대한 디코더의 자체 주의 및 디코더의 주의를 위한 마스킹 텐서 생성
    if sentence is [2, 193, 9, 27, 10003, 1, 1, 1, 3] and 2 denotes <sos>, 3 denotes <eos> and 1 denotes <pad>
    masking tensor will be [False, False, False, False, False, True, True, True, False]
    :param source: [batch size, source length]
    :param target: [batch size, target length]
    :return:
    """
    target_length = target.shape[1]

    subsequent_mask = create_subsequent_mask(target) #삼각 행렬 target 생성

    source_mask = (source == pad_idx) # source에서 pad
    target_mask = (target == pad_idx) # target에서 pad

    dec_enc_mask = source_mask.unsqueeze(1).repeat(1,target_length,1) # source
    target_mask = target_mask.unsqueeze(1).repeat(1,target_length,1) # target mask

    target_mask = target_mask | subsequent_mask

    return target_mask, dec_enc_mask