import pickle
import argparse
import os

import torch
import pandas as pd

from utils import CustomDataset, Params, display_attention, load_dataset
from model.transformer import Transformer

def load_predict(input_uid, data_dir):
    '''
    --input input_uid 입력하면 해당 uid에 대한 sequence & segment 호출
    '''

    cam_sequential_path = os.path.join(data_dir, 'campaign_id_test_seq.csv')
    cate_sequential_path = os.path.join(data_dir, 'cate_id_test_seq.csv')
    price_sequential_path = os.path.join(data_dir, 'price_test_seq.csv')
    segment_path = os.path.join(data_dir, 'test_segment.csv')

    cam_sequential_data = pd.read_csv(cam_sequential_path, encoding='utf-8')
    cate_sequential_data = pd.read_csv(cate_sequential_path, encoding='utf-8')
    price_sequential_data = pd.read_csv(price_sequential_path, encoding='utf-8')

    segment_data = pd.read_csv(segment_path, encoding='utf-8').iloc[:, 1:]
    segment_data.columns = ['user_id', "cms_group_id", "gender", "age_level", "pvalue_level", "shopping_level", 'segment']

    data = cam_sequential_data
    data['cate_sequential'] = cate_sequential_data['seq_space_sep']
    data['price_sequential'] = price_sequential_data['seq_space_sep']

    # input_uid sequence, segment
    user_sequence = data[data['user_id'] == input_uid]
    if user_sequence['num_user'].nunique != 1: # 동일 uid에 대해 여러 세션이 있을 경우 0번째 session만 사용
        user_sequence = user_sequence[user_sequence['num_user'] == 0]

    user_segment = segment_data[segment_data['user_id'] == input_uid]

    user_dataset = pd.merge(user_sequence, user_segment, on='user_id')
    user_dataset = user_dataset.drop(['user_id', 'num_user'], axis=1)
    user_dataset.columns = ['cam_sequential', 'label', 'cate_sequential', 'price_sequential', 'cms',
                        'gender', 'age', 'pvalue', 'shopping', 'segment']

    seq_length = len(user_dataset['cam_sequential'][0].split())
    segment_list = ['cms: ' + str(user_dataset['cms'][0]),
                    'gender: ' + str(user_dataset['gender'][0]),
                    'age: ' + str(user_dataset['age'][0]),
                    'pvalue: ' + str(user_dataset['pvalue'][0]),
                    'shopping: ' + str(user_dataset['shopping'][0])]

    return CustomDataset(user_dataset, 50), seq_length, segment_list


def predict(config):
    input_uid = int(config.input)
    params = Params('config/params.json')
    data_path = '/home/work/2024_CAPSTONE_model'

    uid_dataset, seq_length, segment_list = load_predict(input_uid, data_path)

    ### 이 부분에 pickle 호출해서 embedding 하는 코드 추가해야함 ###
    ### 이하는 임시 코드 ###
    train_data, valid_data = load_dataset('train')
    cam_input_dim = max(train_data.cam_sequential.max().item(), valid_data.cam_sequential.max().item())
    cate_input_dim = max(train_data.cate_sequential.max().item(), valid_data.cate_sequential.max().item())
    price_input_dim = max(train_data.price_sequential.max().item(), valid_data.price_sequential.max().item())
    params.load_vocab(cam_input_dim+1, cate_input_dim+1, price_input_dim+1)
    ### --------------- ###

    print('데이터 전처리 완료!')

    model = Transformer(params)
    model.load_state_dict(torch.load('/home/work/2024_CAPSTONE_model/model.pt'))
    print('load pretrained weight')
    model.to(params.device)
    print('model to divice')
    model.eval()

    print('모델 호출 완료!')

    with torch.no_grad():
        cam_sequential = uid_dataset.cam_sequential.to(params.device)
        cate_sequential = uid_dataset.cate_sequential.to(params.device)
        price_sequential = uid_dataset.price_sequential.to(params.device)
        segment = uid_dataset.segment.to(params.device)

        cms_output, gender_output, age_output, pvalue_output, shopping_output, conv_output, attn_map = model(
            cam_sequential, cate_sequential, price_sequential, segment
        )

        cms_pred = cms_output.squeeze(0).max()
        gender_pred = gender_output.squeeze(0).max()
        age_pred = age_output.squeeze(0).max()
        pvalue_pred = pvalue_output.squeeze(0).max()
        shopping_pred = shopping_output.squeeze(0).max()
        conv_pred = conv_output.squeeze(0).max()

        print(f'user id> {input_uid}')
        print(f'predicted conversion> {conv_pred}')

        print('len : ', len(attn_map)) # 8
        print('origin attn_map shape : ', attn_map[0].shape) # torch.size([1, 5, 50])

        stack_attn_map = torch.stack(attn_map)
        final_attn_map = torch.mean(stack_attn_map, dim=0).squeeze(0)

        print('final_attn_map shape: ', final_attn_map.shape)
        display_attention(cam_sequential, segment_list, final_attn_map, data_path, seq_length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='User ID')
    parser.add_argument('--input', type=str, default='257026') 
    option = parser.parse_args()

    predict(option)