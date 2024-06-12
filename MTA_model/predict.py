import pickle
import argparse
import os
import numpy as np
import json 

import torch
import pandas as pd

from utils import CustomDataset, Params, display_attention, single_display_attention, load_dataset
from model.transformer_decoder import Transformer_decoder
from model.transformer_encoder import Transformer_encoder


def load_predict(input_uid):

    data = cam_sequential_data
    data['cate_sequential'] = cate_sequential_data['seq_space_sep']
    data['price_sequential'] = price_sequential_data['seq_space_sep']
    
    # input_uid sequence, segment
    user_sequence = data[data['user_id'] == input_uid]
    if user_sequence['num_user'].nunique != 1: # 동일 uid에 대해 여러 세션이 있을 경우 가장 첫 session 만 사용 -> 0번으로 안찍히는 경우 존재해서 수정
        first_num_user = user_sequence['num_user'].iloc[0]
        user_sequence = user_sequence[user_sequence['num_user'] == first_num_user]

    user_segment = segment_data[segment_data['user_id'] == input_uid]

    user_dataset = pd.merge(user_sequence, user_segment, on='user_id')
    user_dataset = user_dataset.drop(['user_id', 'num_user'], axis=1).iloc[:,1:]
    #print(user_dataset.columns)
    user_dataset.columns = ['cam_sequential', 'label', 'cate_sequential', 'price_sequential', 'cms',
                        'gender', 'age', 'pvalue', 'shopping', 'segment']

    seq_length = len(user_dataset['cam_sequential'][0].split())
    segment_list = ['cms: ' + str(user_dataset['cms'][0]),
                    'gender: ' + str(user_dataset['gender'][0]),
                    'age: ' + str(user_dataset['age'][0]),
                    'pvalue: ' + str(user_dataset['pvalue'][0]),
                    'shopping: ' + str(user_dataset['shopping'][0])]

    single_user_list = [] # single_user_list structure : 5 segment values + k sequence values + 5*k attention scores 

    # append 5 segment values 
    single_user_list.append(str(user_dataset['cms'][0]))
    single_user_list.append(str(user_dataset['gender'][0]))
    single_user_list.append(str(user_dataset['age'][0]))
    single_user_list.append(str(int(user_dataset['pvalue'][0])))
    single_user_list.append(str(user_dataset['shopping'][0]))
    # print('single_user_list len :',len(single_user_list))

    # append k sequence values 
    seq_lst = []
    for i in user_dataset['cam_sequential'][0].split():
        seq_lst.append(i)
    
    single_user_list.append(seq_lst)
    # print('single_user_list len :',len(single_user_list))

    return CustomDataset(user_dataset, 42), seq_length, segment_list , single_user_list


def predict(input_uid):
    params = Params('config/params.json')
    data_path = '../../Data2' 
    uid_dataset, seq_length, segment_list, *rest = load_predict(input_uid)
    single_user_list = rest[0]

     # vocab.pkl 파일 로드
    with open('../../Data2/vocab.pkl', 'rb') as f:
        vocab_info = pickle.load(f)
    
    # params에 최대 인덱스 값을 로드
    params.load_vocab(vocab_info['cam_input_dim'], vocab_info['cate_input_dim'], vocab_info['price_input_dim'])
    ### --------------- ###

    print('데이터 전처리 완료!')

    encoder = Transformer_encoder(params)
    decoder = Transformer_decoder(params)
    encoder.load_state_dict(torch.load('./model_pt/encoder.pt'))
    decoder.load_state_dict(torch.load('./model_pt/decoder.pt'))
    print('load pretrained weight')
    encoder.to(params.device)
    decoder.to(params.device)
    print('model to device')
    encoder.eval()
    decoder.eval()

    print('모델 호출 완료!')

    with torch.no_grad():
        cam_sequential = uid_dataset.cam_sequential.to(params.device)
        cate_sequential = uid_dataset.cate_sequential.to(params.device)
        price_sequential = uid_dataset.price_sequential.to(params.device)
        segment = uid_dataset.segment.to(params.device)

        encoder_output = encoder(cam_sequential, cate_sequential, price_sequential)
        print(encoder_output.shape)
        cms_output, gender_output, age_output, pvalue_output, shopping_output, conversion_output, attn_map = decoder(
            cam_sequential, cate_sequential, price_sequential, segment, encoder_output)

        cms_pred = cms_output.squeeze(0).max()
        gender_pred = gender_output.squeeze(0).max()
        age_pred = age_output.squeeze(0).max()
        pvalue_pred = pvalue_output.squeeze(0).max()
        shopping_pred = shopping_output.squeeze(0).max()
        conv_pred = conversion_output.squeeze(0).max()

        print(f'user id> {input_uid}')
        print(f'predicted conversion> {conv_pred}')

        print('len : ', len(attn_map)) # 8
        print('origin attn_map shape : ', attn_map[0].shape) # torch.size([1, 5, 50])

        stack_attn_map = torch.stack(attn_map)
        final_attn_map = torch.mean(stack_attn_map, dim=0).squeeze(0)
        single_attn_map = torch.sum(final_attn_map, dim=0).squeeze(0)
        single_attn_map = single_attn_map.unsqueeze(0)

        print('final_attn_map shape: ', final_attn_map.shape)
        print('single_attn_map shape: ', single_attn_map.shape)
        
        # Attention map 저장 - txt file
        vis_attn = final_attn_map[:, :seq_length] #padding 부분 제거 

        # append 5 * k values 
        # 연산량 감소를 위한 반올림 
        vis_attn_list = vis_attn.cpu().numpy().tolist()
        rounded_vis_attn_list = list(map(lambda sublist: list(map(lambda x: round(x, 5), sublist)), vis_attn_list))
        for i in rounded_vis_attn_list:
            single_user_list.append(i)

        # print('single_user_list : ',single_user_list)
        # print('single_user_list len :',len(single_user_list))
        attn_value_lst.append(single_user_list)

        # save_path = os.path.join(data_path, f'attention_map_{input_uid}.txt')
        # np.savetxt(save_path, vis_attn.cpu().numpy())
        # print(f'Attention map saved to {save_path}')

        # Attention map 시각화 (whole seg , single seg)
        # display_attention(source, target, attention, data_path, idx)
        # display_attention(cam_sequential, segment_list, final_attn_map, data_path, seq_length)
        # single_display_attention(cam_sequential, segment_list, single_attn_map, data_path, seq_length)


if __name__ == '__main__':

    # global reading 
    data_path = '../../Data2' 

    cam_sequential_path = os.path.join(data_path, 'camp_test.csv')
    cate_sequential_path = os.path.join(data_path, 'cate_test.csv')
    price_sequential_path = os.path.join(data_path, 'price_test.csv')
    segment_path = os.path.join(data_path, 'test_seg.csv')

    cam_sequential_data = pd.read_csv(cam_sequential_path, encoding='utf-8')
    cate_sequential_data = pd.read_csv(cate_sequential_path, encoding='utf-8')
    price_sequential_data = pd.read_csv(price_sequential_path, encoding='utf-8')
    segment_data = pd.read_csv(segment_path, encoding='utf-8').iloc[:, 1:]

    segment_data.columns = ['user_id', "cms_group_id", "gender", "age_level", "pvalue_level", "shopping_level", 'segment']

    # 전체 Attention map 을 저장할 리스트 선언
    attn_value_lst = []

    # userid를 넣은 반복문  
    for input_uid in segment_data.head(3)['user_id']: #일시적으로 확인을 위해 수정 
        predict(input_uid)

    with open('../../Data2/attn_value_lst.json', 'w') as file:
        json.dump(attn_value_lst, file, indent=4)

    print('done')