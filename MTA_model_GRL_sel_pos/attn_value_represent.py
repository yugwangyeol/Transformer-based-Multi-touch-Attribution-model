import json 
import pandas as pd 
import numpy as np


def base_attn_df():
    """""""""""""모든 Attn values 를 넣을 데이터 프레임 생성 """""""""""""

    ## ROW (User Segment) - segment_lst 
    segment_lst = []
    # CMS: 0 ~ 12
    segment_lst.extend([f'cms_{i}' for i in range(13)])
    # Gender: 0, 1
    segment_lst.extend([f'gender_{i}' for i in range(2)])
    # Age_level: 0 ~ 6
    segment_lst.extend([f'age_{i}' for i in range(7)])
    # pvalue_level: 0 ~ 3
    segment_lst.extend([f'pvalue_{i}' for i in range(4)])
    # shopping_level: 0 ~ 2
    segment_lst.extend([f'shopping_{i}' for i in range(3)])

    ## Columns (Campaign Sequence) - campaign_valid_1000 
    with open('../../Data2/campaign_valid_1000.json', 'r') as file:
        campaign_valid_1000 = json.load(file)
    campaign_valid_1000.insert(0,'index')

    attn_df = pd.DataFrame(np.zeros((len(segment_lst), len(campaign_valid_1000))), columns=campaign_valid_1000)
    attn_df['index'] = pd.Series(segment_lst)
    attn_df = attn_df.set_index('index')
    attn_df = attn_df.applymap(lambda x: np.array([0, 0],dtype=float) if x == 0 else x)
    print(attn_df)
    return attn_df

def update_test_df(df, lst, segment_index, prefix, value_index):
    for i in lst:
        segment_idx = i[segment_index]
        for k, j in enumerate(i[5]):
            column_idx = j
            array_value = df.at[f'{prefix}_{segment_idx}', column_idx].copy()
            array_value[0] += i[value_index][k]
            array_value[1] += 1
            df.at[f'{prefix}_{segment_idx}', column_idx] = array_value


if __name__ == '__main__':
    """"""""""""" attn_value_lst index description """""""""""""
    # [:5] -> segments 
    # [5]  -> list type sequence info 
    # [6:] -> list types attention scores 

    with open('/home/work/nas_code_modify/2024_Capstone/to_pang10_3/attn_value_lst.json', 'r') as file:
        attn_value_lst = json.load(file)

    base_attn_df = base_attn_df()

    update_test_df(base_attn_df, attn_value_lst, 0, 'cms', 6)
    update_test_df(base_attn_df, attn_value_lst, 1, 'gender', 7)
    update_test_df(base_attn_df, attn_value_lst, 2, 'age', 8)
    update_test_df(base_attn_df, attn_value_lst, 3, 'pvalue', 9)
    update_test_df(base_attn_df, attn_value_lst, 4, 'shopping', 10)

    print(base_attn_df)
    

        





    