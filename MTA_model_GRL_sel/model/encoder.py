import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward
from model.ops import create_positional_encoding, create_source_mask, create_position_vector

import torch

class EncoderLayer(nn.Module):
    def __init__(self,params):
        super(EncoderLayer,self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim,eps=1e-6) # Layer norm 설정
        self.self_attention = MultiHeadAttention(params)
        self.postion_wise_ffn = PositionWiseFeedForward(params)

    def forward(self,source,source_mask):
        # source          = [batch size, source length, hidden dim]
        # source_mask     = [batch size, source length, source length]

        # 원래 구현: LayerNorm(x + SubLayer(x)) -> 업데이트된 구현: x + SubLayer(LayerNorm(x))
        normalized_source = self.layer_norm(source)
        output = source + self.self_attention(normalized_source,normalized_source,normalized_source,source_mask)[0] # attention + residual 

        normalized_output = self.layer_norm(output)
        #print(self.postion_wise_ffn(normalized_output).shape)
        #print(output.shape)
        output = output + self.postion_wise_ffn(normalized_output)
        # output = [batch size, source length, hidden dim]

        return output

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.cam_token_embedding = nn.Embedding(int(params.cam_input_dim), params.hidden_dim, padding_idx=params.pad_idx) # embedding 설정
        self.cate_token_embedding = nn.Embedding(int(params.cate_input_dim), params.hidden_dim, padding_idx=params.pad_idx)
        self.price_token_embedding = nn.Embedding(int(params.price_input_dim), params.hidden_dim, padding_idx=params.pad_idx)

        # Embedding 가중치 초기화
        self._init_weights(self.cam_token_embedding, params.hidden_dim)
        self._init_weights(self.cate_token_embedding, params.hidden_dim)
        self._init_weights(self.price_token_embedding, params.hidden_dim)

        self.embedding_scale = params.hidden_dim ** 0.5  # embedding_scale 생성 -> 규제항

        #self.pos_embedding = nn.Embedding.from_pretrained(create_positional_encoding(params.max_len + 1, params.hidden_dim), freeze=True) # positional encoding 생성

        self.encoder_layer = nn.ModuleList([EncoderLayer(params) for _ in range(params.n_layer)])
        self.dropout = nn.Dropout(params.dropout)
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)
        self.pad_idx = params.pad_idx
        self.device = params.device

    def _init_weights(self, embedding, hidden_dim):
        """
        임베딩 가중치 초기화 함수
        """
        # 임베딩 가중치를 정규 분포로 초기화
        with torch.no_grad():
            embedding.weight.normal_(mean=0, std=hidden_dim ** -0.5)

    def forward(self, cam_sequential, cate_sequential, price_sequential):
        # 입력 텐서를 LongTensor로 변환
        cam_sequential = cam_sequential.long()
        cate_sequential = cate_sequential.long()
        price_sequential = price_sequential.long()

        source_mask = create_source_mask(cam_sequential, self.pad_idx)  # pad 마스크 처리
        #print(source_mask)
        #source_pos = create_position_vector(cam_sequential, self.pad_idx, self.device)  # position 벡터 생성

        source = (self.cam_token_embedding(cam_sequential) + self.cate_token_embedding(cate_sequential) + 
                  self.price_token_embedding(price_sequential)) * self.embedding_scale  # embedding 생성    
        #source = self.dropout(source + self.pos_embedding(source_pos))  # source 생성 embedding  + position 
        source = self.dropout(source)

        for encoder_layer in self.encoder_layer:
            source = encoder_layer(source, source_mask)  # layer 만큼 진행
        
        return self.layer_norm(source)