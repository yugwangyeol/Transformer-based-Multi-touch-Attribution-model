import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward
from model.ops import create_positional_encoding, create_source_mask, create_position_vector

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
        output = output + self.postion_wise_ffn(normalized_output)
        # output = [batch size, source length, hidden dim]

        return output

class Encoder(nn.Module):
    def __init__(self,params):
        super(Encoder,self).__init__()
        self.token_embedding = nn.Embedding(params.input_dim, params.hidden_dim, padding_idx=params.pad_idx) #embedding 지정

        nn.init.normal_(self.token_embedding,mean=0,std=params.hidden_dim**-0.5)
        self.embedding_scale = params.hidden_dim**0.5  # embedding_scale 생성 -> 규제항

        self.pos_embedding = nn.Embedding.from_pretrained(create_positional_encoding(params.max_len+1,params.hidden_dim),freeze=True) #positional encoding 생성

        self.encoder_layer = nn.ModuleList([EncoderLayer(params) for _ in range(params.n_layer)])
        self.dropout = nn.Dropout(params.dropout)
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)
    
    def forward(self,source):

        source_mask = create_source_mask(source) # pad 마스크 처리
        source_pos = create_position_vector(source) #position 벡터 생성

        source = self.token_embedding(source)*self.embedding_scale
        source = self.dropout(source+self.pos_embedding(source_pos)) # source 생성 embedding  + position 

        for encoder_layer in self.encoder_layer:
            source = encoder_layer(source,source_mask) # later 만큼 진행
        
        return self.layer_norm(source)