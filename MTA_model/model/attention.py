import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.ops import init_weight

class MultiHeadAttention(nn.Module):
    def __init__(self,params):
        super(MultiHeadAttention,self).__init__()
        assert params.hidden_dim % params.n_head == 0 # hidden_dimmension이 n_head로 안 떨어지면 에러 발생
        self.attentions = nn.Modulist([SelfAttention(params) for _ in range(params.n_head)])
        self.o_w = nn.Linear(params.hidden_dim,params.hidden_dim,bias=False)
        init_weight(self.o_w)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self,query,key,value, mask=None):

        self_attention = [attention(query,key,value) for attention in self.attentions]  # modulelist에 attetin 수 만큼 -> n_head
        weighted_vs = [weighted_v[0] for weighted_v in self_attention] # weight_vs
        attentions = [weighted_v[1] for weighted_v in self_attention]# attention_score

        weighted_v = torch.cat(weighted_vs,dim=-1) # weight_vs끼리 결합
        output = self.dropout(self.o_w(weighted_v))

        return output, attentions


class SelfAttention(nn.Module):
    def __init__(self,params):
        super(SelfAttention, self).__init__()
        self.hidden_dim = params.hidden_dim
        self.attention_dim = params.hidden_dim // params.n_head # attention 차원 생성 hidden_dim을 n_head로 나눈 나머지?를 어텐션 차원으로?

        self.q_w = nn.linear(self.hidden_dim,self.attention_dim,bias=False) # 쿼리 생성
        self.k_w = nn.linear(self.hidden_dim,self.attention_dim,bias=False) # 키 생성
        self.v_w = nn.linear(self.hidden_dim,self.attention_dim,bias=False) # 벨류 생성
        init_weight(self.q_w) # 쿼리 초기화
        init_weight(self.k_w) # 키 초기화
        init_weight(self.v_w) # 벨류 초기화

        self.dropout = nn.Dropout(params.dropout)
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.attention_dim])).to(params.device) # 루트 d
    
    def forward(self, query,key,value,mask=None):
        # query, key, value = [batch size, sentence length, hidden dim]
        # 동일한 입력 문장을 사용하여 Q, K, V 행렬을 만들어 자기 주의 점수를 계산

        q = self.q_w(query)
        k = self.k_w(key)
        v = self.v_w(value)
         # q, k, v = [batch size, sentence length, attention dim]

        self_attention = torch.bmm(q,k.permute(0,2,1)) #q*k k -> permute를 통해 transpose torch.bmm 배치 행렬곱
        self_attention = self_attention / self.scale_factor

        if mask is not None:# mask가 있으면
            self_attention = self_attention.masked_fill(mask, -np.inf) #mask -inf값 생성
        
        # 각 행에 소프트 최대 함수를 적용하여 자체 주의 점수를 정규화합니다
        attention_score = F.softmax(self_attention,dim=-1)
        norm_attention_score = self.dropout(attention_score)

        weighted_v = torch.bmm(norm_attention_score,v)

        return self.dropout(weighted_v), attention_score
        
