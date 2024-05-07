import torch.nn as nn
import torch.nn.functional as F

from model.ops import init_weigth

class PositionWiseFeedForward(nn.Module): # 마지막에 늘렸다가 줄이는 부분
    def __init__(self,params):
        super(PositionWiseFeedForward, self).__init__()
        
        self.conv1 = nn.Conv1d(params.hidden_dim,params.feed_forward_dim,kernel_size=1)
        self.conv2 = nn.Conv1d(params.feed_forward_dim,params.hidden_dim,kernel_size=1)

        init_weigth(self.conv1)
        init_weigth(self.conv2)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self,x):

        #x = [batch size, sentence length, hidden dim]
        # nn을 적용할 permute x의 인덱스.입력 'x'의 Conv1d

        x = x.permute(0,2,1) # x = [batch size, hidden dim, sentence length] #구조 변경
        output = self.dropout(self.conv1(x))
        output - self.conv2(output)

        output = output.permute(0,2,1)
        return self.dropout(output)