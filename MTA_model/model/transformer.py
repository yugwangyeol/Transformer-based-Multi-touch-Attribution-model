import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder
from model.GRL import domain_classifier, label_classifier

class Transformer(nn.Module):
    def __init__(self, params):
        super(Transformer, self).__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.sex_classifier = domain_classifier()
    
    def forward(self,sequential,segment):

        encoder_output = self.encoder(sequential)
        sex_output = self.sex_classifier(encoder_output)
        output, attn_map = self.decoder(segment,sequential,encoder_output)

        return output, attn_map, sex_output
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)