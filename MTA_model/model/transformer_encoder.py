import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder
from model.GRL import cms_classifier, gender_classifier, age_classifier, pvalue_classifier, shopping_classifier, ConversionClassifier
class Transformer_encoder(nn.Module):
    def __init__(self, params):
        super(Transformer_encoder, self).__init__()
        self.encoder = Encoder(params)
    
    def forward(self,cam_sequential,cate_sequential,price_sequential,segment):

        encoder_output = self.encoder(cam_sequential,cate_sequential,price_sequential)
        return  encoder_output
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)