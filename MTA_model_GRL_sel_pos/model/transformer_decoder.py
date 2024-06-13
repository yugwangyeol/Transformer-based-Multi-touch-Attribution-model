import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder
from model.GRL import gender_classifier, age_classifier, ConversionClassifier

class Transformer_decoder(nn.Module):
    def __init__(self, params):
        super(Transformer_decoder, self).__init__()
        self.decoder = Decoder(params)
        self.gender_classifier = gender_classifier()
        self.age_classifier = age_classifier()
        self.conversion_classifier = ConversionClassifier()
    
    def forward(self,cam_sequential,cate_sequential,price_sequential,segment,encoder_output):

        gender_output = self.gender_classifier(encoder_output)
        age_output = self.age_classifier(encoder_output)

        output, attn_map = self.decoder(segment,cam_sequential,encoder_output)
        conversion_output = self.conversion_classifier(output)

        return  gender_output, age_output, conversion_output, attn_map #
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)