import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder
from model.GRL import cms_classifier, gender_classifier, age_classifier, pvalue_classifier, shopping_classifier, ConversionClassifier

class Transformer_decoder(nn.Module):
    def __init__(self, params):
        super(Transformer_decoder, self).__init__()
        self.decoder = Decoder(params)
        #self.cms_classifier = cms_classifier()
        #self.gender_classifier = gender_classifier()
        #self.age_classifier = age_classifier()
        #self.pvalue_classifier = pvalue_classifier()
        #self.shopping_classifier = shopping_classifier()
        self.conversion_classifier = ConversionClassifier()
    
    def forward(self,cam_sequential,cate_sequential,price_sequential,segment,encoder_output):

        #cms_output = self.cms_classifier(encoder_output)
        #gender_output = self.gender_classifier(encoder_output)
        #age_output = self.age_classifier(encoder_output)
        #pvalue_output = self.pvalue_classifier(encoder_output)
        #shopping_output = self.shopping_classifier(encoder_output)

        output, attn_map = self.decoder(segment,cam_sequential,encoder_output)
        conversion_output = self.conversion_classifier(output)

        return conversion_output, attn_map # cms_output, gender_output, age_output, pvalue_output, shopping_output, 
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)