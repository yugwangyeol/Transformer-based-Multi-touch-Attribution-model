import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder
from model.GRL import cms_classifier, gender_classifier, age_classifier, pvalue_classifier, shopping_classifier, conversion_classifier
class Transformer(nn.Module):
    def __init__(self, params):
        super(Transformer, self).__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.cms_classifier = cms_classifier()
        self.gender_classifier = gender_classifier()
        self.age_classifier = age_classifier()
        self.pvalue_classifier = pvalue_classifier()
        self.shopping_classifier = shopping_classifier()
        self.conversion_classifier = conversion_classifier()
    
    def forward(self,sequential,segment):

        encoder_output = self.encoder(sequential)
        cms_output = self.cms_classifier(encoder_output)
        gender_output = self.gender_classifier(encoder_output)
        age_output = self.age_classifier(encoder_output)
        pvalue_output = self.pvalue_classifier(encoder_output)
        shopping_output = self.shopping_classifier(encoder_output)
        output, attn_map = self.decoder(segment,sequential,encoder_output)
        conversion_output = self.conversion_classifier(encoder_output+output)

        return cms_output, gender_output, age_output, pvalue_output, shopping_output, conversion_output, attn_map
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)