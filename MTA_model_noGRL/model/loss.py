import torch.nn as nn

class MTA_Loss(nn.Module):
    def __init__(self):
        super(MTA_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss() # 다중 분류용
        self.BCE = nn.BCELoss() # logit 분류용
    
    def forward(self, conversion_output, conversion_label): 
        # cms_output, cms_label, gender_output, gender_label, age_output, age_label,
        # pvalue_output, pvalue_label, shopping_output, shopping_label, 
        #cms_label = cms_label.long()
        #age_label = age_label.long()
        #pvalue_label = pvalue_label.long()
        #shopping_label = shopping_label.long()
        
        conversion_output = conversion_output.float()
        conversion_label = conversion_label.float()

        conversion_loss = self.BCE(conversion_output, conversion_label)
        return conversion_loss 