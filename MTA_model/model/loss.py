import torch.nn as nn

class MTA_Loss(nn.Module):
    def __init__(self):
        super(MTA_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss() # 다중 분류용
        self.BCE = nn.BCELoss() # logit 분류용
    
    def forward(self, cms_output,cms_label,gender_output, gender_label, age_output, age_label,
                pvalue_output, pvalue_label, shopping_output, shopping_label,conversion_output,conversion_label):
        conversion_loss = self.BCE(conversion_output, conversion_label)
        cms_loss = self.CE(cms_output, cms_label)
        gender_loss = self.BCE(gender_output,gender_label)
        age_loss = self.CE(age_output, age_label)
        pvalue_loss = self.CE(pvalue_output, pvalue_label)
        shopping_loss = self.CE(shopping_output, shopping_label)
        return conversion_loss + cms_loss + gender_loss + age_loss + pvalue_loss + shopping_loss

