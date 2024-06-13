import torch.nn as nn

class MTA_Loss(nn.Module):
    def __init__(self):
        super(MTA_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss() # 다중 분류용
        self.BCE = nn.BCELoss() # logit 분류용
    
    def forward(self, gender_output, gender_label, age_output, age_label, conversion_output, conversion_label): #
        age_label = age_label.long()

        conversion_output = conversion_output.float()
        conversion_label = conversion_label.float()
        gender_output = gender_output.float()
        gender_label = gender_label.float()

        #print(conversion_label)
        #print(conversion_output)

        conversion_loss = self.BCE(conversion_output, conversion_label)
        gender_loss = self.BCE(gender_output.squeeze(1), gender_label)
        age_loss = self.CE(age_output, age_label)
        return (conversion_loss + gender_loss + age_loss ), conversion_loss 