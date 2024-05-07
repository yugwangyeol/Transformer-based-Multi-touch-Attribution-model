import torch.nn as nn

class MTA_Loss(nn.Module):
    def __init__(self):
        super(MTA_Loss, self).__init__()
        self.domain_criterion = nn.BCELoss() # segmment loss
        self.label_criterion = nn.BCELoss() # conversion loss
    
    def forward(self, domain_output, label_output, domain_label, label_label):
        domain_loss = self.domain_criterion(domain_output, domain_label)
        label_loss = self.label_criterion(label_output, label_label)
        return domain_loss + label_loss

