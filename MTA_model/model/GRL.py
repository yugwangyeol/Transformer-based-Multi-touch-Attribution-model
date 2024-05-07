import torch.nn as nn
import torch.nn.functional as F
import torch

class GradReverse(torch.autograd.Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output): # 역전파 시에 gradient에 음수를 취함
        return (grad_output * -1)

class domain_classifier(nn.Module):
    def __init__(self):
        super(domain_classifier, self).__init__()
        self.fc1 = nn.Linear(100, 10)
        self.fc2 = nn.Linear(10, 1) # segment 별로 생성해야 하나?
    def forward(self, x):
        x = GradReverse.apply(x) # gradient reverse
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

class label_classifier(nn.Module):
    def __init__(self):
        super(label_classifier, self).__init__()
        self.fc1 = nn.Linear(100, 25)
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

# https://jimmy-ai.tistory.com/365