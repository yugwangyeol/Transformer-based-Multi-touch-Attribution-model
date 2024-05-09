import torch.nn as nn
import torch.nn.functional as F
import torch

class GradReverse(torch.autograd.Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output): # 역전파 시에 gradient에 음수를 취함
        return (grad_output * -1)

class cms_classifier(nn.Module):
    def __init__(self):
        super(cms_classifier, self).__init__()
        self.fc0 = nn.Linear(512, 100) # representation 크기 따라 변경
        self.fc1 = nn.Linear(100, 20) 
        self.fc2 = nn.Linear(20, 13)
    def forward(self, x):
        x = GradReverse.apply(x) # gradient revers
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

class gender_classifier(nn.Module):
    def __init__(self):
        super(gender_classifier, self).__init__()
        self.fc0 = nn.Linear(512, 100)
        self.fc1 = nn.Linear(100, 10) 
        self.fc2 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = GradReverse.apply(x)
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

class age_classifier(nn.Module):
    def __init__(self):
        super(age_classifier, self).__init__()
        self.fc0 = nn.Linear(512, 100)
        self.fc1 = nn.Linear(100, 25)
        self.fc2 = nn.Linear(25, 7)

    def forward(self, x):
        x = GradReverse.apply(x)
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

class pvalue_classifier(nn.Module):
    def __init__(self):
        super(pvalue_classifier, self).__init__()
        self.fc0 = nn.Linear(512, 100)
        self.fc1 = nn.Linear(100, 10)
        self.fc2 = nn.Linear(10, 4)

    def forward(self, x):
        x = GradReverse.apply(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

class shopping_classifier(nn.Module):
    def __init__(self):
        super(shopping_classifier, self).__init__()
        self.fc0 = nn.Linear(512, 100)
        self.fc1 = nn.Linear(100, 25) 
        self.fc2 = nn.Linear(25, 3)

    def forward(self, x):
        x = GradReverse.apply(x)
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

class conversion_classifier(nn.Module):
    def __init__(self):
        super(conversion_classifier, self).__init__()
        self.fc0 = nn.Linear(512, 100)
        self.fc1 = nn.Linear(100, 25)
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        self.fc0 = nn.Linear(512, 100)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

# https://jimmy-ai.tistory.com/365