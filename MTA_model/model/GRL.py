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
        self.fc0 = nn.Linear(25600, 2560) # representation 크기 따라 변경
        self.fc1 = nn.Linear(2560, 256) 
        self.fc2 = nn.Linear(256, 100)
        self.fc3 = nn.Linear(100, 13)
    def forward(self, x):
        input_dim = x.size(1) * x.size(2)  # 입력 크기 (2560) 계산
        x = x.view(-1, input_dim)
        x = GradReverse.apply(x) # gradient revers
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

class gender_classifier(nn.Module):
    def __init__(self):
        super(gender_classifier, self).__init__()
        self.fc0 = nn.Linear(25600, 2560) # representation 크기 따라 변경
        self.fc1 = nn.Linear(2560, 256) 
        self.fc2 = nn.Linear(256, 100)
        self.fc3 = nn.Linear(100, 1)
    def forward(self, x):
        input_dim = x.size(1) * x.size(2)  # 입력 크기 (2560) 계산
        x = x.view(-1, input_dim)
        x = GradReverse.apply(x) # gradient revers
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 1) # [batch_size, 1] (예: [128, 1])
        return torch.sigmoid(x)

class age_classifier(nn.Module):
    def __init__(self):
        super(age_classifier, self).__init__()
        self.fc0 = nn.Linear(25600, 2560) # representation 크기 따라 변경
        self.fc1 = nn.Linear(2560, 256) 
        self.fc2 = nn.Linear(256, 100)
        self.fc3 = nn.Linear(100, 7)
    def forward(self, x):
        input_dim = x.size(1) * x.size(2)  # 입력 크기 (2560) 계산
        x = x.view(-1, input_dim)
        x = GradReverse.apply(x) # gradient revers
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

class pvalue_classifier(nn.Module):
    def __init__(self):
        super(pvalue_classifier, self).__init__()
        self.fc0 = nn.Linear(25600, 2560) # representation 크기 따라 변경
        self.fc1 = nn.Linear(2560, 256) 
        self.fc2 = nn.Linear(256, 100)
        self.fc3 = nn.Linear(100, 4)
    def forward(self, x):
        input_dim = x.size(1) * x.size(2)  # 입력 크기 (2560) 계산
        x = x.view(-1, input_dim)
        x = GradReverse.apply(x) # gradient revers
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

class shopping_classifier(nn.Module):
    def __init__(self):
        super(shopping_classifier, self).__init__()
        self.fc0 = nn.Linear(25600, 2560) # representation 크기 따라 변경
        self.fc1 = nn.Linear(2560, 256) 
        self.fc2 = nn.Linear(256, 100)
        self.fc3 = nn.Linear(100, 3)
    def forward(self, x):
        input_dim = x.size(1) * x.size(2)  # 입력 크기 (2560) 계산
        x = x.view(-1, input_dim)
        x = GradReverse.apply(x) # gradient revers
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ConversionClassifier(nn.Module):
    def __init__(self):
        super(ConversionClassifier, self).__init__()
        self.fc0 = nn.Linear(2560, 100)
        self.fc1 = nn.Linear(100, 25)
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        input_dim = x.size(1) * x.size(2)  # 입력 크기 (2560) 계산
        x = x.view(-1, input_dim)
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 1) # [batch_size, 1] (예: [128, 1])
        return torch.sigmoid(x)

# https://jimmy-ai.tistory.com/365