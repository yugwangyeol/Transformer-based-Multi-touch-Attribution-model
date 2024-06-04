import torch.nn as nn
import torch.nn.functional as F
import torch

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class cms_classifier(nn.Module):
    def __init__(self):
        super(cms_classifier, self).__init__()
        self.fc0 = nn.Linear(5376, 1024) # representation 크기 따라 변경
        self.fc1 = nn.Linear(1024, 100) 
        self.fc2 = nn.Linear(100, 13)
    def forward(self, x):
        input_dim = x.size(1) * x.size(2)  # 입력 크기 (2560) 계산
        x = x.view(-1, input_dim)
        x = grad_reverse(x) # gradient revers
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

class gender_classifier(nn.Module):
    def __init__(self):
        super(gender_classifier, self).__init__()
        self.fc0 = nn.Linear(5376, 1024) # representation 크기 따라 변경
        self.fc1 = nn.Linear(1024, 100) 
        self.fc2 = nn.Linear(100, 1)
    def forward(self, x):
        input_dim = x.size(1) * x.size(2)  # 입력 크기 (2560) 계산
        x = x.view(-1, input_dim)
        x = grad_reverse(x)  # gradient revers
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 1)
        return torch.sigmoid(x)

class age_classifier(nn.Module):
    def __init__(self):
        super(age_classifier, self).__init__()
        self.fc0 = nn.Linear(5376, 1024) # representation 크기 따라 변경
        self.fc1 = nn.Linear(1024, 100) 
        self.fc2 = nn.Linear(100, 7)
    def forward(self, x):
        input_dim = x.size(1) * x.size(2)  # 입력 크기 (2560) 계산
        x = x.view(-1, input_dim)
        x = grad_reverse(x)  # gradient revers
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

class pvalue_classifier(nn.Module):
    def __init__(self):
        super(pvalue_classifier, self).__init__()
        self.fc0 = nn.Linear(5376, 1024) # representation 크기 따라 변경
        self.fc1 = nn.Linear(1024, 100) 
        self.fc2 = nn.Linear(100, 4)
    def forward(self, x):
        input_dim = x.size(1) * x.size(2)  # 입력 크기 (2560) 계산
        x = x.view(-1, input_dim)
        x = grad_reverse(x)  # gradient revers
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

class shopping_classifier(nn.Module):
    def __init__(self):
        super(shopping_classifier, self).__init__()
        self.fc0 = nn.Linear(5376, 1024) # representation 크기 따라 변경
        self.fc1 = nn.Linear(1024, 100) 
        self.fc2 = nn.Linear(100, 3)
    def forward(self, x):
        input_dim = x.size(1) * x.size(2)  # 입력 크기 (2560) 계산
        x = x.view(-1, input_dim)
        x = grad_reverse(x)  # gradient revers
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ConversionClassifier(nn.Module):
    def __init__(self):
        super(ConversionClassifier, self).__init__()
        self.fc0 = nn.Linear(2560, 1024)
        self.fc1 = nn.Linear(1024, 100)
        self.fc2 = nn.Linear(100, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        input_dim = x.size(1) * x.size(2)  # 입력 크기 (2560) 계산
        x = x.view(-1, input_dim)
        x = F.leaky_relu(self.fc0(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 1) # [batch_size, 1] (예: [128, 1])
        return torch.sigmoid(x)

# https://jimmy-ai.tistory.com/365