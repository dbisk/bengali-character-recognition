# import built-in modules
# import 3rd-party modules
import numpy as np
import torch.nn as nn
import torchvision

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x

class PretrainedResnet(nn.Module):

    def __init__(self, total_roots, total_vowels, total_cons):
        super(PretrainedResnet, self).__init__()

        self.resnet = torchvision.models.resnet18(pretrained=True)
        res_features = self.resnet.fc.in_features
        self.resnet.fc = Identity()
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.fc_root = nn.Linear(res_features, total_roots)
        self.fc_vowel = nn.Linear(res_features, total_vowels)
        self.fc_cons = nn.Linear(res_features, total_cons)

    def forward(self, x):
        # x is expected to be 3x224x224 in original resnet18
        x = self.resnet(x)
        root = self.fc_root(x)
        vowel = self.fc_vowel(x)
        cons = self.fc_cons(x)
        return root, vowel, cons
