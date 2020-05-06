import torch.nn as nn



class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1,16,5,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(33 * 58 * 32, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 168 * 3),
        )



    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = out.view(-1, 33 * 58 * 32)
        out = self.fc(out)
        out = out.view(-1, 168, 3)
        return out