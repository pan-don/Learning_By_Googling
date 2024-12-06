import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_dim=224, input_c=3, output=6, hidden_dim=128, dropout=0.5):
        super(CNN, self).__init__()

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=input_c, out_channels=16, kernel_size=(5,5), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=(3, 3))
        )
        
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=(3, 3))
        )
        
        self.convBlock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=(3, 3))
        )
        
        self.convBlock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=(3, 3))
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128*1*1, out_features=output)
        )
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = self.convBlock4(x)
        x = self.fc(torch.flatten(x, 1))
        x = self.softmax(x)
        return x

# dummy = torch.randn(16,3,224,224)
# model = CNN()
# output = model(dummy)
# print('Output:', output.shape)
