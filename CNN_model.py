import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter



class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3, 32, 5,stride=1,padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.AvgPool2d(2,stride=2),

            nn.Conv2d(32, 64, 5,stride=1,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.AvgPool2d(2,stride=2),

            nn.Conv2d(64, 128, 5,stride=1,padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.AvgPool2d(2,stride=2),

            nn.Conv2d(128, 256, 5, stride=1, padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),

            nn.Conv2d(256, 512, 5, stride=1, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Flatten(),

            nn.Dropout(0.5),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
    def forward(self, input):
        output = self.model(input)
        output=output.to(torch.float32)
        return output

if __name__=='__main__':
    cnn=CNN()
    input=torch.ones(25,3,128,128)
    output=cnn(input)
    writer=SummaryWriter("graph")
    writer.add_graph(cnn,input)
    writer.close()
    print(output.shape)
