import torch.nn as nn
import torch


class ResBlock(nn.Module):
    def __init__(self, planes):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(planes, planes, kernel_size=3, padding=0),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(planes, planes, kernel_size=3, padding=0),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        x = x + self.block(x)

        return x


# ResNet
class Generator(nn.Module):
    def __init__(self, n_block=6):
        super(Generator, self).__init__()

        self.n_block = n_block

        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.downsampling = nn.Sequential(
            # downsampling
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        b = []
        for _ in range(n_block):
            b.append(ResBlock(256))

        self.block = nn.Sequential(*b)

        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 1, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, real, fake):
        x = torch.cat((real, fake), 1)

        x = self.layer1(x)
        x = self.downsampling(x)
        x = self.block(x)
        x = self.upsampling(x)
        x = self.output(x)

        return x


# GlobalGAN
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=2, padding=0),
            nn.LeakyReLU(0.2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        self.output = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=0),
            nn.Conv2d(1, 1, kernel_size=6, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, real, fake):
        x = torch.cat((real, fake), 1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)

        return x



'''
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = Discriminator().to(device)

summary(model, (3, 256, 256))
'''