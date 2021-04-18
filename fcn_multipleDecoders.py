import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class Down(nn.Module):
    def __init__(self, nfilters, doubleconv, i):
        super(Down, self).__init__()
        self.convs = nn.ModuleList()

        if i == 0:
            self.convs.append(nn.Conv2d(3, nfilters[i], (3, 3), padding=(1, 1)))
        else:
            self.convs.append(nn.Conv2d(nfilters[i - 1], nfilters[i], (3, 3), padding=(1, 1)))
        self.convs.append(nn.BatchNorm2d(nfilters[i]))
        if doubleconv:
            self.convs.append(nn.ReLU(inplace=True))
            self.convs.append(nn.Conv2d(nfilters[i], nfilters[i], (3, 3), padding=(1, 1)))
            self.convs.append(nn.BatchNorm2d(nfilters[i]))
            self.convs.append(nn.ReLU(inplace=True))
            self.convs.append(nn.Conv2d(nfilters[i], nfilters[i], (3, 3), padding=(1, 1)))
            self.convs.append(nn.BatchNorm2d(nfilters[i]))

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class LastDown(nn.Module):
    def __init__(self, nfilters, doubleconv):
        super(LastDown, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.doubleconv = doubleconv
        self.convs = nn.ModuleList()
        i = len(nfilters) - 1

        self.seq1 = nn.Sequential(
            nn.Conv2d(nfilters[i - 1], nfilters[i], (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(nfilters[i])
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(nfilters[i], nfilters[i], (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(nfilters[i])
        )
        self.seq3 = nn.Sequential(
            nn.Conv2d(nfilters[i], nfilters[i], (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(nfilters[i])
        )

        self.convs.append(nn.ReLU(inplace=True))
        if doubleconv:
            self.convs.append(nn.ReLU(inplace=True))
            self.convs.append(nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.seq1(x)
        y = self.relu(x)
        if self.doubleconv:
            x = self.seq2(y)
            y = self.relu(x)
            x = self.seq3(y)
            y = self.relu(x)
        return x, y


class Upsample(nn.Module):
    def __init__(self, nfilters, i):
        super(Upsample, self).__init__()
        self.conv = nn.ConvTranspose2d(
            nfilters[i + 1], nfilters[i +1], kernel_size=(2, 2), stride=(2, 2))
    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, nfilters, doubleconv, i):
        super(Up, self).__init__()
        self.convs = nn.ModuleList()

        self.convs.append(nn.Conv2d(nfilters[i]+nfilters[i+1], nfilters[i], kernel_size=(3, 3), padding=(1, 1)))
        self.convs.append(nn.BatchNorm2d(nfilters[i]))
        self.convs.append(nn.ReLU(inplace=True))
        if doubleconv:
            self.convs.append(nn.Conv2d(nfilters[i], nfilters[i], kernel_size=(3, 3), padding=(1, 1)))
            self.convs.append(nn.BatchNorm2d(nfilters[i]))
            self.convs.append(nn.ReLU(inplace=True))

            self.convs.append(nn.Conv2d(nfilters[i], nfilters[i], kernel_size=(3, 3), padding=(1, 1)))
            self.convs.append(nn.BatchNorm2d(nfilters[i]))
            self.convs.append(nn.ReLU(inplace=True))

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class CNN(nn.Module):
    def __init__(self, nfilters, nclass, doubleconv):
        super(CNN, self).__init__()
        self.nlayers = len(nfilters)
        self.nclass = nclass
        self.downs = nn.ModuleList()
        self.last_down = LastDown(nfilters, doubleconv)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for i in range(0, self.nlayers - 1):
            self.downs.append(Down(nfilters, doubleconv, i))

        self.ups = []
        self.upsamples = []
        self.last_convs = nn.ModuleList()
        for c in range(nclass):
            ups = nn.ModuleList()
            upsamples = nn.ModuleList()
            for i in range(self.nlayers-2, -1, -1):
                ups.append(Up(nfilters, doubleconv, i))
                upsamples.append(Upsample(nfilters, i))
            self.ups.append(ups)
            self.upsamples.append(upsamples)
            self.last_convs.append(nn.Conv2d(nfilters[0], 1, kernel_size=(3, 3), padding=(1, 1)))

    def forward(self, x):
        skip_connections = []
        i = 0
        for down_conv in self.downs:
            x = down_conv(x)
            skip_connections.append(self.relu(x))
            x = self.pool(skip_connections[i])
            i = i+1
        x, y = self.last_down(x)

        skip_connections = skip_connections[::-1]

        for c in range(self.nclass):
            for j in range(len(self.ups)):
                if j == 0:
                    x = self.upsamples[c][j](y)
                else:
                    x = self.upsamples[c][j](x)
                if x.shape != skip_connections[j].shape:
                    x = TF.resize(x, size=skip_connections[j].shape[2:])
                x = torch.cat((x, skip_connections[j]), dim=1)
                x = self.ups[c][j](x)
            x = self.last_convs[j](x)
            if c == 0:
                z = x
            else:
                z = torch.cat((x, z), dim=1)
        return z, y

def test():
    x = torch.randn((3, 3, 178, 218))
    nfilters = [32, 32, 64, 128, 256, 512]
    model = CNN(nfilters, 4, True)
    z, y = model(x)

    #model = Upsample(nfilters, len(nfilters)-2)
    #preds = model(x)
    print(x.shape)
    print(z.shape)
    print(y.shape)
if __name__ == "__main__":
    test()