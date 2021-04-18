import torch
import torch.nn as nn
from fcn_multipleDecoders import CNN

class Model(nn.Module):
    def __init__(self, LightVectorSize, bSize, nfilters, nclass):
        super(Model, self).__init__()
        self.LightVectorSize = LightVectorSize
        self.bSize = bSize
        self.cnn = CNN(nfilters, nclass, True)

        dims = LightVectorSize + bSize
        self.conv = nn.Sequential(
            # fc1
            nn.Conv2d(512, 512, (4, 4), stride=(2, 2)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # fc2
            nn.Conv2d(512, 512, (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # fc3
            nn.Conv2d(512, dims, (1, 1))
        )
    def forward(self, x):
        x, y = self.cnn(x)
        x = py_to_mat(x)
        #print(y.shape)
        prediction = self.conv(y)
        prediction = py_to_mat(prediction)
        lightingparameters = prediction[:, :, 0:self.LightVectorSize, :]
        nbatch = list(prediction.size())[3]
        #print(prediction.shape)
        b = torch.reshape(prediction[:, :, self.LightVectorSize:self.LightVectorSize+self.bSize, :], (self.bSize, nbatch))

        fmel = x[:, :, 0, :].unsqueeze(2)
        fblood = x[:, :, 1, :].unsqueeze(2)
        Shading = x[:, :, 2, :].unsqueeze(2)
        specmask = x[:, :, 3, :].unsqueeze(2)

        return lightingparameters, b, fmel, fblood, Shading, specmask

def mat_to_py(x):
    return x.permute((3, 2, 0, 1))

def py_to_mat(x):
    return x.permute((2, 3, 1, 0))

def test():
    x = torch.randn((2, 3, 90, 90)) #NOT QUITE SURE IMAGE SIZE
    nfilters = [32, 64, 128, 256, 512]
    model = Model(15, 2, nfilters, 4)
    lightingparameters, b, fmel, fblood, Shading, specmask = model(x)
    print(lightingparameters.shape)
    print(b.shape)
    print(fmel.shape)
    print(fblood.shape)
    print(Shading.shape)
    print(specmask.shape)

if __name__ == "__main__":
    test()