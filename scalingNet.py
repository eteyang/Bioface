import torch
import torch.nn as nn

def scalingNet(lightingparameters,b,fmel,fblood,Shading,specmask,bSize):
    nbatch = nbatch = list(lightingparameters.size())[3]

    softmax = nn.Softmax(dim=2)
    sigmoid = nn.Sigmoid()

    lightingweights = softmax(lightingparameters[:, :, 0:14, :])
    weightA = lightingweights[:, :, 0, :].unsqueeze(2)
    weightD = lightingweights[:, :, 1, :].unsqueeze(2)
    Fweights = lightingweights[:, :, 2:14, :]
    CCT = lightingparameters[:, :, 14, :].unsqueeze(2)
    CCT = torch.round(((22 - 1)/(1 + torch.exp(-CCT)))) #CHANGE TO DISCRETE, CORECT?
    #CCT = ((22 - 1) / (1 + torch.exp(-CCT)))  # CHANGE TO DISCRETE, CORECT?
    b = 6*sigmoid(b)-3
    BGrid = torch.reshape(b, (bSize, 1, 1, nbatch))
    BGrid = BGrid/3

    fmel = sigmoid(fmel)*2-1
    fblood = sigmoid(fblood)*2-1
    Shading = torch.exp(Shading)
    specmask = torch.exp(specmask)

    return weightA, weightD, CCT, Fweights, b, BGrid, fmel, fblood, Shading, specmask


# function [weightA,weightD,CCT,Fweights,b,BGrid,fmel,fblood,Shading,specmask] = scalingNet(lightingparameters,b,fmel,fblood,Shading,specmask,bSize)
# % Inputs/Output:
# %     weightA  : 1 x 1 x 1 x B
# %     weightD  : 1 x 1 x 1 x B
# %     CCT      : 1 x 1 x 1 x B
# %     Fweights : 1 x 1 x 12 x B
# %     b        : 1 x 1 x 2 x B
# %     fmel     : 224 x 224 x 1 x B
# %     fblood   : 224 x 224 x 1 x B
# %     Shading  : 224 x 224 x 1 x B
# %     specmask : 224 x 224 x 1 x B
# %     bSize    : 2
# % Output:
# % Scaled inputs
# nbatch = size(lightingparameters,4);

# lightingparameters = torch.rand(1, 1, 15, 10)
# b = torch.rand(1, 1, 2, 10)
# fmel = torch.rand(224, 224, 1, 10)
# fblood = torch.rand(224, 224, 1, 10)
# shading = torch.rand(224, 224, 1, 10)
# specmask = torch.rand(224, 224, 1, 10)
# weightA, weightD, CCT, Fweights, b, BGrid, fmel, fblood, Shading, specmask = scalingNet(lightingparameters, b, fmel, fblood, shading, specmask, 2)
# # print(weightA.shape)
# # print(weightD.shape)
# # print(CCT.shape)
# # print(Fweights.shape)
# # print(b.shape)
# # print(BGrid.shape)
# # print(fmel.shape)
# # print(fblood.shape)
# # print(Shading.shape)
# # print(specmask.shape)
# print(CCT)