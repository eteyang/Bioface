import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, b, images, rgbim, actualmasks, alpha, Specularities):
        blossweight = 1e-4
        appweight = 1e-3
        Shadingweight = 1e-5
        sparseweight = 1e-5

        if torch.isnan(b).any():
            print("b contains nan")

        if torch.isnan(rgbim).any():
            print("rgbim is nan")

        if torch.isnan(alpha).any():
            print("alpha is nan")

        if torch.isnan(Specularities).any():
            print("specularities is nan")

        # print(b.shape)
        # print(images.shape)
        # print(rgbim.shape)
        # print(actualmasks.shape)
        # print(alpha.shape)
        # print(Specularities.shape)

        priorB = torch.sum(b**2)
        priorloss = (priorB)*blossweight
        #ZY = torch.ones(len(priorloss))
        #priorloss = priorloss * ZY
        print("prior loss:")
        print(priorloss)

        delta = (images - rgbim) * actualmasks
        #print((torch.sum(delta**2)/(90*90)))
        appearanceloss = (torch.sum(delta**2)/(224 * 224))*appweight*100
        #Y = torch.ones(len(appearanceloss))
        #appearanceloss = appearanceloss * Y
        print("appearance loss:")
        print(appearanceloss)

        shadingloss = torch.sum(alpha**2) * Shadingweight
        #ff = torch.ones(len(shadingloss))
        #shadingloss = shadingloss * ff
        print("shading loss:")
        print(shadingloss)

        #print(Specularities.shape)
        sparsityloss = torch.sum(Specularities)*sparseweight
        #J = torch.ones(len(sparsityloss))
        #sparsityloss = sparsityloss * J
        print("sparsity loss:")
        print(sparsityloss)

        loss = appearanceloss + priorloss + sparsityloss #+ shadingloss
        return loss

# b = torch.rand(2,2)
# images = torch.rand(90, 90, 3, 2)
# rgbim = torch.rand(90, 90, 3, 2)
# actualmasks = torch.rand(90, 90, 1, 2)
# alpha = torch.rand(90, 90, 90, 2)
# Specularities = torch.rand(90, 90, 3, 2)
# model = Loss()
# loss = model(b, images, rgbim, actualmasks, alpha, Specularities)
# print(loss)