import torch

def illuminationModel(weightA,weightD,Fweights,CCT,illumA, illumDNorm,illumFNorm):
    nbatch = list(weightD.size())[3]
    illuminantA = illumA*weightA

    #illuminantD = illumDNorm*weightD
    illumD = torch.zeros(1, 1, 33, nbatch)
    for i in range (nbatch):
        illumD[0, 0, :, i] = illumDNorm[0, 0, :, int(CCT[0, 0, 0, i].item())]
    illumD = illumD*weightD

    illumFNorm = illumFNorm.permute((0, 2, 3, 1))
    illuminantF = illumFNorm*Fweights
    #print(illuminantF.shape)
    illuminantF = torch.sum(illuminantF, 2).unsqueeze(2)
    #print(illuminantF.shape)
    illuminantF = illuminantF.permute((0, 2, 1, 3))

    #e = illuminantA+illuminantD+illuminantF
    e = illuminantA + illuminantF#+illumD   #Not using illuminant D
    esums = torch.sum(e, 2, keepdim=True)
    e = e/esums
    return e

if __name__ == '__main__':
    weightA = torch.rand(1, 1, 1, 10)
    weightD = torch.rand(1, 1, 1, 10)
    Fweights = torch.rand(1, 1, 12, 10)
    CCT = torch.rand(1, 1, 1, 10)
    CCT = torch.round(((22 - 1) / (1 + torch.exp(-CCT))))
    Ftyper = torch.rand(1, 1,12, 10)
    illumA = torch.rand(1, 1, 33, 10)
    illumDNorm = torch.rand(1, 1, 33, 22)
    illumFNorm = torch.rand(1, 1, 33, 12)

    print(illuminationModel(weightA, weightD, Fweights, CCT, illumA, illumDNorm, illumFNorm).shape)