import torch
import torch.nn.functional as f

def BiotoSpectralRef(fmel, fblood, Newskincolour):
    biophysicalMaps = torch.cat((fblood, fmel), 2)
    biophysicalMaps = biophysicalMaps.permute(3, 0, 1, 2)



    nbatch = list(fmel.size())[3]
    Newskincolour = Newskincolour.repeat(nbatch, 1, 1, 1)
    Newskincolour = Newskincolour.permute(1, 2, 3, 0)

    Newskincolour = Newskincolour.permute(3, 2, 0, 1)
    #mu = torch.transpose(mu, 0, 1)  #bit sus

    R_total = f.grid_sample(Newskincolour, biophysicalMaps, mode='bilinear', align_corners=True, padding_mode='reflection')
    R_total = R_total.permute(2, 3, 1, 0)
    return R_total


# H = 20
# W = 30
# B = 10
# fmel = torch.rand((H, W, 1, B))
# fblood = torch.rand((H, W, 1, B))
# Newskincolour = torch.rand((256, 256, 33, B))
# R_total = BiotoSpectralRef(fmel, fblood, Newskincolour)
# print(R_total.shape)
