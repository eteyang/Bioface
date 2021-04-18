import torch
import torch.nn.functional as f

def findT(Tmatrix, BGrid):
    nbatch = list(BGrid.size())[3]
    Tmatrix = Tmatrix.repeat(nbatch, 1, 1, 1)
    Tmatrix = Tmatrix.permute(1, 2, 3, 0)

    Tmatrix = Tmatrix.permute(3, 2, 0, 1)
    BGrid = BGrid.permute(3, 1, 2, 0)
    T_RAW2XYZ = f.grid_sample(Tmatrix, BGrid, mode='bilinear', padding_mode='zeros', align_corners=True)
    T_RAW2XYZ = T_RAW2XYZ.permute(2, 3, 1, 0)
    return T_RAW2XYZ

# Tmatrix = torch.rand(128, 128, 9, 15)
# BGrid = torch.rand(2, 1, 1, 15)
# print(findT(Tmatrix, BGrid).shape)
