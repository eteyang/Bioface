import torch

def ImageFormation(R_total, Sr, Sg, Sb, e, Specularities, Shading):
    spectraRef = R_total*e

    rChannel = torch.unsqueeze(torch.sum((spectraRef * Sr), 2), 2)
    gChannel = torch.unsqueeze(torch.sum((spectraRef * Sg), 2), 2)
    bChannel = torch.unsqueeze(torch.sum((spectraRef * Sb), 2), 2)

    diffuseAlbedo = torch.cat((rChannel, gChannel, bChannel), 2)
    shadedDiffuse = diffuseAlbedo*Shading
    rawAppearance = shadedDiffuse+Specularities

    return rawAppearance, diffuseAlbedo

# R_total = torch.rand(20, 30, 33, 10)
# Shading = torch.rand(20, 30, 1, 10)
# Specularities = torch.rand(20, 30, 1, 10)
# sr = torch.rand(1, 1, 33, 10)
# sg = torch.rand(1, 1, 33, 10)
# sb = torch.rand(1, 1, 33, 10)
# e = torch.rand(1, 1, 33, 10)
# ImageFormation(R_total, sr, sg, sb, e, Specularities, Shading)