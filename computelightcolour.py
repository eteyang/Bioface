import torch

def computelightcolour(e, Sr, Sg, Sb):
    size = list(e.size())[3]
    sr = torch.sum(Sr*e, dim=2).reshape(size).detach().numpy()
    sg = torch.sum(Sg*e, dim=2).reshape(size).detach().numpy()
    sb = torch.sum(Sb*e, dim=2).reshape(size).detach().numpy()
    lightcolour = torch.tensor([sr, sg, sb])
    lightcolour = torch.reshape(lightcolour, (1, 1, 3, size))
    return lightcolour

# Sr = torch.rand(1, 1, 33, 10)
# Sg = torch.rand(1, 1, 33, 10)
# Sb = torch.rand(1, 1, 33, 10)
# e = torch.rand(1, 1, 33, 10)
# light = computelightcolour(e, Sr, Sg, Sb)
# print(light.shape)

