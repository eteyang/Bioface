import torch
import torch.nn.functional as F

def cameraModel(mu, PC, b, wavelength):
    nbatch = list(b.size())[1]
    # mu = mu.repeat(nbatch, 1)   # bit sus
    # mu = torch.transpose(mu, 0, 1)  #bit sus
    #
    mu = torch.reshape(mu, (99, 1))
    S = torch.matmul(PC, b)+mu
    S = F.relu(S)
    Sr = torch.reshape(S[0:wavelength, :], (1, 1, wavelength, nbatch))
    Sg = torch.reshape(S[wavelength: 2*wavelength, :], (1, 1, wavelength, nbatch))
    Sb = torch.reshape(S[2*wavelength: 3*wavelength, :], (1, 1, wavelength, nbatch))
    return Sr, Sg, Sb

# mu = torch.rand(99, 10)
# PC = torch.rand(99, 15)
# b = torch.rand(15, 10)
# wavelength = 33
# r, g, b = cameraModel(mu, PC, b, wavelength)
# print(r.shape)
# print(g.shape)
# print(b.shape)
