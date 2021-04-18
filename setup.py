import torch
from utils import *
#from CameraSensitivtyPCA import *

LightVectorSize = 15  # 15 paramters of light model
wavelength = 33
bSize = 2  # 2 parameters of camera model

illF = torch.reshape(illF, (1, 1, 33, 12))
illumDmeasured = torch.transpose(illumDmeasured, 0, 1)
illumDmeasured = torch.reshape(illumDmeasured, (1, 1, 33, 22))

illumA = illumA/torch.sum(illumA[:])
illumA = illumA.unsqueeze(3)
illumDNorm = torch.zeros(1, 1, 33, 22)
for i in range(22):
    illumDNorm[0,0,:,i] = illumDmeasured[0,0,:,i]/torch.sum(illumDmeasured[0,0,:,i])

illumFNorm = torch.zeros(1,1,33,12)
for i in range(12):
    illumFNorm[0,0,:,i] = illF[0,0,:,i]/torch.sum(illF[0,0,:,i])

# print(illumA.shape)
# print(illF.shape)
# print(illumDNorm.shape)
# print(illumFNorm.shape)