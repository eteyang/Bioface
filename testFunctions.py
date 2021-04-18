import scipy.io as spio
import torch
import numpy as np
from scalingNet import scalingNet
from setup import *
from illuminationModel import illuminationModel
from CameraModel import cameraModel
from computelightcolour import computelightcolour
from computeSpecularities import computeSpecularities
from BiotoSpectralRef import BiotoSpectralRef
from ImageFormation import ImageFormation
from WhiteBalance import WhiteBalance
from findT import findT
from fromRAWTosRGB import fromRawTosRGB

TEST_VAR_DIR = "..\..\Downloads\BioFaces-master\BioFaces-master\\testVar"


def load_mat_data(file):
    file_path = f"{TEST_VAR_DIR}\\{file}.mat"
    data = spio.loadmat(file_path, mat_dtype=True)
    np_data = np.array(data[file])

    tensor_data = torch.Tensor(np_data)
    return tensor_data

lightingparameters = load_mat_data("lightingparams")
b= load_mat_data("b")
fmel = load_mat_data("fmel")
fblood = load_mat_data("fblood")
predictedShading = load_mat_data("predictedShading")
specmask = load_mat_data("specmask")
PC = load_mat_data("PC")
mu = load_mat_data("mu")

weightA, weightD, CCT, Fweights, b, BGrid, fmel, fblood, predictedShading, specmask = scalingNet(lightingparameters, b,
                                                                                                 fmel,
                                                                                                 fblood,
                                                                                                 predictedShading,
                                                                                                 specmask,
                                                                                                  2)
Sr, Sg, Sb = cameraModel(mu, PC, b, wavelength)
# print(weightA)
# print(illumA)
# print((illumA*weightA).shape)
# print(illumA*weightA)
# print((illumA*weightA)[0, 0, 26, 53])
e = illuminationModel(weightA,weightD,Fweights,CCT,illumA, illumDNorm,illumFNorm)
# print(weightA.shape)
Sr, Sg, Sb = cameraModel(mu, PC, b, wavelength)
lightcolour = computelightcolour(e, Sr, Sg, Sb)
Specularities = computeSpecularities(specmask,lightcolour)

#R_total = BiotoSpectralRef(fmel, fblood, Newskincolour)
R_total = load_mat_data("R_total")
rawAppearance, diffuseAlbedo = ImageFormation(R_total, Sr, Sg, Sb, e, Specularities, predictedShading)
ImwhiteBalanced = WhiteBalance(rawAppearance, lightcolour)
#T_RAW2XYZ = findT(Tmatrix, BGrid)
T_RAW2XYZ = load_mat_data("T_RAW2XYZ")
sRGBim = fromRawTosRGB(ImwhiteBalanced, T_RAW2XYZ)
print(sRGBim[:, :, 1, 0])