import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torchvision
import torch.optim as optim
from CNN import Model
from scalingNet import scalingNet
from setup import *
from illuminationModel import illuminationModel
from CameraModel import cameraModel
from computelightcolour import computelightcolour
from computeSpecularities import computeSpecularities
from BiotoSpectralRef import BiotoSpectralRef
from ImageFormation import ImageFormation
from WhiteBalance import WhiteBalance
from fromRAWTosRGB import fromRawTosRGB
from findT import findT
from Loss import Loss
from dataset import Dataset
from torch.utils.data import DataLoader
from CNN import mat_to_py
from CNN import py_to_mat
from ml_utils import (
    load_checkpoint,
    save_checkpoint,
    save_predictions_as_imgs,
    save_current
)
import math
from torchvision import transforms

averageImage = torch.tensor([129.1863, 104.7624, 93.5940])
muim = torch.reshape(averageImage,(1,1,3,1))

SAVE_IMAGE_PROGRESS = True
SAVE_IMAGE_FOLDER = "saved_images2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
nfilters = [32, 64, 128, 256, 512]
nclass = 4
#model = Model(LightVectorSize, bSize, nfilters, nclass)
model = Model(15, 2, nfilters, 4)
nimages = 50765
batchSize = 64

numEpochs = 200
learningRate = 1e-4

optimizer = optim.Adam(model.parameters(), lr=learningRate)
scaler = torch.cuda.amp.GradScaler()

dataset = Dataset()
loader = DataLoader(
    dataset,
    batchSize,
    num_workers=2,
    pin_memory=True,
    shuffle=True,
)

def train():
    loop = tqdm(loader)

    for batch_idx, (images, actualshading, actualmasks) in enumerate(loop):
        images = images.to(device=DEVICE)
        actualshading = actualshading.to(device=DEVICE)
        actualmasks = actualmasks.to(device=DEVICE)
        actualshading = py_to_mat(actualshading)
        actualmasks = py_to_mat(actualmasks)

        # forward
        with torch.cuda.amp.autocast():
            lightingparameters, b, fmel, fblood, predictedShading, specmask = model(images)
            image_predictedShading = predictedShading
            image_specmask = specmask
            #if(SAVE_IMAGE_PROGRESS):
            #    save_current(fmel, fblood, predictedShading, specmask, mat_to_py(actualmasks), images, SAVE_IMAGE_FOLDER)
            weightA, weightD, CCT, Fweights, b, BGrid, fmel, fblood, predictedShading, specmask = scalingNet(lightingparameters, b, fmel,
                                                                                                    fblood, predictedShading, specmask,
                                                                                                    bSize)
            e = illuminationModel(weightA,weightD,Fweights,CCT,illumA, illumDNorm,illumFNorm)
            Sr, Sg, Sb = cameraModel(mu, PC, b, wavelength)
            lightcolour = computelightcolour(e, Sr, Sg, Sb)
            Specularities = computeSpecularities(specmask,lightcolour)
            R_total = BiotoSpectralRef(fmel, fblood, Newskincolour)
            rawAppearance, diffuseAlbedo = ImageFormation(R_total, Sr, Sg, Sb, e, Specularities, predictedShading)
            ImwhiteBalanced = WhiteBalance(rawAppearance, lightcolour)
            T_RAW2XYZ = findT(Tmatrix, BGrid)
            #print(T_RAW2XYZ)
            sRGBim = fromRawTosRGB(ImwhiteBalanced, T_RAW2XYZ)

            #print(sRGBim)

            #print(sRGBim.shape)
            scaleRGB = sRGBim*255
            #Y1 = torch.ones(len(muim))
            #print(Y1.shape)
            #Y1 = Y1 * muim
            #print(Y1.shape)
            #rgbim = scaleRGB - Y1
            #X1 = torch.ones(len(rgbim))
            #rgbim = rgbim * X1
            rgbim = scaleRGB-125


            if torch.isnan(predictedShading).any():
                print("predictedshading is nan")
                quit()


            scale = torch.sum(torch.sum((actualshading*predictedShading)*actualmasks, 0, keepdim=True), 1, keepdim=True) / torch.sum(torch.sum(
                 torch.sum((predictedShading**2) * actualmasks, 1), 2))


            if torch.isnan(scale).any():
                print("scale is nan")
                quit()

            predictedShading = predictedShading*scale
            alpha = (actualshading - predictedShading) * actualmasks

        loss_function = Loss()
        loss = loss_function(b, py_to_mat(images), sRGBim, actualmasks, alpha, Specularities)
        if math.isnan(loss):
            print("nan loss")
            quit()

        if (SAVE_IMAGE_PROGRESS and batch_idx%10 == 0):
            # test_image = mat_to_py(sRGBim)
            # torchvision.utils.save_image(
            #     test_image, f"{SAVE_IMAGE_FOLDER}/test_rgb.png", normalize=True, scale_each=True
            # )
            # torchvision.utils.save_image(
            #     mat_to_py(rawAppearance), f"{SAVE_IMAGE_FOLDER}/test_white.png", normalize=True, scale_each=True
            # )
            # test_image = mat_to_py(diffuseAlbedo)
            # folder = "saved_images"
            # torchvision.utils.save_image(
            #     test_image, f"{SAVE_IMAGE_FOLDER}/test_albedo.png", normalize=True, scale_each=True, reversed=True
            # )
           save_current(fmel, fblood, image_predictedShading, image_specmask, mat_to_py(actualmasks), images,
                        mat_to_py(diffuseAlbedo), mat_to_py(sRGBim), SAVE_IMAGE_FOLDER)
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)

        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())
if __name__ == '__main__':
    LOAD_MODEL = True
    SAVE_INITIAL_IMAGE = False
    START_EPOCH = 1

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

        # check_accuracy(val_loader, model, device=DEVICE)
        if SAVE_INITIAL_IMAGE:
            save_predictions_as_imgs(
                loader, model, START_EPOCH, folder=SAVE_IMAGE_FOLDER, device=DEVICE,
            )
    else:
        START_EPOCH = 0
    for epoch in range(numEpochs):
        train()

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        #check_accuracy(val_loader, model, device=DEVICE)
        save_predictions_as_imgs(
            loader, model, epoch+START_EPOCH+1, folder=SAVE_IMAGE_FOLDER, device=DEVICE,
        )