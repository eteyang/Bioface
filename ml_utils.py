import torch
import torchvision
from CNN import mat_to_py
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

def save_predictions_as_imgs(
    loader, model, epoch, folder="saved_images/", device="cuda"
):
    print("> Saving images")
    model.eval()
    dataiter = iter(loader)
    images, shading, mask = dataiter.next()
    images = images.to(device=device)

    with torch.no_grad():
        lightingparameters, b, fmel, fblood, predictedShading, specmask = model(images)
    fmel = mat_to_py(fmel)
    fblood = mat_to_py(fblood)
    predictedShading = mat_to_py(predictedShading)
    specmask = mat_to_py(specmask)

    fmel = fmel*mask
    fblood = fblood*mask
    predictedShading = predictedShading*mask
    specmask = specmask*mask
    images = images*mask


    torchvision.utils.save_image(
        mask, f"{folder}/{epoch}_mask.png"
    )
    mask_img = cv2.imread(f"{folder}/{epoch}_mask.png", 0)
    torchvision.utils.save_image(
        fmel, f"{folder}/{epoch}_fmel.png"
    )
    fmel = cv2.imread(f"{folder}/{epoch}_fmel.png", 0)
    heatmap = cv2.applyColorMap(fmel, cv2.COLORMAP_JET)
    heatmap2 = cv2.bitwise_and(heatmap, heatmap, mask=mask_img)
    cv2.imwrite(f"{folder}/{epoch}_fmel.png", heatmap2)

    torchvision.utils.save_image(
        fblood, f"{folder}/{epoch}_fblood.png"
    )
    fblood = cv2.imread(f"{folder}/{epoch}_fblood.png", 0)
    heatmap = cv2.applyColorMap(fblood, cv2.COLORMAP_JET)
    heatmap2 = cv2.bitwise_and(heatmap, heatmap, mask=mask_img)
    cv2.imwrite(f"{folder}/{epoch}_fblood.png", heatmap2)

    torchvision.utils.save_image(
        predictedShading, f"{folder}/{epoch}_shading.png"
    )
    torchvision.utils.save_image(
        specmask, f"{folder}/{epoch}_spec.png"
    )
    torchvision.utils.save_image(
        images, f"{folder}/{epoch}_true.png"
    )
    os.remove(f"{folder}/{epoch}_mask.png")
    print("> Finish saving images")

# def save_current(
#     fmel, fblood, predictedShading, specmask, mask, images, folder="saved_images"
# ):
#     fmel = mat_to_py(fmel)
#     fblood = mat_to_py(fblood)
#     predictedShading = mat_to_py(predictedShading)
#     specmask = mat_to_py(specmask)
#
#     fmel = fmel*mask
#     fblood = fblood*mask
#     predictedShading = predictedShading*mask
#     specmask = specmask*mask
#     images = images*mask
#
#
#     torchvision.utils.save_image(
#         mask, f"{folder}/CURRENT_mask.png"
#     )
#     mask_img = cv2.imread(f"{folder}/CURRENT_mask.png", 0)
#     torchvision.utils.save_image(
#         fmel, f"{folder}/CURRENT_fmel.png"
#     )
#     fmel = cv2.imread(f"{folder}/CURRENT_fmel.png", 0)
#     heatmap = cv2.applyColorMap(fmel, cv2.COLORMAP_JET)
#     heatmap2 = cv2.bitwise_and(heatmap, heatmap, mask=mask_img)
#     cv2.imwrite(f"{folder}/CURRENT_fmel.png", heatmap2)
#
#     torchvision.utils.save_image(
#         fblood, f"{folder}/CURRENT_fblood.png"
#     )
#     fblood = cv2.imread(f"{folder}/CURRENT_fblood.png", 0)
#     heatmap = cv2.applyColorMap(fblood, cv2.COLORMAP_JET)
#     heatmap2 = cv2.bitwise_and(heatmap, heatmap, mask=mask_img)
#     cv2.imwrite(f"{folder}/CURRENT_fblood.png", heatmap2)
#
#     torchvision.utils.save_image(
#         predictedShading, f"{folder}/CURRENT_shading.png"
#     )
#     torchvision.utils.save_image(
#         specmask, f"{folder}/CURRENT_spec.png"
#     )
#     torchvision.utils.save_image(
#         images, f"{folder}/CURRENT_true.png"
#     )
#     os.remove(f"{folder}/CURRENT_mask.png")

def save_current(
    fmel, fblood, predictedShading, specmask, mask, images, diffuseAlbedo, reconstruction, folder="saved_images"
):
    fmel = mat_to_py(fmel)
    fblood = mat_to_py(fblood)
    predictedShading = mat_to_py(predictedShading)
    specmask = mat_to_py(specmask)

    fmel = fmel*mask
    fblood = fblood*mask
    predictedShading = predictedShading*mask
    specmask = specmask*mask
    images = images*mask


    diffuseAlbedo = diffuseAlbedo*mask
    reconstruction = reconstruction*mask

    torchvision.utils.save_image(
        diffuseAlbedo, f"{folder}/CURRENT_diffuseAlbedo.png", normalize=True, scale_each=True
    )
    torchvision.utils.save_image(
        reconstruction, f"{folder}/CURRENT_reconstruction.png", normalize=True, scale_each=True
    )

    torchvision.utils.save_image(
        mask, f"{folder}/CURRENT_mask.png"
    )
    mask_img = cv2.imread(f"{folder}/CURRENT_mask.png", 0)
    torchvision.utils.save_image(
        fmel, f"{folder}/CURRENT_fmel.png", normalize=True, value_range=(-0.5, 0.4)
    )
    fmel = cv2.imread(f"{folder}/CURRENT_fmel.png", 0)
    heatmap = cv2.applyColorMap(fmel, cv2.COLORMAP_JET)
    heatmap2 = cv2.bitwise_and(heatmap, heatmap, mask=mask_img)
    cv2.imwrite(f"{folder}/CURRENT_fmel.png", heatmap2)

    torchvision.utils.save_image(
        fblood, f"{folder}/CURRENT_fblood.png", normalize=True, value_range=(-0.2, 0.3)
    )
    fblood = cv2.imread(f"{folder}/CURRENT_fblood.png", 0)
    heatmap = cv2.applyColorMap(fblood, cv2.COLORMAP_JET)
    heatmap2 = cv2.bitwise_and(heatmap, heatmap, mask=mask_img)
    cv2.imwrite(f"{folder}/CURRENT_fblood.png", heatmap2)

    torchvision.utils.save_image(
        predictedShading, f"{folder}/CURRENT_shading.png", normalize=True, value_range=(0, 0.3)
    )
    torchvision.utils.save_image(
        specmask, f"{folder}/CURRENT_spec.png"
    )
    torchvision.utils.save_image(
        images, f"{folder}/CURRENT_true.png"
    )
    os.remove(f"{folder}/CURRENT_mask.png")