import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from matplotlib import pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torchvision
from CNN import mat_to_py
from ml_utils import show
from torchvision.utils import make_grid

MASK_DIR = "../unet/segmentation_masks"
IMAGE_DIR = "../unet/img_align_celeba"
TRANSFORM = A.Compose(
        [
            A.Crop(x_min=11, x_max=167, y_min=40, y_max=196),
            A.Resize(width=90, height=90),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
        additional_targets={"shading": "mask"},
    )

class Dataset(Dataset):
    def __init__(self):
        self.image_dir = IMAGE_DIR
        self.mask_dir = MASK_DIR
        self.transform = TRANSFORM
        self.mask_images = os.listdir(self.mask_dir)

    def __len__(self):
        return len(self.mask_images)

    def __getitem__(self, index):
        mask_path = os.path.join(self.mask_dir, self.mask_images[index])
        img_path = os.path.join(self.image_dir, self.mask_images[index].replace(".bmp", ".jpg"))
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        shading = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask >= 129] = 0.0
        mask[mask >= 1] = 1.0

        augmentations = self.transform(image=image, mask=mask, shading=shading)
        image = augmentations["image"]
        mask = augmentations["mask"]
        shading = augmentations["shading"]

        shading = shading.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return image, shading, mask


if __name__ == "__main__":
    dataset = Dataset()
    test_loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
        pin_memory=True,
        shuffle=True,
    )

    dataiter = iter(test_loader)
    images, shading, mask = dataiter.next()
    print(images.shape)
    print(shading.shape)
    print(mask.shape)

    torchvision.utils.save_image(
        images, f"saved_images/NORMALIZE_TEST.png"
    )

    # plt.imshow(np.transpose(images[0], (1, 2, 0)))
    # plt.show()
    #
    # # plt.imshow(np.transpose(shading[0], (1, 2, 0)))
    # # plt.show()
    # plt.imshow(shading[0][0], cmap='gray')
    # plt.show()
    #
    #
    # plt.imshow(mask[0][0], cmap='gray')
    # plt.show()