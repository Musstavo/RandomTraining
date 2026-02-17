import os
import random
from PIL import Image
import torch
from torch import nn
import requests
import zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# if image_path.is_dir():
#     print(f"{image_path} exists")
#
# else:
#     print(f"Didn't find, creating one..")
#     image_path.mkdir(parents=True, exist_ok=True)
#
#     with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
#         request = requests.get(
#             "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
#         )
#         print("Downloading..")
#         f.write(request.content)
#
#     with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
#         print("Unzippin..")
#         zip_ref.extractall(image_path)


def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'."
        )


# walk_through_dir(image_path)

train_dir = image_path / "train"
test_dir = image_path / "test"

random.seed(42)
image_path_list = list(image_path.glob("*/*/*.jpg"))
random_image_path = random.choice(image_path_list)
image_class = random_image_path.parent.stem
img = Image.open(random_image_path)

# img.show()
# print(f"Random image path: {random_image_path}")
# print(f"Image class: {image_class}")
# print(f"Image height: {img.height}")
# print(f"Image width: {img.width}")

# img_as_array = np.asarray(img)
# plt.figure(figsize=(10, 7))
# plt.imshow(img_as_array)
# plt.axis(False)
# plt.show()

data_transform = transforms.Compose(
    [
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ]
)


def plot_transformed_images(image_path, transform, n=3, seed=42):
    random.seed(seed)
    random_images_paths = random.sample(image_path_list, k=n)
    for random_path in random_images_paths:
        with Image.open(random_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"OG \nSIZE: {f.size}")
            ax[0].axis("off")

            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {random_path.parent.stem}", fontsize=16)
            plt.show()


# plot_transformed_images(image_path_list, transform=data_transform, n=3)

train_data = datasets.ImageFolder(
    root=train_dir, transform=data_transform, target_transform=None
)

test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)

class_names = train_data.classes
img, label = train_data[0][0], train_data[0][1]

img_permute = img.permute(1, 2, 0)
plt.figure(figsize=(10, 7))
plt.imshow(img.permute(1, 2, 0))
plt.axis("off")
plt.title(class_names[label], fontsize=14)
plt.show()
