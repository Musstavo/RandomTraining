import os
import random
from PIL import Image
import torch
from torch import nn
import requests
import zipfile
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

if image_path.is_dir():
    print(f"{image_path} exists")

else:
    print(f"Didn't find, creating one..")
    image_path.mkdir(parents=True, exist_ok=True)

    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get(
            "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
        )
        print("Downloading..")
        f.write(request.content)

    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzippin..")
        zip_ref.extractall(image_path)


def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'."
        )

# walk_through_dir(image_path)

train_dir = image_path / "train"
test_dir = image_path / "test"

random.seed(42)

image_path_list = list(image_path.glob("*/*/*.jpg")
random_image_path = random.choice(image_path_list)

image_class = random_image_path.parent.stem

