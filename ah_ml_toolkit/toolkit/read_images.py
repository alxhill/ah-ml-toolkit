from pathlib import Path

from PIL import Image
from torch import Tensor
from torchvision import transforms


def images_from_dir(input_path: str) -> list[Image]:
    if not Path(input_path).exists():
        raise Exception("Directory not found")

    return [Image.open(img) for img in Path(input_path).glob("*")]


def constrain_to_size(imgs: list[Image], limits: tuple[int, int]) -> Image:
    out = [img.copy() for img in imgs]
    for img in out:
        img.thumbnail(limits)
    return out


def imgs_to_tensors(imgs: list[Image]) -> list[Tensor]:
    return [transforms.ToTensor()(img) for img in imgs]
