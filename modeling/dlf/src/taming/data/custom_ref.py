import os
import json
import random as rd
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F


class ImageFolder_w_ref(Dataset):
    def __init__(self, root_foler, json_file, crop_size=384):
        with open(json_file, 'r') as f:
            self.img_dict = json.load(f)
            self.image_names = list(self.img_dict.keys())
        self.img_root = os.path.join(root_foler, "img_original")
        self.ref_root = os.path.join(root_foler, "img_sample")

        self.crop_size = crop_size
        print(f"Datasets: {len(self.image_names)} images are in {root_foler}")

    def __getitem__(self, index):
        # 1. load image
        image_path = os.path.join(self.img_root, self.image_names[index])
        img = Image.open(image_path).convert("RGB")

        # 2. load adapter's recon
        sample_list = self.img_dict[self.image_names[index]]["ref"]
        sample_name = rd.choice(sample_list)
        sample_path = os.path.join(self.ref_root, sample_name)
        sample = Image.open(sample_path).convert("RGB")
        
        # 3. random crop
        x = rd.randint(0, img.width - self.crop_size)
        y = rd.randint(0, img.height - self.crop_size)
        img = img.crop((x, y, x + self.crop_size, y + self.crop_size))
        sample = sample.crop((x, y, x + self.crop_size, y + self.crop_size))

        # 4. convert to tensor
        img = np.array(img).transpose(2, 0, 1)
        img = torch.as_tensor(img.astype(np.float32) / 255.0, dtype=torch.float32)
        img = img * 2.0 - 1.0
        sample = np.array(sample).transpose(2, 0, 1)
        sample = torch.as_tensor(sample.astype(np.float32) / 255.0, dtype=torch.float32)
        sample = sample * 2.0 - 1.0

        # 5. load caption
        caption_list = self.img_dict[self.image_names[index]]["caption"]
        caption = rd.choice(caption_list)

        return img, sample, caption


    def __len__(self):
        return len(self.image_names)


class ImageFolder_val_w_ref(Dataset):
    def __init__(self, root_foler, json_file, crop_size=384):
        with open(json_file, 'r') as f:
            self.img_dict = json.load(f)
            self.image_names = list(self.img_dict.keys())
        self.img_root = os.path.join(root_foler, "img_original")
        self.ref_root = os.path.join(root_foler, "img_sample")

        self.crop_size = crop_size
        print(f"Datasets: {len(self.image_names)} images are in {root_foler}")

    def __getitem__(self, index):
        # 1. load image
        image_path = os.path.join(self.img_root, self.image_names[index])
        img = Image.open(image_path).convert("RGB")

        # 2. load adapter's recon
        sample_list = self.img_dict[self.image_names[index]]["ref"]
        sample_name = sample_list[0]
        sample_path = os.path.join(self.ref_root, sample_name)
        sample = Image.open(sample_path).convert("RGB")

        # 3. convert to tensor
        img = np.array(img).transpose(2, 0, 1)
        img = torch.as_tensor(img.astype(np.float32) / 255.0, dtype=torch.float32)
        img = img * 2.0 - 1.0
        sample = np.array(sample).transpose(2, 0, 1)
        sample = torch.as_tensor(sample.astype(np.float32) / 255.0, dtype=torch.float32)
        sample = sample * 2.0 - 1.0

        if self.crop_size:
            img = F.center_crop(img, self.crop_size)
            sample = F.center_crop(sample, self.crop_size)

        caption_list = self.img_dict[self.image_names[index]]["caption"]
        caption = caption_list[0]

        return img, sample, caption


    def __len__(self):
        return len(self.image_names)


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, sample, caption = self.data[i]
        example = dict(image=img, reference=sample, caption=caption)
        return example



class CustomTrain(CustomBase):
    def __init__(self, root_foler, json_file, crop_size):
        super().__init__()
        self.data = ImageFolder_w_ref(root_foler=root_foler, json_file=json_file, crop_size=crop_size)


class CustomTest(CustomBase):
    def __init__(self, root_foler, json_file, crop_size):
        super().__init__()
        self.data = ImageFolder_val_w_ref(root_foler=root_foler, json_file=json_file, crop_size=crop_size)
