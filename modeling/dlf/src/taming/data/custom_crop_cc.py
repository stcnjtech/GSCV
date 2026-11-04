import os
import random as rd
import numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from datasets.splits import Split
from datasets.arrow_dataset import ArrowReader, DatasetInfo
from datasets.arrow_dataset import Dataset as ArrowDataset


class modified_ArrowReader(ArrowReader):
    def read(self, name, instructions, split_infos, in_memory, length=None):
        files = self.get_file_instructions(name, instructions, split_infos)
        if not files:
            msg = f'Instruction "{instructions}" corresponds to no data!'
            raise ValueError(msg)
        if length is not None:
            files = files[:length]
        return self.read_files(files=files, original_instructions=instructions, in_memory=in_memory)


class ResizeIfSmall(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, img):
        _, height, width = F.get_dimensions(img)
        if height < self.patch_size or width < self.patch_size:
            img = F.resize(img, self.patch_size)
        return img


class CommonCanvas_HF_crop(Dataset):
    def __init__(self, data_dir, crop_size=None, loading_length=None, in_memory=False):
        # define base common canvas dataset
        self.dataset_info = DatasetInfo().from_directory(data_dir)
        self.dataset_name = "commoncatalog-cc-by-nd"
        self.dataset_split = Split.TRAIN
        self.in_memory = in_memory
        
        dataset_kwargs = modified_ArrowReader(data_dir, self.dataset_info).read(
            name=self.dataset_name,
            instructions=self.dataset_split,
            split_infos=self.dataset_info.splits.values(),
            in_memory=in_memory,
            length=loading_length
        )
        self.base_dataset = ArrowDataset(**dataset_kwargs)
        
        # define transforms
        transform_list = [
            ResizeIfSmall(crop_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
        self.transform = transforms.Compose(transform_list)


    def __len__(self):
        return len(self.base_dataset)


    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        img = self.transform(item['jpg'].convert('RGB')) * 2. - 1.
        img = img.permute(1, 2, 0)
        caption = item['blip2_caption']
        img_name = item['unickname']
        return {
            'image': img, 
            'caption': caption, 
            'name': img_name, 
        }


# class CustomBase(Dataset):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.data = None

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, i):
#         example = self.data[i]
#         return example


# class CustomTrain(CustomBase):
#     def __init__(self, data_dir, crop_size=None, loading_length=None, in_memory=False):
#         super().__init__()
#         self.data = CommonCanvas_HF_crop(data_dir, crop_size, loading_length, in_memory)