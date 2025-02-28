import os
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms


class MyData(Dataset):
    def __init__(self, image_folder, excel_file, transform=None):
        self.image_folder = image_folder
        self.excel_file = excel_file
        self.transform = transform
        self.images_data = self.load_images()
        self.excel_data = self.load_excel()
        self.data_tensors = self.prepare_data_tensors()

    def load_images(self):
        images = {}
        for filename in os.listdir(self.image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(self.image_folder, filename)
                images[filename] = Image.open(image_path).convert('RGB')
        return images

    def load_excel(self):
        return pd.read_excel(self.excel_file, engine='openpyxl')

    def prepare_data_tensors(self):
        data_tensors = []
        for index, row in self.excel_data.iterrows():
            image_name = row['image']
            if image_name in self.images_data:
                data = torch.tensor(
                    [row['LFW'], row['LDW'], row['LA']], dtype=torch.float32)
                data_tensors.append(data)
            else:
                raise ValueError(
                    f"No image found for the data entry: {image_name}")
        return data_tensors

    def __len__(self):
        return len(self.data_tensors)

    def __getitem__(self, idx):
        image_name = self.excel_data.iloc[idx, 0]
        image = self.images_data[image_name]
        data = self.data_tensors[idx]

        if self.transform:
            image = self.transform(image)

        return image, data
