import torch
from torch.utils.data import Dataset, DataLoader

# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        image_tensor = torch.tensor(image, dtype=torch.float32)
        return image_tensor