import pandas as pd
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class UAVCottonDataset(Dataset):
    """
    UAV Cotton Crop Disease Dataset loader reading from manifest CSV.
    """
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.image_paths = self.df['image_path'].tolist()
        self.labels = self.df['label_idx'].tolist()
        self.label_names = self.df['label'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(image_size=224):
    """
    Returns UAV-specific training and validation transforms.
    Since UAV imagery is captured from an overhead aerial drone,
    it is rotationally invariant (no canonical top/down orientation).
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def get_dataloaders(splits_dir, image_size=224, batch_size=32, num_workers=4):
    splits_dir = Path(splits_dir)
    train_transform, val_transform = get_transforms(image_size=image_size)

    train_dataset = UAVCottonDataset(splits_dir / 'train.csv', transform=train_transform)
    val_dataset = UAVCottonDataset(splits_dir / 'val.csv', transform=val_transform)
    test_dataset = UAVCottonDataset(splits_dir / 'test.csv', transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
