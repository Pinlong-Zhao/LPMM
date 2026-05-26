"""
Data loading utilities for various datasets.
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


def get_dataset(name, data_path='./data', download=True):
    """
    Get dataset by name.
    
    Args:
        name: Dataset name ('cifar10', 'cifar100', 'imagenet', etc.)
        data_path: Path to data directory
        download: Whether to download if not exists
        
    Returns:
        train_dataset, test_dataset
    """
    name = name.lower()
    
    if name == 'cifar10':
        return get_cifar10(data_path, download)
    elif name == 'cifar100':
        return get_cifar100(data_path, download)
    elif name == 'imagenet':
        return get_imagenet(data_path)
    elif name == 'tiny-imagenet':
        return get_tiny_imagenet(data_path)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_cifar10(data_path, download=True):
    """Load CIFAR-10 dataset."""
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # No augmentation for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=download,
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=download,
        transform=transform_test
    )
    
    return train_dataset, test_dataset


def get_cifar100(data_path, download=True):
    """Load CIFAR-100 dataset."""
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    train_dataset = datasets.CIFAR100(
        root=data_path,
        train=True,
        download=download,
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR100(
        root=data_path,
        train=False,
        download=download,
        transform=transform_test
    )
    
    return train_dataset, test_dataset


def get_imagenet(data_path):
    """Load ImageNet dataset."""
    
    train_dir = os.path.join(data_path, 'imagenet', 'train')
    val_dir = os.path.join(data_path, 'imagenet', 'val')
    
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(val_dir, transform=transform_test)
    
    return train_dataset, test_dataset


def get_tiny_imagenet(data_path):
    """Load Tiny-ImageNet dataset."""
    
    train_dir = os.path.join(data_path, 'tiny-imagenet-200', 'train')
    val_dir = os.path.join(data_path, 'tiny-imagenet-200', 'val')
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(val_dir, transform=transform_test)
    
    return train_dataset, test_dataset


class SyntheticDataset(Dataset):
    """
    Dataset wrapper for synthetic distilled data.
    """
    
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: Tensor of images (N, C, H, W)
            labels: Tensor of labels (N,)
            transform: Optional transform to apply
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
