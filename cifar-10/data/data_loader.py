import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader


def get_loader(config):
    transform_image = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.cifar.CIFAR10(root='./data/dataset/',
                                                       train=True,
                                                       transform=transform_image,
                                                       download=True)

    test_dataset = torchvision.datasets.cifar.CIFAR10(root='./data/dataset/',
                                                      train=False,
                                                      transform=transform_image)

    # Data loader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             shuffle=False)

    return train_loader, test_loader
