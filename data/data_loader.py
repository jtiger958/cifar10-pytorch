import torchvision
import torch
import torchvision.transforms as transforms

def get_loader(batch_size):
    # MNIST dataset

    transform_image = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.cifar.CIFAR100(root='./data/dataset/',
                                               train=True,
                                               transform=transform_image,
                                               download=True)

    test_dataset = torchvision.datasets.cifar.CIFAR100(root='./data/dataset/',
                                              train=False,
                                              transform=transform_image)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_loader, test_loader