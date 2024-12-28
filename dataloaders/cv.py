import random
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from datasets import load_dataset

normalize_factors = {
        'CIFAR10': {'mean': np.array((0.4914, 0.4822, 0.4465)), 'std': np.array((0.2023, 0.1994, 0.2010))},
        'OCEA': {'mean': np.array((0.485, 0.456, 0.406)), 'std': np.array((0.229, 0.224, 0.225))},
        'MRI': {'mean': np.array((0.485, 0.456, 0.406)), 'std': np.array((0.229, 0.224, 0.225))},
        'SVHN': {'mean': np.array((0.5, 0.5, 0.5)), 'std': np.array((0.5, 0.5, 0.5))},
    }

def cifar10(n_test=10, random_state=42, subsample=False, reduce_sample=False):
    random.seed(random_state)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data/cifar', train=True, download=True, transform=transform_train)
    
    tens = list(range(0, len(trainset), 10))

    if reduce_sample:
        labels = trainset.targets
        class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)

        # Sample 5% of each class
        sampled_indices = []
        for label, indices in class_indices.items():
            sample_size = max(1, int(0.05 * len(indices)))  # At least one sample
            sampled_indices.extend(random.sample(indices, sample_size))

        # Create a subset of the training dataset
        trainset = Subset(trainset, sampled_indices)

        # Use the sampled dataset in a DataLoader (optional)
        trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    else:
        if subsample:
            trainset_1 = torch.utils.data.Subset(trainset, tens)

            trainloader = torch.utils.data.DataLoader(
                trainset_1, batch_size=128, shuffle=False)
        else:
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=128, shuffle=False)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data/cifar', train=False, download=True, transform=transform_test)
    
    G = torch.Generator()
    G.manual_seed(random_state)
    sampler = torch.utils.data.RandomSampler(testset, replacement=False, 
                                             generator=G)
    
    testloader = torch.utils.data.DataLoader(
    testset, batch_size=n_test, sampler=sampler, generator=G)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes


def ocea(n_test, random_state=42, subsample=False, reduce_sample=False):
    random.seed(random_state)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    trainset = torchvision.datasets.ImageFolder("data/ocea/train/", transform=transform_train)
    
    tens = list(range(0, len(trainset), 10))

    if subsample:
        trainset_1 = torch.utils.data.Subset(trainset, tens)

        trainloader = torch.utils.data.DataLoader(
            trainset_1, batch_size=128, shuffle=False)
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=False)
    

    testset = torchvision.datasets.ImageFolder("data/ocea/test/", transform=transform_test)
    
    G = torch.Generator()
    G.manual_seed(random_state)
    sampler = torch.utils.data.RandomSampler(testset, replacement=False, 
                                             generator=G)
    
    testloader = torch.utils.data.DataLoader(
    testset, batch_size=n_test, sampler=sampler, generator=G)
    
    classes = ('CC', 'EC', 'HGSC', 'LGSC', 'MC')
    
    return trainloader, testloader, classes


def mri(n_test, random_state=42, subsample=False):
    transform_train = transforms.Compose([
        transforms.Resize((128,128)), 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((128,128)), 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    trainset = torchvision.datasets.ImageFolder("data/mri/train/", transform=transform_train)
    
    tens = list(range(0, len(trainset), 1))

    if subsample:
        trainset_1 = torch.utils.data.Subset(trainset, tens)

        trainloader = torch.utils.data.DataLoader(
            trainset_1, batch_size=128, shuffle=False)
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=False)
    

    testset = torchvision.datasets.ImageFolder("data/mri/test/", transform=transform_test)
    
    G = torch.Generator()
    G.manual_seed(random_state)
    sampler = torch.utils.data.RandomSampler(testset, replacement=False, 
                                             generator=G)
    
    testloader = torch.utils.data.DataLoader(
    testset, batch_size=n_test, sampler=sampler, generator=G)
    
    classes = ('Pituitary', 'Meningioma', 'Glioma', 'No Tumor')
    
    return trainloader, testloader, classes


def svhn(n_test=10, random_state=42, subsample=False, reduce_sample=False):
    random.seed(random_state)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.SVHN(
        root='./data/svhn', split='train', download=True, transform=transform_train)

    tens = list(range(0, len(trainset), 10))
    if reduce_sample:
        labels = trainset.targets
        class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)

        # Sample 5% of each class
        sampled_indices = []
        for label, indices in class_indices.items():
            sample_size = max(1, int(0.01 * len(indices)))  # At least one sample
            sampled_indices.extend(random.sample(indices, sample_size))

        # Create a subset of the training dataset
        trainset = Subset(trainset, sampled_indices)

        # Use the sampled dataset in a DataLoader (optional)
        trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    else:
        if subsample:
            trainset_1 = torch.utils.data.Subset(trainset, tens)

            trainloader = torch.utils.data.DataLoader(
                trainset_1, batch_size=128, shuffle=False)
        else:
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=128, shuffle=False)
    
    if subsample:
        trainset_1 = torch.utils.data.Subset(trainset, tens)

        trainloader = torch.utils.data.DataLoader(
            trainset_1, batch_size=128, shuffle=False)
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=False)

    testset = torchvision.datasets.SVHN(
        root='./data/svhn', split='test', download=True, transform=transform_test)

    G = torch.Generator()
    G.manual_seed(random_state)
    sampler = torch.utils.data.RandomSampler(testset, replacement=False,
                                             generator=G)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=n_test, sampler=sampler, generator=G)

    classes = tuple(str(i) for i in range(10))  # SVHN classes are digits from 0 to 9

    return trainloader, testloader, classes


def svhn_224(n_test, random_state=42, subsample=False, reduce_sample=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size=512
    # Custom PyTorch Dataset class for SVHN
    class SVHNDataset(Dataset):
        def __init__(self, dataset, transform=None):
            """
            Args:
                dataset (Dataset): Hugging Face dataset object.
                transform (callable, optional): A function/transform to apply to the images.
            """
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            # Get the image and label
            data = self.dataset[idx]
            image, label = data["image"], data["label"]

            # Apply transformations if provided
            if self.transform:
                image = self.transform(image)

            return image, label

    # Data preparation
    transform = Compose([
        Resize((224, 224)),  # Resize to ViT input size
        ToTensor(),  # Convert image to PyTorch Tensor
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    # Load SVHN dataset using the Hugging Face datasets library
    svhn_dataset = load_dataset('svhn', 'cropped_digits')
    
    # Wrap the training and test datasets with the custom class
    train_dataset = SVHNDataset(svhn_dataset["train"], transform=transform)
    val_dataset = SVHNDataset(svhn_dataset["test"], transform=transform)
        
    # DataLoader setup
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    testloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    classes = tuple(str(i) for i in range(10))  # SVHN classes are digits from 0 to 9

    return trainloader, testloader, classes


def cifar10_224(n_test, subsample=False):
    batch_size = 512
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to ViT input size
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    train_dataset = datasets.CIFAR10(root="./data/CIFAR10/", train=True, transform=transform, download=True)
    val_dataset = datasets.CIFAR10(root="./data/CIFAR10/", train=False, transform=transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,num_workers=8)

    return train_loader, val_loader, ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    a = cifar10()
    print(a)