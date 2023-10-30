import torch
import torchvision
import torchvision.transforms as transforms


def cifar10(n_test=10, random_state=42, subsample=False):
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


def ocea(n_test, random_state=42, subsample=False):
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