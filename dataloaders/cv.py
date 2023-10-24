import torch
import torchvision
import torchvision.transforms as transforms


def cifar10(n_test_sample, random_state=42):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    
    tens = list(range(0, len(trainset), 10))

    trainset_1 = torch.utils.data.Subset(trainset, tens)

    trainloader = torch.utils.data.DataLoader(
        trainset_1, batch_size=128, shuffle=False)
    

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    G = torch.Generator()
    G.manual_seed(random_state)
    sampler = torch.utils.data.RandomSampler(testset, replacement=False, 
                                             num_samples=n_test_sample, 
                                             generator=G)
    
    testloader = torch.utils.data.DataLoader(
    testset, batch_size=10, sampler=sampler, generator=G)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes