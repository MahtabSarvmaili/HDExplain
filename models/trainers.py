import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def targets(self):
        return self.y
    
class ClassifierTrainer(object):
    def __init__(self, model):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.01)

    def train(self, model, dataloader, name="", save=True, epochs=100):
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(dataloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        if save:
            torch.save(model.state_dict(), "checkpoints/{0}.pt".format(name))

        print('Finished Training')