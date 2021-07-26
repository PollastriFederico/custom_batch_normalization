import torch
from torch import nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from matplotlib import pyplot as plt

import time

from model import LeNet

root_path = '/homes/fpollastri/data/'


class Trainer:
    def __init__(self, learning_rate, bn, workers):
        # Hyper-parameters
        self.learning_rate = learning_rate
        self.bn = bn
        self.workers = workers
        self.batch_size = 2048
        # Neural Network
        self.n = LeNet(bn=self.bn).to('cuda')

        # Data Loaders, images are padded to 32x32 as described in the LeNet paper, and normalized (mean=0 & std=1), no Data Augmentation is performed.
        train_dataset = datasets.MNIST(root_path, train=True, download=True, transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        test_dataset = datasets.MNIST(root_path, train=False, download=True, transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

        self.data_loader = DataLoader(train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=self.workers,
                                      pin_memory=True)
        self.test_data_loader = DataLoader(test_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=self.workers,
                                           drop_last=False,
                                           pin_memory=True)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.n.parameters()), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', verbose=True)

        # variables for visualization purposes
        self.last_cm = [[]]
        self.metrics = []
        self.losses = []

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.n.train()
            epoch_losses = []
            start_time = time.time()
            for i, (x, target) in enumerate(self.data_loader):
                x = x.to('cuda')
                target = target.to('cuda', torch.long)

                output = self.n(x)
                loss = self.criterion(output, target)
                epoch_losses.append(loss.item())
                # compute gradient and optimizer step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            balanced_acc = self.eval()
            print(f'Epoch: {epoch} | loss: {np.mean(epoch_losses):.4f} | Balanced Accuracy: {balanced_acc:.3f} | time: {time.time() - start_time}\n')

            self.metrics.append(balanced_acc)
            self.losses.append(np.mean(epoch_losses))
            # update scheduler, and STOP the training process when the learning rate gets reduced beyond a certain threshold
            self.scheduler.step(np.mean(epoch_losses))
            if self.learning_rate // self.optimizer.param_groups[0]['lr'] > 10**2:
                print("Training process will be stopped now due to the low learning rate reached")
                return

    def eval(self):

        with torch.no_grad():
            self.n.eval()

            # initialize arrays to store predictions and ground truths
            preds = np.zeros(len(self.test_data_loader.dataset))
            gts = np.zeros(len(self.test_data_loader.dataset))
            for i, (x, target) in enumerate(self.test_data_loader):
                # compute output
                x = x.to('cuda')
                output = torch.squeeze(self.n(x))
                target = target.to('cuda', torch.long)
                check_output = torch.argmax(output, dim=-1)
                gts[i * self.test_data_loader.batch_size:i * self.test_data_loader.batch_size + len(target)] = target.to('cpu')
                preds[i * self.test_data_loader.batch_size:i * self.test_data_loader.batch_size + len(target)] = check_output.to('cpu')

            # confusion matrix is saved, after the training process is complete the last one obtained will be printed
            self.last_cm = confusion_matrix(gts, preds)
            # balanced accuracy is chosen as the metric to evaluate this multiclass task
            balanced_acc = balanced_accuracy_score(gts, preds)
        return balanced_acc

    def final_print(self):
        print(self.last_cm)
        plt.plot(self.losses, 'r')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()
        plt.plot(self.metrics, 'g')
        plt.xlabel("Epochs")
        plt.ylabel("Balanced Accuracy")
        plt.show()
