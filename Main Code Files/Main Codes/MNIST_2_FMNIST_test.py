'''
Filename: MNIST_2_FMNIST_test.py
Description: Simulation testing for continual learning in pattern recognition tasks.
Usage: For academic research purposes only.
Note: Import device measurement data independently (memristor conductance; peak current data of GMCs;
    sigma1, sigma2, k)
    When switching device data,it is necessary to adjust the hyperparameters
    (lr, coefficient k, grid size, grid range and etc.) accordingly.
Characterization of Gaussian-like Basis Functions:
                                    f(x) = c * exp{-[(x-center_i)/(sigma/k)]^2}
    c: Weight coefficient of a basis function based on differential pair mechanism;
    x: Network input, represented by gate voltage;
    center_i: Center of the i-th basis function;
    sigma: When x ≤ center_i, sigma = sigma1; when x > center_i, sigma = sigma2;
    k: Factor controlling the scaling of the basis function width.
'''

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from openpyxl import load_workbook
import torch.nn.functional as F
import numpy as np

class KANLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, grid_size=4, spline_order=3, scale_noise=0.1,
                 scale_base=1.0, scale_spline=1.0, base_activation=torch.nn.ReLU, grid_eps=0.02,
                 grid_range=[-1, 1], spline_weight_init_scale=0.1, sigma_left=0.745, sigma_right=0.787,):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_weight_init_scale = spline_weight_init_scale
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size)
        self.register_buffer("grid", grid)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, grid_size))
        self.sigma_left = sigma_left
        self.sigma_right = sigma_right
        self.gamma = (grid_range[1] - grid_range[0]) / (grid_size - 1)
        self.a = torch.pi * self.gamma
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        torch.nn.init.trunc_normal_(self.spline_weight, mean=0, std=self.spline_weight_init_scale)

    def PrewiseRadialBasisFunction(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        bases = torch.zeros(x.size(0), self.in_features, self.grid_size, device=x.device)
        for i in range(self.grid_size):
            center = grid[i]
            sigma = torch.where(x <= center, self.sigma_left, self.sigma_right)
            bases[:, :, i] = torch.exp(-((x - center) ** 2) / (sigma ** 2))
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size,
        )
        return bases.contiguous()

    def forward(self, x: torch.Tensor):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.PrewiseRadialBasisFunction(x).view(x.size(0), -1),
                                 self.spline_weight.view(self.out_features, -1))
        return base_output + spline_output # After the setup, the network includes residual connections based on the memristor differential pairs.
        # return spline_output # After the setup, the network only contains the GMC arrays.

class KAN(torch.nn.Module):
    def __init__(self, layers_hidden, grid_size=4, spline_order=3, scale_noise=0.1, scale_base=1.0,
                 scale_spline=1.0, base_activation=torch.nn.ReLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(KAN, self).__init__()
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(in_features, out_features, grid_size=grid_size, spline_order=spline_order,
                          scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                          base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range)
            )
    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

# Load MNIST
transform_mnist = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset_mnist = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform_mnist)
trainloader_mnist = DataLoader(trainset_mnist, batch_size=64, shuffle=True)

# Load Fashion-MNIST
transform_fmnist = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((-0.5,), (0.5,))
])
trainset_fmnist = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_fmnist)
trainloader_fmnist = DataLoader(trainset_fmnist, batch_size=64, shuffle=True)

model = KAN([28 * 28, 100, 10]) # Define network size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-4) # The actual hyperparameters are specified in the subsequent code
criterion = nn.CrossEntropyLoss()

def calculate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def save_to_excel(data, filename, sheet_name, startcol):
    if os.path.exists(filename):
        book = load_workbook(filename)
        writer = pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='overlay')
    else:
        writer = pd.ExcelWriter(filename, engine='openpyxl')
    data.to_excel(writer, sheet_name=sheet_name, index=False, header=False, startcol=startcol)
    writer.close()

'''Please import the actually measured device (memristors & GMCs) data manually.'''
# Load memristor conductance data
df = pd.read_excel(r'E:\download\GMCSimu-main\GMCSimu-main\memristor conductance.xlsx', sheet_name='sheet_name', usecols='A', header=None)
memData = df.to_numpy().flatten()
memData = (memData - min(memData)) / (max(memData) - min(memData))
memDiffMat = np.subtract.outer(memData, memData)
mem_synapse = np.unique(memDiffMat.flatten())
synapse = torch.tensor(mem_synapse)
# Load GMC peak current data
df = pd.read_excel(r'E:\download\GMCSimu-main\GMCSimu-main\GMC peak current.xlsx', sheet_name='sheet_name', usecols='A', header=None)
aatData = df.to_numpy().flatten()
aatData = (aatData - min(aatData)) / (max(aatData) - min(aatData))
aatDiffMat = np.subtract.outer(aatData, aatData)
c_data = np.unique(aatDiffMat.flatten())
spline_coef = torch.tensor(c_data)

def changeMEM(model, synapse):
    synapse = synapse.to(next(model.parameters()).device)
    for name, param in model.named_parameters():
        if not 'spline_weight' in name:
            tensor = param.data
            diff = torch.abs(tensor.reshape(-1, 1) - synapse.reshape(1, -1))
            index = torch.argmin(diff, dim=1)
            new_tensor = synapse[index].reshape(tensor.shape)
            param.data.copy_(new_tensor)

def changeAAT(model, spline_coef):
    spline_coef = spline_coef.to(next(model.parameters()).device)
    for name, param in model.named_parameters():
        if 'spline_weight' in name:
            tensor = param.data
            diff = torch.abs(tensor.reshape(-1, 1) - spline_coef.reshape(1, -1))
            index = torch.argmin(diff, dim=1)
            new_tensor = spline_coef[index].reshape(tensor.shape)
            param.data.copy_(new_tensor)

train_accuracies_mnist = []
train_accuracies_fmnist = []

for epoch in range(20):
    if epoch < 10:
        loader = trainloader_mnist  # MNIST is used in the first 10 epochs.
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1
    else:
        loader = trainloader_fmnist  # Fashion-MNIST is used in the subsequent 10 epochs.
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.03

    model.train()
    for images, labels in tqdm(loader):
        images = images.view(-1, 28 * 28).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        changeMEM(model, synapse) # training the memristor array (residual connections)
        changeAAT(model, spline_coef) # training the GMC array
    accuracy_mnist = calculate_accuracy(model, trainloader_mnist)
    accuracy_fmnist = calculate_accuracy(model, trainloader_fmnist)
    train_accuracies_mnist.append(accuracy_mnist)
    train_accuracies_fmnist.append(accuracy_fmnist)
    print(f"Epoch {epoch + 1}, MNIST Accuracy: {accuracy_mnist:.4f}, Fashion-MNIST Accuracy: {accuracy_fmnist:.4f}")

df_mnist = pd.DataFrame(train_accuracies_mnist, columns=['MNIST Accuracy'])
df_fmnist = pd.DataFrame(train_accuracies_fmnist, columns=['Fashion-MNIST Accuracy'])
save_to_excel(df_mnist, './accuracy.xlsx', sheet_name='MNIST_CL', startcol=0)
save_to_excel(df_fmnist, './accuracy.xlsx', sheet_name='FMNIST_CL', startcol=0)
print("Completed")

