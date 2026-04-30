'''
Filename: GKAN_1DRG.py
Description: 1D regression simulation test.
Usage: For academic research purposes only.
Note: Import device measurement data independently (memristor conductance; peak current data of GMCs;
    sigma1, sigma2 and k)
    When switching device data,it is necessary to adjust the hyperparameters
    (lr, weight_decay, coefficient k, grid size, grid range and etc.) accordingly.
Characterization of Gaussian-like Basis Functions:
                                    f(x) = c * exp{-[(x-center_i)/(sigma/k)]^2}
    c: Weight coefficient of a basis function based on differential pair mechanism;
    x: Network input, represented by gate voltage;
    center_i: Center of the i-th basis function;
    sigma: When x ≤ center_i, sigma = sigma1; when x > center_i, sigma = sigma2;
    k: Factor controlling the scaling of the basis function width.
'''

import torch
import torch.nn.functional as F
import math
import pandas as pd
import numpy as np

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

sigma1 = 0.2
sigma2 = 0.3
k = 2.0
sigma_left = sigma1 / k
sigma_right = sigma2 / k

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

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=100, # User-defined topological hyperparameter
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=False,
        base_activation=torch.nn.ReLU,
        grid_eps=0.02,
        grid_range=[-1, 1], # Default grid range
        spline_weight_init_scale=0.1,
        sigma_left=0.1,
        sigma_right=0.1,
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_weight_init_scale = spline_weight_init_scale
        # self.spline_order = spline_order
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size)
        self.register_buffer("grid", grid)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        # self.spline_weight = torch.nn.Parameter(
        #     torch.Tensor(out_features, in_features, grid_size + spline_order)
        # )
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size)
        )
        self.sigma_left = sigma_left
        self.sigma_right = sigma_right
        self.gamma = (grid_range[1] - grid_range[0]) / (grid_size - 1)
        self.a = torch.pi * self.gamma
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        torch.nn.init.trunc_normal_(self.spline_weight, mean=0, std=self.spline_weight_init_scale)

    def PrewiseRadialBasisFunction(self, x: torch.Tensor):
        '''
        :param x:
        :return: Gaussian-like Basis Function
        '''
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

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.PrewiseRadialBasisFunction(x).transpose(
            0, 1
        )
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(
            A, B
        ).solution
        result = solution.permute(
            2, 0, 1
        )
        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def scaled_spline_weight_clamped(self):
        return torch.clamp(self.scaled_spline_weight, min=-1, max=1)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.PrewiseRadialBasisFunction(x).view(x.size(0), -1),
            self.scaled_spline_weight_clamped().view(self.out_features, -1),
        )
        return base_output + spline_output # After the setup, the network includes residual connections based on the memristor differential pairs.
        # return spline_output # After the setup, the network only contains the GMC arrays.

class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=100, # User-defined topological hyperparameter
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.ReLU,
        grid_eps=0.02,
        grid_range=[-1, 1], # Default
        sigma_left=0.1,
        sigma_right=0.1,
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    sigma_left=sigma_left,
                    sigma_right=sigma_right,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

import os
from openpyxl import load_workbook
# def export_spline_weights(model, filename, sheet_name):
#     spline_weights = model.layers[-1].spline_weight.detach().cpu().numpy()
#     reshaped_weights = spline_weights.reshape(spline_weights.shape[2], -1)
#     df = pd.DataFrame(reshaped_weights)
#     if os.path.exists(filename):
#         book = load_workbook(filename)
#         writer = pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='replace')
#         writer.book = book
#     else:
#         writer = pd.ExcelWriter(filename, engine='openpyxl')
#     df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
#     writer.close()

import numpy as np
import torch
import matplotlib.pyplot as plt

datasets = []
n_peak = 5
n_num_per_peak = 1000
n_sample = n_peak * n_num_per_peak
x_grid = torch.linspace(-1, 1, steps=n_sample)
x_centers = 2 / n_peak * (np.arange(n_peak) - n_peak / 2 + 0.5)
x_sample = torch.stack(
    [torch.linspace(-1 / n_peak, 1 / n_peak, steps=n_num_per_peak) + center for center in x_centers]).reshape(-1, )
y = 0.
for center in x_centers:
    y += torch.exp(-(x_grid - center) ** 2 * 300)
y_sample = 0.
for center in x_centers:
    y_sample += torch.exp(-(x_sample - center) ** 2 * 300)

plt.plot(x_grid.detach().numpy(), y.detach().numpy())
plt.scatter(x_sample.detach().numpy(), y_sample.detach().numpy())
plt.show()

plt.subplots(1, 5, figsize=(15, 2))
plt.subplots_adjust(wspace=0, hspace=0)


for i in range(1,6):
    plt.subplot(1,5,i)
    group_id = i - 1
    plt.plot(x_grid.detach().numpy(), y.detach().numpy(), color='black', alpha=0.1)
    plt.scatter(x_sample[group_id*n_num_per_peak:(group_id+1)*n_num_per_peak].detach().numpy(), y_sample[group_id*n_num_per_peak:(group_id+1)*n_num_per_peak].detach().numpy(), color="black", s=2)
    plt.xlim(-1,1)
    plt.ylim(-1,2)
plt.show()

import torch
ys = []
model = KAN([1, 1],grid_size=100, scale_noise=0.1,
            sigma_left=sigma_left, sigma_right=sigma_right,
            )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

criterion = torch.nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-4)

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
epoch_times = []

inputs = torch.randn(32, 1)
targets = torch.randint(0, 1, (32,))

for group_id in range(n_peak):
    dataset = {}
    dataset['train_input'] = x_sample[group_id * n_num_per_peak:(group_id + 1) * n_num_per_peak][:, None]
    dataset['train_label'] = y_sample[group_id * n_num_per_peak:(group_id + 1) * n_num_per_peak][:, None]
    dataset['test_input'] = x_sample[group_id * n_num_per_peak:(group_id + 1) * n_num_per_peak][:, None]
    dataset['test_label'] = y_sample[group_id * n_num_per_peak:(group_id + 1) * n_num_per_peak][:, None]
    dataset['train_input'] = dataset['train_input'].to(device)
    dataset['train_label'] = dataset['train_label'].to(device)
    dataset['test_input'] = dataset['test_input'].to(device)
    dataset['test_label'] = dataset['test_label'].to(device)

    changeAAT(model, spline_coef)
    model.train()
    for i in range(200):
        outputs = model(dataset['train_input'])
        loss = criterion(outputs, dataset['train_label'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        changeMEM(model, synapse)
        changeAAT(model, spline_coef)
    print(f'Loss: {loss.item()}')
    y_pred = model(x_grid[:, None].to(device))
    ys.append(y_pred.detach().cpu().numpy()[:, 0])
plt.subplots(1, 5, figsize=(15, 2))
plt.subplots_adjust(wspace=0, hspace=0)

for i in range(1,6):
    plt.subplot(5,1,i)
    group_id = i - 1
    plt.plot(x_grid.detach().numpy(), y.detach().numpy(), color='black', alpha=0.1)
    plt.plot(x_grid.detach().numpy(), ys[i-1], color='black')
    plt.xlim(-1,1)
    plt.ylim(-1,2)
plt.show()
df = pd.DataFrame(x_grid.detach().numpy(), columns=['X_grid'])
df['Ideal'] = y.detach().numpy()
for i in range(5):
    df[f'Predicted_{i+1}'] = ys[i]
with pd.ExcelWriter('./continue.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='1D_RG', index=False)
print("Completed")
