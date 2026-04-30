'''
Filename: GKAN_PDE_simulation.py
Description: PDE solving simulation test.
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
import torch.nn as nn
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch import autograd
from tqdm import tqdm

'''Please import the actually measured device (memristors & GMCs) data manually.'''
# Load memristor conductance data
df = pd.read_excel('memristor conductance.xlsx', sheet_name='sheet_name', usecols='usecol', nrows='nrows', header=None)
memData = df.to_numpy().flatten()
memData = (memData - min(memData)) / (max(memData) - min(memData))
memDiffMat = np.subtract.outer(memData, memData)
mem_synapse = np.unique(memDiffMat.flatten())
synapse = torch.tensor(mem_synapse)
# Load GMC peak current data
df = pd.read_excel('GMC peak current.xlsx', sheet_name='sheet_name', usecols='usecol', nrows='nrows', header=None)
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


class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=3,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=False,
            base_activation=torch.nn.ReLU,
            grid_range=[-1, 1],  # Default
            spline_weight_init_scale=0.1,
            sigma_left='sigma1 / k',  # Import manually
            sigma_right='sigma2 / k',  # Import manually
    ):
        super(KANLinear, self).__init__()
        # Set sigmas on the left and right sides of the peak axis
        self.sigma_left = sigma_left
        self.sigma_right = sigma_right
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_weight_init_scale = spline_weight_init_scale
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size)
        self.register_buffer("grid", grid)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size)
        )
        self.denominator = (grid_range[1] - grid_range[0]) / (grid_size - 1)
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
        # Assign configuration parameters
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()  # Instantiate base activation layer
        self.reset_parameters()  # Initialize parameters

    def reset_parameters(self):
        """Parameter initialization"""
        # Initialize base weights using He initialization
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        # Initialize spline weights using truncated normal distribution
        torch.nn.init.trunc_normal_(self.spline_weight, mean=0, std=self.spline_weight_init_scale)

    def PrewiseRadialBasisFunction(self, x: torch.Tensor):
        """
        Compute piecewise radial basis functions for the given input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Piecewise radial basis tensor of shape
                          (batch_size, in_features, grid_size).
        """
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
        A = self.PrewiseRadialBasisFunction(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.PrewiseRadialBasisFunction(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output  # After the setup, the network includes residual connections based on the memristor differential pairs.
        # return spline_output # After the setup, the network only contains the GMC arrays.

class KAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            grid_size=5, # User-defined topological hyperparameter
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.ReLU,
            grid_range=[-1, 1], # Default
    ):
        super(KAN, self).__init__()
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features, out_features,
                    grid_size=grid_size,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

class MLP(nn.Module):
    def __init__(
            self,
            layers_hidden,
            activation=nn.ReLU,
            init_method='he',
            dropout_prob=0.0
    ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation()
        for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(nn.Linear(in_dim, out_dim))
            if dropout_prob > 0:
                self.layers.append(nn.Dropout(dropout_prob))
        self._initialize_weights(init_method)

    def _initialize_weights(self, method):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if method == 'xavier':
                    nn.init.xavier_normal_(layer.weight)
                elif method == 'he':
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                else:
                    raise ValueError(f"Unsupported init method: {method}")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1 and not isinstance(layer, nn.Dropout):
                x = self.activation(x)
        return x


dim = 2
np_i = 51  # number of interior points (along each dimension)
np_b = 51  # number of boundary points (along each dimension)
ranges = [-1, 1]

def batch_jacobian(func, x, create_graph=False):
    # x in shape (Batch, Length)
    def _func_sum(x):
        return func(x).sum(dim=0)

    return autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1, 0, 2)

# define solution
sol_fun = lambda x: torch.sin(torch.pi * x[:, [0]]) * torch.sin(torch.pi * x[:, [1]])
source_fun = lambda x: -2 * torch.pi ** 2 * torch.sin(torch.pi * x[:, [0]]) * torch.sin(torch.pi * x[:, [1]])

# interior
sampling_mode = 'mesh'  # 'radnom' or 'mesh'
x_mesh = torch.linspace(ranges[0], ranges[1], steps=np_i)
y_mesh = torch.linspace(ranges[0], ranges[1], steps=np_i)
X, Y = torch.meshgrid(x_mesh, y_mesh, indexing="ij")
if sampling_mode == 'mesh':
    # mesh
    x_i = torch.stack([X.reshape(-1, ), Y.reshape(-1, )]).permute(1, 0)
else:
    # random
    x_i = torch.rand((np_i ** 2, 2)) * 2 - 1

# boundary, 4 sides
helper = lambda X, Y: torch.stack([X.reshape(-1, ), Y.reshape(-1, )]).permute(1, 0)
xb1 = helper(X[0], Y[0])
xb2 = helper(X[-1], Y[0])
xb3 = helper(X[:, 0], Y[:, 0])
xb4 = helper(X[:, 0], Y[:, -1])
x_b = torch.cat([xb1, xb2, xb3, xb4], dim=0)

alpha = 0.01
log = 1

grids = 5
steps = 250

kan_losses = []
mlp1_losses = []
mlp2_losses = []

kan_losses_l2 = []
mlp1_losses_l2 = []
mlp2_losses_l2 = []

model_kan = KAN([2, 10, 1], grid_size=grids)
model_mlp_1 = MLP([2, 10, 1])
model_mlp_2 = MLP([2, 100, 100, 100, 1])

def train():
    optimizer_kan = torch.optim.AdamW(model_kan.parameters(), lr=0.01)
    optimizer_mlp_1 = torch.optim.AdamW(model_mlp_1.parameters(), lr=0.01)
    optimizer_mlp_2 = torch.optim.AdamW(model_mlp_2.parameters(), lr=0.01)

    pbar = tqdm(range(steps), desc='description', ncols=100)

    changeMEM(model_mlp_1, spline_coef)
    changeMEM(model_mlp_2, spline_coef)

    changeMEM(model_kan, spline_coef)
    changeAAT(model_kan, spline_coef)

    for _ in pbar:
        def closure_mlp_1():
            global pde_loss_mlp_1, bc_loss_mlp_1
            optimizer_mlp_1.zero_grad()
            # interior loss
            sol = sol_fun(x_i)
            sol_D1_fun_mlp_1 = lambda x: batch_jacobian(model_mlp_1, x, create_graph=True)[:, 0, :]
            sol_D1 = sol_D1_fun_mlp_1(x_i)
            sol_D2_mlp_1 = batch_jacobian(sol_D1_fun_mlp_1, x_i, create_graph=True)[:, :, :]
            lap_mlp_1 = torch.sum(torch.diagonal(sol_D2_mlp_1, dim1=1, dim2=2), dim=1, keepdim=True)
            source = source_fun(x_i)
            pde_loss_mlp_1 = torch.mean((lap_mlp_1 - source) ** 2)

            # boundary loss
            bc_true = sol_fun(x_b)
            bc_pred_mlp_1 = model_mlp_1(x_b)
            bc_loss_mlp_1 = torch.mean((bc_pred_mlp_1 - bc_true) ** 2)

            loss_mlp_1 = alpha * pde_loss_mlp_1 + bc_loss_mlp_1
            loss_mlp_1.backward()
            return loss_mlp_1

        optimizer_mlp_1.step(closure_mlp_1)
        changeMEM(model_mlp_1, synapse)
        sol = sol_fun(x_i)
        loss_mlp_1 = alpha * pde_loss_mlp_1 + bc_loss_mlp_1
        l2_mlp_1 = torch.mean((model_mlp_1(x_i) - sol) ** 2)

        def closure_mlp_2():
            global pde_loss_mlp_2, bc_loss_mlp_2
            optimizer_mlp_2.zero_grad()
            # interior loss
            sol = sol_fun(x_i)
            sol_D1_fun_mlp_2 = lambda x: batch_jacobian(model_mlp_2, x, create_graph=True)[:, 0, :]
            sol_D1 = sol_D1_fun_mlp_2(x_i)
            sol_D2_mlp_2 = batch_jacobian(sol_D1_fun_mlp_2, x_i, create_graph=True)[:, :, :]
            lap_mlp_2 = torch.sum(torch.diagonal(sol_D2_mlp_2, dim1=1, dim2=2), dim=1, keepdim=True)
            source = source_fun(x_i)
            pde_loss_mlp_2 = torch.mean((lap_mlp_2 - source) ** 2)

            # boundary loss
            bc_true = sol_fun(x_b)
            bc_pred_mlp_2 = model_mlp_2(x_b)
            bc_loss_mlp_2 = torch.mean((bc_pred_mlp_2 - bc_true) ** 2)

            loss_mlp_2 = alpha * pde_loss_mlp_2 + bc_loss_mlp_2
            loss_mlp_2.backward()
            return loss_mlp_2

        optimizer_mlp_2.step(closure_mlp_2)
        changeMEM(model_mlp_2, synapse)
        sol = sol_fun(x_i)
        l2_mlp_2 = torch.mean((model_mlp_2(x_i) - sol) ** 2)

        def closure_kan():
            global pde_loss_kan, bc_loss_kan
            optimizer_kan.zero_grad()
            # interior loss
            sol = sol_fun(x_i)
            sol_D1_fun_kan = lambda x: batch_jacobian(model_kan, x, create_graph=True)[:, 0, :]
            sol_D1_kan = sol_D1_fun_kan(x_i)
            sol_D2_kan = batch_jacobian(sol_D1_fun_kan, x_i, create_graph=True)[:, :, :]
            lap_kan = torch.sum(torch.diagonal(sol_D2_kan, dim1=1, dim2=2), dim=1, keepdim=True)
            source = source_fun(x_i)
            pde_loss_kan = torch.mean((lap_kan - source) ** 2)
            # boundary loss
            bc_true = sol_fun(x_b)
            bc_pred_kan = model_kan(x_b)
            bc_loss_kan = torch.mean((bc_pred_kan - bc_true) ** 2)
            loss_kan = alpha * pde_loss_kan + bc_loss_kan
            loss_kan.backward()
            return loss_kan

        optimizer_kan.step(closure_kan)
        changeMEM(model_kan, synapse)
        changeAAT(model_kan, spline_coef)
        sol = sol_fun(x_i)
        l2_kan = torch.mean((model_kan(x_i) - sol) ** 2)

        if _ % log == 0:
            pbar.set_description(
                f"mlp_1 loss: {l2_mlp_1.cpu().detach().numpy()} | mlp_2 loss: {l2_mlp_2.cpu().detach().numpy()} | kan loss: {l2_kan.cpu().detach().numpy()} ")

        mlp1_losses_l2.append(l2_mlp_1.cpu().detach().numpy())
        mlp2_losses_l2.append(l2_mlp_2.cpu().detach().numpy())
        kan_losses_l2.append(l2_kan.cpu().detach().numpy())

train()

plt.plot(mlp1_losses_l2, marker='x')
plt.plot(mlp2_losses_l2, marker='x')
plt.plot(kan_losses_l2, marker='o')
plt.yscale('log')
plt.xlabel('steps')
plt.legend(['mlp1 loss', 'mlp2 loss', 'kan loss'])
dt = np.array([np.array(mlp1_losses_l2).reshape(-1), np.array(mlp2_losses_l2).reshape(-1), np.array(kan_losses_l2).reshape(-1)]).T
dt = pd.DataFrame(dt, columns=['mlp1_loss', 'mlp2_loss', 'kan_loss'])
dt.to_csv('./PDE_losses_l2.csv')