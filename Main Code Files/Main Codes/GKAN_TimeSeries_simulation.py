'''
Filename: GKAN_TimeSeries_simulation.py
Description: Time-series forecasting simulation test.
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
#df = pd.read_excel(r'E:\download\GMCSimu-main\GMCSimu-main\memristor conductance.xlsx', sheet_name='sheet_name', usecols='usecol', nrows='nrows', header=None)
df = pd.read_excel(r'E:\download\GMCSimu-main\GMCSimu-main\memristor conductance.xlsx', sheet_name='sheet_name', usecols='A', header=None)
memData = df.to_numpy().flatten()
memData = (memData - min(memData)) / (max(memData) - min(memData))
memDiffMat = np.subtract.outer(memData, memData)
mem_synapse = np.unique(memDiffMat.flatten())
synapse = torch.tensor(mem_synapse)

# Load GMC peak current data
#df = pd.read_excel(r'E:\download\GMCSimu-main\GMCSimu-main\GMC peak current.xlsx', sheet_name='sheet_name', usecols='usecol', nrows='nrows', header=None)
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

sigma1 = 0.2
sigma2 = 0.2
k = 2.0

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
            sigma_left= sigma1 / k, # Import manually
            sigma_right= sigma2 / k, # Import manually
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
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))  # Base function weights
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size)  # Spline function weights
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()  # Instantiate base activation layer
        self.reset_parameters()  # Initialize parameters

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

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.PrewiseRadialBasisFunction(x).transpose(0, 1)  # Adjust dimensions for matrix computation
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution  # Solve the linear system
        return solution.permute(2, 0, 1).contiguous()  # Reorder dimensions

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
            grid_size=5,  # User-defined topological hyperparameter
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.ReLU,
            grid_range=[-1, 1], # Default
    ):
        super(KAN, self).__init__()
        self.layers = torch.nn.ModuleList()

        # Build KANLinear layers
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
if __name__ == '__main__':
    windows_size = 20 #20

    model_KAN = KAN(
                layers_hidden=[windows_size, 5, 1],
                grid_size=10 #5
            )
    model_MLP_1 = MLP(layers_hidden=[windows_size, 5, 1])
    model_MLP_2 = MLP(layers_hidden=[windows_size, 50, 50, 1])
    optimizer_KAN = torch.optim.AdamW(model_KAN.parameters(), lr=0.01)
    optimizer_MLP_1 = torch.optim.AdamW(model_MLP_1.parameters(), lr=0.01)
    optimizer_MLP_2 = torch.optim.AdamW(model_MLP_2.parameters(), lr=0.01)
    criterion_kan = nn.MSELoss()
    criterion_mlp_1 = nn.MSELoss()
    criterion_mlp_2 = nn.MSELoss()

    def Mackey_Glass(T, a=0.2, b=0.1, c=10, tau=18):
      x=np.arange(0,T+tau+1,dtype=np.float64)
      for t in range(tau,tau+T):
        x[t+1]=x[t]+(-b*x[t]+a*x[t-tau]/(1+x[t-tau]**c))
      return x[tau:]

    data = Mackey_Glass(T=1000)
    def create_dataset(data, window_size=5):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size])
        return np.array(X), np.array(y)

    X, y = create_dataset(data, windows_size)
    split_ratio = 0.8
    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(2)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(2)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    epochs = 1000
    changeMEM(model_MLP_1, spline_coef)
    for epoch in range(epochs):
        model_MLP_1.train()
        # print(X_train_tensor.shape)
        outputs = model_MLP_1(X_train_tensor)
        loss_MLP_1 = criterion_mlp_1(outputs, y_train_tensor)
        optimizer_MLP_1.zero_grad()
        loss_MLP_1.backward()
        optimizer_MLP_1.step()
        # changeMEM(model_MLP_1, synapse)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss_MLP_1.item()}')

    model_MLP_1.eval()
    y_pre = model_MLP_1(X_train_tensor)
    y_pred_train_mlp_1 = y_pre.detach().numpy()
    x = y_pre.detach().numpy()[-windows_size:] # 取决于windows_size
    pred_1 = []
    for i in range(split_idx, len(data)-windows_size):
        pre = model_MLP_1(torch.Tensor(x).unsqueeze(0))
        y_pred_1 = pre.detach().numpy()
        x = np.append(x[1:], y_pred_1).reshape((windows_size, 1))
        pred_1.append(y_pred_1)

    pred_mlp_1_test = np.array(pred_1).squeeze(-1).squeeze(-1)

    rmse_mlp_1 = np.sqrt(np.mean((pred_mlp_1_test - y_test)**2))
    print(f'Test RMSE_2: {rmse_mlp_1}')

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(y_train, label='train_label', color='green')
    plt.plot(outputs.detach().numpy(), label='mlp_1_train_pred', color='yellow', linestyle='--')
    plt.title('Mackey-Glass Training Prediction (mlp_1)')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(y_test, label='test_label', color='blue')
    plt.plot(pred_mlp_1_test, label='mlp_1_pred', color='red', linestyle='--')
    plt.title('Mackey-Glass Testing Prediction (mlp_1)')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.tight_layout()
    plt.show()

    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(2)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(2)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    epochs = 1000
    changeMEM(model_MLP_2, spline_coef)
    for epoch in range(epochs):
        model_MLP_2.train()
        # print(X_train_tensor.shape)
        outputs = model_MLP_2(X_train_tensor)
        loss_MLP_2 = criterion_mlp_2(outputs, y_train_tensor)
        optimizer_MLP_2.zero_grad()
        loss_MLP_2.backward()
        optimizer_MLP_2.step()
        #changeMEM(model_MLP_2, synapse)
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss_MLP_2.item()}')
    model_MLP_2.eval()
    y_pre = model_MLP_2(X_train_tensor)
    y_pred_train_mlp_2 = y_pre.detach().numpy()
    x = y_pre.detach().numpy()[-windows_size:]
    pred_2 = []
    for i in range(split_idx, len(data)-windows_size):
        pre = model_MLP_2(torch.Tensor(x).unsqueeze(0))
        y_pred_2 = pre.detach().numpy()
        x = np.append(x[1:], y_pred_2).reshape((windows_size, 1))
        pred_2.append(y_pred_2)

    pred_mlp_2_test = np.array(pred_2).squeeze(-1).squeeze(-1)

    rmse_mlp_2 = np.sqrt(np.mean((pred_mlp_2_test - y_test)**2))
    print(f'Test RMSE_2: {rmse_mlp_2}')

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(y_train, label='train_label', color='green')
    plt.plot(outputs.detach().numpy(), label='mlp_2_train_pred', color='yellow', linestyle='--')
    plt.title('Mackey-Glass Training Prediction (mlp_2)')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(y_test, label='test_label', color='blue')
    plt.plot(pred_mlp_2_test, label='mlp_2_pred', color='red', linestyle='--')
    plt.title('Mackey-Glass Testing Prediction (mlp_2)')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Value')
    plt.legend()

    plt.tight_layout()
    plt.show()

    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(2)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(2)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    epochs = 1000
    changeMEM(model_MLP_2, spline_coef)
    for epoch in range(epochs):
        model_MLP_2.train()
        # print(X_train_tensor.shape)
        outputs = model_MLP_2(X_train_tensor)
        loss_MLP_2 = criterion_mlp_2(outputs, y_train_tensor)

        optimizer_MLP_2.zero_grad()
        loss_MLP_2.backward()
        optimizer_MLP_2.step()
        # changeMEM(model_MLP_2, synapse)
        # changeAAT(model_MLP, spline_coef)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss_MLP_2.item()}')

    model_MLP_2.eval()
    y_pre = model_MLP_2(X_train_tensor)
    y_pred_train_mlp_2 = y_pre.detach().numpy()
    x = y_pre.detach().numpy()[-windows_size:]
    pred_2 = []
    for i in range(split_idx, len(data)-windows_size):
        pre = model_MLP_2(torch.Tensor(x).unsqueeze(0))
        y_pred_2 = pre.detach().numpy()
        x = np.append(x[1:], y_pred_2).reshape((windows_size, 1))
        pred_2.append(y_pred_2)

    pred_mlp_2_test = np.array(pred_2).squeeze(-1).squeeze(-1)

    rmse_mlp_2 = np.sqrt(np.mean((pred_mlp_2_test - y_test)**2))
    print(f'Test RMSE_2: {rmse_mlp_2}')

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(y_train, label='train_label', color='green')
    plt.plot(outputs.detach().numpy(), label='mlp_2_train_pred', color='yellow', linestyle='--')
    plt.title('Mackey-Glass Training Prediction (mlp_2)')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(y_test, label='test_label', color='blue')
    plt.plot(pred_mlp_2_test, label='mlp_2_pred', color='red', linestyle='--')
    plt.title('Mackey-Glass Testing Prediction (mlp_2)')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Value')
    plt.legend()

    plt.tight_layout()
    plt.show()

    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(2)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(2)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    epochs = 1000
    changeMEM(model_KAN, synapse)
    changeAAT(model_KAN, spline_coef)
    for epoch in range(epochs):
        model_KAN.train()
        # print(X_train_tensor.shape)
        outputs = model_KAN(X_train_tensor)
        loss_KAN = criterion_kan(outputs, y_train_tensor)
        optimizer_KAN.zero_grad()
        loss_KAN.backward()
        optimizer_KAN.step()
        # changeMEM(model_KAN, synapse)
        # changeAAT(model_KAN, spline_coef)
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss_KAN.item()}')

    model_KAN.eval()
    y_pre = model_KAN(X_train_tensor)
    y_pred_train_kan = y_pre.detach().numpy()
    x = y_pre.detach().numpy()[-windows_size:]
    pred = []
    for i in range(split_idx, len(data) - windows_size):
        pre = model_KAN(torch.Tensor(x).unsqueeze(0))
        y_pred = pre.detach().numpy()
        x = np.append(x[1:], y_pred).reshape((windows_size, 1))
        pred.append(y_pred)

    pred_kan_test = np.array(pred).squeeze(-1).squeeze(-1)
    print(pred_kan_test.shape)

    rmse_kan = np.sqrt(np.mean((pred_kan_test - y_test) ** 2))
    print(f'Test RMSE_KAN: {rmse_kan}')  # 这里顺手把打印提示的RMSE_2改成了RMSE_KAN，方便区分

    plt.figure(figsize=(12, 10))

    # 第一个子图：训练集结果
    plt.subplot(2, 1, 1)
    plt.plot(y_train, label='train_label', color='green')
    plt.plot(outputs.detach().numpy(), label='kan_train_pred', color='yellow', linestyle='--')
    plt.title('Mackey-Glass Training Prediction (KAN)')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Value')
    plt.legend()

    # 第二个子图：测试集结果（这里加上了声明）
    plt.subplot(2, 1, 2)
    plt.plot(y_test, label='test_label', color='blue')
    plt.plot(pred_kan_test, label='kan_pred', color='red', linestyle='--')
    plt.title('Mackey-Glass Testing Prediction (KAN)')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Value')
    plt.legend()

    plt.tight_layout()
    plt.show()

    train_dt = np.array([y_train.reshape(-1), y_pred_train_mlp_1.reshape(-1),y_pred_train_mlp_2.reshape(-1), y_pred_train_kan.reshape(-1)]).T
    train_dt = pd.DataFrame(train_dt, columns=['label', 'mlp1', 'mlp2', 'kan'])
    train_dt.to_csv(f'./series_train.csv')
    test_dt = np.array([y_test.reshape(-1), pred_mlp_1_test.reshape(-1), pred_mlp_2_test.reshape(-1), pred_kan_test.reshape(-1)]).T
    test_dt = pd.DataFrame(test_dt, columns=['label', f'mlp_1={rmse_mlp_1}', f'mlp_2={rmse_mlp_2}', f'kan_{rmse_kan}'])
    test_dt.to_csv(f'./series_test.csv')