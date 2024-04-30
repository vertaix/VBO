import os

import pandas as pd
import numpy as np

import torch
import gpytorch
import botorch


test_path = "../data/DB_10042.csv"
train_path = "../data/Label1000.csv"

test_df = pd.read_csv(test_path, index_col=0).iloc[:, 25:]
train_df = pd.read_csv(train_path, index_col=0)

test_uid = list(test_df.index)
train_uid = list(train_df.index)

train_ind = []

for ind in train_uid:
    assert ind in test_uid
    train_ind.append(test_uid.index(ind))

csv_path = "../data/DB_10042.csv"
node_sim_path = "../data/DB_10042_node_sims2.npy"
linker_sim_path = "../data/DB_10042_linker_sims2.npy"
distributional_sim_path = "../data/DB_10042_distributional_sims.npy"
fixed_covariance_matrix_path = "../data/DB_10042_fixed_covariance_matrix.npy"

node_covariance_matrix = torch.tensor(np.load(node_sim_path))
linker_covariance_matrix = torch.tensor(np.load(linker_sim_path))
dist_covariance_matrix = torch.tensor(np.load(distributional_sim_path))
fixed_covariance_matrix = np.load(fixed_covariance_matrix_path)

features = torch.tensor(test_df.values)
numerical_features = features[:, :6]
numerical_features = (
    numerical_features - numerical_features.mean(dim=0)
) / numerical_features.std(dim=0)


class NumericalKernel(gpytorch.kernels.Kernel):
    def __init__(self, base_kernel, numerical_features):
        super().__init__()
        self.base_kernel = base_kernel
        self.numerical_features = numerical_features
        
    def forward(self, x1_ind, x2_ind, **params):
        return self.base_kernel(
            self.numerical_features[..., x1_ind.flatten(), :],
            self.numerical_features[..., x2_ind.flatten(), :],
            **params
        )
    

class PrecomputedKernel(gpytorch.kernels.Kernel):
    def __init__(self, covariance_matrix):
        super().__init__()
        self.covariance_matrix = covariance_matrix
        
    def forward(self, x1_ind, x2_ind, **params):
        return self.covariance_matrix[..., x1_ind.flatten(), :][..., :, x2_ind.flatten()]


class GPModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    _num_outputs = 1
    
    def __init__(self, train_ind, train_y, likelihood):
        super().__init__(train_ind, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = (
            gpytorch.kernels.ScaleKernel(
                PrecomputedKernel(node_covariance_matrix)
            )
            + gpytorch.kernels.ScaleKernel(
                PrecomputedKernel(linker_covariance_matrix)
            )
            + gpytorch.kernels.ScaleKernel(
                NumericalKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=6),
                    numerical_features,
                )
            ) + gpytorch.kernels.ScaleKernel(
                PrecomputedKernel(dist_covariance_matrix)
            )
        )
        
    def forward(self, ind):
        mean = self.mean_module(ind)
        covar = self.covar_module(ind)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


def normalize_y_values(y):
    return (y - y.mean()) / y.std()


def fit_gp_model(this_train_ind, this_train_y, num_train_iters=1000):
    if torch.unique(this_train_y).numel() > 1:
        this_train_y = normalize_y_values(this_train_y)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(this_train_ind, this_train_y, likelihood)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()

    for i in range(num_train_iters):
        optimizer.zero_grad()

        output = model(this_train_ind)
        loss = -mll(output, this_train_y)

        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    
    return model, likelihood


def calculate_vendi_score(kernel_mat):
    _, s, _ = np.linalg.svd(kernel_mat / kernel_mat.shape[0])

    return np.exp(-(s * np.log(s)).sum())
