import torch
import torch.nn as nn

class NaiveLaplaceKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, initial_gridsize, addbias=True):
        super(NaiveLaplaceKANLayer, self).__init__()
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        # Learnable gridsize parameter
        self.gridsize_param = nn.Parameter(torch.tensor(initial_gridsize, dtype=torch.float32))

        # Laplace coefficients as a learnable parameter with Xavier initialization
        self.laplacecoeffs = nn.Parameter(torch.empty(2, outdim, inputdim, initial_gridsize))
        nn.init.xavier_uniform_(self.laplacecoeffs)

        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        gridsize = torch.clamp(self.gridsize_param, min=1).round().int()
        xshp = x.shape
        outshape = xshp[:-1] + (self.outdim,)
        x = torch.reshape(x, (-1, self.inputdim))

        # Create a grid of lambda values
        lambdas = torch.reshape(torch.linspace(0.1, 1., gridsize, device=x.device), (1, 1, 1, gridsize))
        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))

        # Exponential functions for Laplace transform
        exp_neg = torch.exp(-lambdas * xrshp)
        exp_pos = torch.exp(lambdas * xrshp)

        # Applying Laplace coefficients to the input
        y = torch.sum(exp_neg * self.laplacecoeffs[0:1, :, :, :gridsize], (-2, -1))
        y += torch.sum(exp_pos * self.laplacecoeffs[1:2, :, :, :gridsize], (-2, -1))

        if self.addbias:
            y += self.bias
        y = torch.reshape(y, outshape)
        return y
