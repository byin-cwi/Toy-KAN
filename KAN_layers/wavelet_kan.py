import torch
import torch.nn as nn
import numpy as np

class NaiveWaveletKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, initial_gridsize, addbias=True):
        super(NaiveWaveletKANLayer, self).__init__()
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        # Learnable gridsize parameter
        self.gridsize_param = nn.Parameter(torch.tensor(initial_gridsize, dtype=torch.float32))

        # Wavelet coefficients as a learnable parameter with Xavier initialization
        self.waveletcoeffs = nn.Parameter(torch.empty(2, outdim, inputdim, initial_gridsize))
        nn.init.xavier_uniform_(self.waveletcoeffs)

        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        gridsize = torch.clamp(self.gridsize_param, min=1).round().int()
        xshp = x.shape
        outshape = xshp[:-1] + (self.outdim,)
        x = torch.reshape(x, (-1, self.inputdim))

        # Create a range of scales and translations for the wavelet
        scales = torch.linspace(1, gridsize, gridsize, device=x.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        translations = torch.linspace(0, 1, gridsize, device=x.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Morlet wavelet calculations
        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        u = (xrshp - translations) * scales
        real = torch.cos(np.pi*u) * torch.exp(-u**2 /2.)
        imag = torch.sin(np.pi*u) * torch.exp(-u**2 /2.)

        # Apply wavelet coefficients to the wavelet transform outputs
        y_real = torch.sum(real * self.waveletcoeffs[0:1, :, :, :gridsize], (-2, -1))
        y_imag = torch.sum(imag * self.waveletcoeffs[1:2, :, :, :gridsize], (-2, -1))
        y = y_real + y_imag

        if self.addbias:
            y += self.bias
        y = torch.reshape(y, outshape)
        return y
