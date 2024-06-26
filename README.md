# FourierKAN, LaplaceKAN, Wavelet-KAN and Legendre KAN

PyTorch Layers for FourierKAN, LaplaceKAN, Wavelet-KAN, and LegendreKAN

## Overview
This repository provides custom PyTorch layers designed as replacements for traditional linear layers followed by non-linear activations in neural network architectures. These layers are inspired by Kolmogorov-Arnold Networks (KAN) and utilize 1D Fourier, Laplace, Wavelet, or Legendre polynomial coefficients. By leveraging these mathematical transformations, these layers aim to provide efficient and numerically stable alternatives to spline-based methods commonly used in neural networks.

## Features
- **FourierKAN Layer**: Utilizes Fourier transformations to capture periodic features in data, offering a global approach to function approximation. This is beneficial for handling functions with periodicity and provides dense function representations.
  
- **LaplaceKAN Layer**: Applies Laplace transformations suited for data with exponential characteristics, facilitating a better handling of growth and decay dynamics in the dataset.

- **Wavelet-KAN Layer**: Employs wavelet transformations to efficiently capture both frequency and location information from data, making it suitable for tasks where data has non-periodic fluctuations.

- **LegendreKAN Layer**: Uses Legendre polynomials for creating polynomial-based transformations of inputs. This approach is advantageous for its orthogonality and ability to handle wider ranges without the need for manual re-scaling of the input space.

## Advantages
- **Density and Optimization**: Fourier and other global transformation methods are generally denser and potentially easier to optimize than local methods like splines due to their global support over the input space.
  
- **Numerical Stability**: The periodic nature of the Fourier transforms ensures that the functions are more numerically bounded, avoiding issues related to going out of the defined input grid. This makes the layers particularly robust to variations in input data.

- **Flexibility and Performance**: After achieving convergence with these transformations, it's possible to switch to spline approximations for faster evaluation while maintaining similar output quality. This offers flexibility in balancing between performance and computational efficiency.

- **Ease of Integration**: Designed to be drop-in replacements for linear layers, these layers can be seamlessly integrated into existing PyTorch models, enhancing their capability to capture complex patterns in data without extensive modifications to the architecture.

## Usage
These layers are intended to replace linear layers followed by non-linear activations in your existing PyTorch models. They can be integrated simply by substituting the traditional layers in your model's architecture with the corresponding KAN layer based on your specific needs (Fourier, Laplace, Wavelet, or Legendre). The choice of layer should align with the nature of the data and the specific requirements of the task.

## Example
```python
import torch
from KAN_layers import NaiveFourierKANLayer, NaiveLaplaceKANLayer, NaiveWaveletKANLayer, RecurrentLegendreLayer

# Define your model architecture
class Model(nn.Module):
    def __init__(self, params_list):
        super(Model, self).__init__()
        self.layer1 = RecurrentLegendreLayer(3, params_list[0], params_list[1])
        self.layer2 = RecurrentLegendreLayer(3, params_list[1], params_list[2])

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# Initialize model and pass data
model = Model([128,10,1])
input_data = torch.randn(1, 128)
output = model(input_data)

