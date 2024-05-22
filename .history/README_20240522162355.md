# FourierKAN, LaplaceKAN and Wavelet-KAN

Pytorch Layer for FourierKAN, LaplaceKAN and Wavelet-KAN

It is a layer intended to be a substitution for Linear + non-linear activation

This is inspired by Kolmogorov-Arnold Networks but using 1d fourier/laplace/wavelet coefficients instead of splines coefficients
It should be easier to optimize as fourier are more dense than spline (global vs local)
Once convergence is reached you can replace the 1d function with spline approximation for faster evaluation giving almost the same result
The other advantage of using fourier over spline is that the function are periodic, and therefore more numerically bounded
Avoiding the issues of going out of grid
