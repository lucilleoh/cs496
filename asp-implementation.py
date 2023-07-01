# This implementation uses PyTorch's complex-valued tensors and the torch.fft module for performing Fourier transforms
# Input field, propagation distance, wavelength, pixel size → propagated field (as a complex-valued tensor)

import torch
import torch.fft as fft

def angular_spectrum_propagator(input_field, z, wavelength, pixel_size):
    """
    Propagates an input field using the Angular Spectrum Propagator method in PyTorch.

    Args:
        input_field (torch.Tensor): 2D complex-valued tensor representing the input field.
        z (float): Propagation distance (in meters).
        wavelength (float): Wavelength of the input field (in meters).
        pixel_size (float): Size of each pixel in the input field (in meters).

    Returns:
        torch.Tensor: 2D complex-valued tensor representing the propagated field.

    """

    # Get input field shape
    height, width = input_field.shape

    # Construct frequency grid
    fx = torch.linspace(-1 / (2 * pixel_size), 1 / (2 * pixel_size), width, device=input_field.device)
    fy = torch.linspace(-1 / (2 * pixel_size), 1 / (2 * pixel_size), height, device=input_field.device)
    FX, FY = torch.meshgrid(fx, fy)
    kx = 2 * torch.pi * FX
    ky = 2 * torch.pi * FY

    # Compute the wave number
    k = 2 * torch.pi / wavelength

    # Compute the spectral transfer function
    transfer_function = torch.exp(1j * torch.sqrt(k**2 - kx**2 - ky**2) * z)

    # Apply the Fourier transform to the input field
    input_field = fft.fftshift(input_field)
    input_field = fft.fftn(input_field)

    # Apply the spectral transfer function
    propagated_field = input_field * transfer_function

    # Apply the inverse Fourier transform
    propagated_field = fft.ifftn(propagated_field)
    propagated_field = fft.ifftshift(propagated_field)

    return propagated_field