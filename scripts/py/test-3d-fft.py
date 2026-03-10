import numpy as np
from numpy.fft import fft, fftn, ifft, ifftn

"""
3D-FFT with pencil-pencil-pencil (3pen) decomposition

"""

# Initialize random data
n0, n1, n2 = 64, 32, 16
X = np.random.randn(n0, n1, n2) # + 1j * np.random.randn(n0, n1, n2)

# reference FFT
X_hat = fftn(X)

# partition points
sp0 = n0 // 2
sp1 = n1 // 2
sp2 = n2 // 2

# initial split data
FX1 = X[:, :sp1, :sp2]
FX2 = X[:, sp1:, :sp2]
FX3 = X[:, :sp1, sp2:]
FX4 = X[:, sp1:, sp2:]

# check if split was correct
XS12 = np.concatenate((FX1, FX2), axis=1)
XS34 = np.concatenate((FX3, FX4), axis=1)
XS1234 = np.concatenate((XS12, XS34), axis=2)
print(f"error in split: {np.linalg.norm(X - XS1234):e}")

# first 1D-FFT
FX1h1 = fft(FX1, axis=0)
FX2h1 = fft(FX2, axis=0)
FX3h1 = fft(FX3, axis=0)
FX4h1 = fft(FX4, axis=0)

# print(FX1h1.shape, FX2h1.shape, FX3h1.shape, FX4h1.shape)

# first concatenation + transpose
FX1h1_t = np.concatenate(
    (
        np.transpose(FX1h1[:sp0, :, :], axes=(1, 2, 0)), 
        np.transpose(FX2h1[:sp0, :, :], axes=(1, 2, 0))
    ),
    axis=0
)

FX3h1_t = np.concatenate(
    (
        np.transpose(FX1h1[sp0:, :, :], axes=(1, 2, 0)), 
        np.transpose(FX2h1[sp0:, :, :], axes=(1, 2, 0))
    ),
    axis=0
)

FX2h1_t = np.concatenate(
    (
        np.transpose(FX3h1[:sp0, :, :], axes=(1, 2, 0)), 
        np.transpose(FX4h1[:sp0, :, :], axes=(1, 2, 0))
    ),
    axis=0
)

FX4h1_t = np.concatenate(
    (
        np.transpose(FX3h1[sp0:, :, :], axes=(1, 2, 0)), 
        np.transpose(FX4h1[sp0:, :, :], axes=(1, 2, 0))
    ),
    axis=0
)

# print(FX1h1_t.shape, FX2h1_t.shape, FX3h1_t.shape, FX4h1_t.shape)

# second 1D-FFT
FX1h2 = fft(FX1h1_t, axis=0)
FX2h2 = fft(FX2h1_t, axis=0)
FX3h2 = fft(FX3h1_t, axis=0)
FX4h2 = fft(FX4h1_t, axis=0)

# print(FX1h2.shape, FX2h2.shape, FX3h2.shape, FX4h2.shape)

# second concatenation + transpose
FX1h2_t = np.concatenate(
    (
        np.transpose(FX1h2[:sp1, :, :], axes=(1, 2, 0)), 
        np.transpose(FX2h2[:sp1, :, :], axes=(1, 2, 0))
    ),
    axis=0
)

FX3h2_t = np.concatenate(
    (
        np.transpose(FX1h2[sp1:, :, :], axes=(1, 2, 0)), 
        np.transpose(FX2h2[sp1:, :, :], axes=(1, 2, 0))
    ),
    axis=0
)

FX2h2_t = np.concatenate(
    (
        np.transpose(FX3h2[:sp1, :, :], axes=(1, 2, 0)), 
        np.transpose(FX4h2[:sp1, :, :], axes=(1, 2, 0))
    ),
    axis=0
)

FX4h2_t = np.concatenate(
    (
        np.transpose(FX3h2[sp1:, :, :], axes=(1, 2, 0)), 
        np.transpose(FX4h2[sp1:, :, :], axes=(1, 2, 0))
    ),
    axis=0
)

# print(FX1h2_t.shape, FX2h2_t.shape, FX3h2_t.shape, FX4h2_t.shape)

# third 1D-FFT
FX1h3 = fft(FX1h2_t, axis=0)
FX2h3 = fft(FX2h2_t, axis=0)
FX3h3 = fft(FX3h2_t, axis=0)
FX4h3 = fft(FX4h2_t, axis=0)

# print(FX1h3.shape, FX2h3.shape, FX3h3.shape, FX4h3.shape)

# third concatenation + transpose
FX1h3_t = np.concatenate(
    (
        np.transpose(FX1h3[:sp2, :, :], axes=(1, 2, 0)), 
        np.transpose(FX2h3[:sp2, :, :], axes=(1, 2, 0))
    ),
    axis=0
)

FX3h3_t = np.concatenate(
    (
        np.transpose(FX1h3[sp2:, :, :], axes=(1, 2, 0)), 
        np.transpose(FX2h3[sp2:, :, :], axes=(1, 2, 0))
    ),
    axis=0
)

FX2h3_t = np.concatenate(
    (
        np.transpose(FX3h3[:sp2, :, :], axes=(1, 2, 0)), 
        np.transpose(FX4h3[:sp2, :, :], axes=(1, 2, 0))
    ),
    axis=0
)

FX4h3_t = np.concatenate(
    (
        np.transpose(FX3h3[sp2:, :, :], axes=(1, 2, 0)), 
        np.transpose(FX4h3[sp2:, :, :], axes=(1, 2, 0))
    ),
    axis=0
)

# print(FX1h3_t.shape, FX2h3_t.shape, FX3h3_t.shape, FX4h3_t.shape)

# final concatenation
FXS12 = np.concatenate((FX1h3_t, FX3h3_t), axis=2)
FXS34 = np.concatenate((FX2h3_t, FX4h3_t), axis=2)
FXS1234 = np.concatenate((FXS12, FXS34), axis=1)
print(f"error in fft: {np.linalg.norm(X_hat - FXS1234):e}")