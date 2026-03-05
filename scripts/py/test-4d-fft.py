import numpy as np

from numpy.fft import fft2, fftn, ifft2, ifftn

# Initialize random data
n0, n1, n2, n3 = 32, 32, 32, 32
X = np.random.randn(n0, n1, n2, n3) # + 1j * np.random.randn(n0, n1, n2, n3)

# reference FFT
X_hat = fftn(X)

# partition points
sp2 = n2 // 2
sp3 = n3 // 2

# initial split data
FX1 = X[:, :, :sp2, :sp3]
FX2 = X[:, :, sp2:, :sp3]
FX3 = X[:, :, :sp2, sp3:]
FX4 = X[:, :, sp2:, sp3:]

# check if split was correct
XS12 = np.concatenate((FX1, FX2), axis=2)
XS34 = np.concatenate((FX3, FX4), axis=2)
XS1234 = np.concatenate((XS12, XS34), axis=3)
print(f"error in split: {np.linalg.norm(X - XS1234):e}")

# first 2D-FFT
FX1h1 = fft2(FX1, axes=(0, 1))
FX2h1 = fft2(FX2, axes=(0, 1))
FX3h1 = fft2(FX3, axes=(0, 1))
FX4h1 = fft2(FX4, axes=(0, 1))

# first concatenation + transpose
FX1h1_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(FX1h1[:sp2, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(FX2h1[:sp2, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(FX3h1[:sp2, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(FX4h1[:sp2, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

FX2h1_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(FX1h1[sp2:, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(FX2h1[sp2:, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(FX3h1[sp2:, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(FX4h1[sp2:, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

FX3h1_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(FX1h1[:sp2, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(FX2h1[:sp2, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(FX3h1[:sp2, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(FX4h1[:sp2, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

FX4h1_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(FX1h1[sp2:, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(FX2h1[sp2:, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(FX3h1[sp2:, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(FX4h1[sp2:, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

# second 2D-FFT
FX1h2 = fft2(FX1h1_t, axes=(0, 1))
FX2h2 = fft2(FX2h1_t, axes=(0, 1))
FX3h2 = fft2(FX3h1_t, axes=(0, 1))
FX4h2 = fft2(FX4h1_t, axes=(0, 1))

# second concatenation + transpose (reverse)
FX1h2_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(FX1h2[:sp2, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(FX2h2[:sp2, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(FX3h2[:sp2, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(FX4h2[:sp2, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

FX2h2_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(FX1h2[sp2:, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(FX2h2[sp2:, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(FX3h2[sp2:, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(FX4h2[sp2:, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

FX3h2_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(FX1h2[:sp2, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(FX2h2[:sp2, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(FX3h2[:sp2, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(FX4h2[:sp2, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

FX4h2_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(FX1h2[sp2:, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(FX2h2[sp2:, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(FX3h2[sp2:, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(FX4h2[sp2:, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

# final concatenation
FXS12 = np.concatenate((FX1h2_t, FX2h2_t), axis=2)
FXS34 = np.concatenate((FX3h2_t, FX4h2_t), axis=2)
FXS1234 = np.concatenate((FXS12, FXS34), axis=3)
print(f"error in fft: {np.linalg.norm(X_hat - FXS1234):e}")

# inverse 4D-FFT
X_ref = ifftn(X_hat)

# inverse 4D-FFT with slab-slab decomposition begins here

# initial split data
BX1 = FXS1234[:, :, :sp2, :sp3]
BX2 = FXS1234[:, :, sp2:, :sp3]
BX3 = FXS1234[:, :, :sp2, sp3:]
BX4 = FXS1234[:, :, sp2:, sp3:]

# check if split was correct
BXS12 = np.concatenate((BX1, BX2), axis=2)
BXS34 = np.concatenate((BX3, BX4), axis=2)
BXS1234 = np.concatenate((BXS12, BXS34), axis=3)
print(f"error in inverse split: {np.linalg.norm(FXS1234 - BXS1234):e}")

# first inverse 2D-FFT
BX1h1 = ifft2(BX1, axes=(0, 1))
BX2h1 = ifft2(BX2, axes=(0, 1))
BX3h1 = ifft2(BX3, axes=(0, 1))
BX4h1 = ifft2(BX4, axes=(0, 1))

# first concatenation + transpose
BX1h1_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(BX1h1[:sp2, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(BX2h1[:sp2, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(BX3h1[:sp2, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(BX4h1[:sp2, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

BX2h1_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(BX1h1[sp2:, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(BX2h1[sp2:, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(BX3h1[sp2:, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(BX4h1[sp2:, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

BX3h1_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(BX1h1[:sp2, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(BX2h1[:sp2, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(BX3h1[:sp2, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(BX4h1[:sp2, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

BX4h1_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(BX1h1[sp2:, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(BX2h1[sp2:, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(BX3h1[sp2:, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(BX4h1[sp2:, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

# second inverse 2D-FFT
BX1h2 = ifft2(BX1h1_t, axes=(0, 1))
BX2h2 = ifft2(BX2h1_t, axes=(0, 1))
BX3h2 = ifft2(BX3h1_t, axes=(0, 1))
BX4h2 = ifft2(BX4h1_t, axes=(0, 1))

# second concatenation + transpose (reverse)
BX1h2_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(BX1h2[:sp2, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(BX2h2[:sp2, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(BX3h2[:sp2, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(BX4h2[:sp2, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

BX2h2_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(BX1h2[sp2:, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(BX2h2[sp2:, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(BX3h2[sp2:, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(BX4h2[sp2:, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

BX3h2_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(BX1h2[:sp2, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(BX2h2[:sp2, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(BX3h2[:sp2, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(BX4h2[:sp2, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

BX4h2_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(BX1h2[sp2:, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(BX2h2[sp2:, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(BX3h2[sp2:, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(BX4h2[sp2:, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

# final concatenation
BXS12 = np.concatenate((BX1h2_t, BX2h2_t), axis=2)
BXS34 = np.concatenate((BX3h2_t, BX4h2_t), axis=2)
BXS1234 = np.concatenate((BXS12, BXS34), axis=3)
print(f"error in inverse 4D-fft: {np.linalg.norm(X_ref - BXS1234):e}")
