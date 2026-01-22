import numpy as np

from numpy.fft import fft2, fftn

# Initialize random data
n0, n1, n2, n3 = 16, 16, 16, 16
X = np.random.randn(n0, n1, n2, n3) # + 1j * np.random.randn(n0, n1, n2, n3)

# reference FFT
X_hat = fftn(X)

# partition points
sp2 = n2 // 2
sp3 = n3 // 2

# initial split data
X1 = X[:, :, :sp2, :sp3]
X2 = X[:, :, sp2:, :sp3]
X3 = X[:, :, :sp2, sp3:]
X4 = X[:, :, sp2:, sp3:]

# check if split was correct
XS12 = np.concatenate((X1, X2), axis=2)
XS34 = np.concatenate((X3, X4), axis=2)
XS1234 = np.concatenate((XS12, XS34), axis=3)
print(f"4D error in split: {np.linalg.norm(X - XS1234):e}")

# first 2D-FFT
X1h1 = fft2(X1, axes=(0, 1))
X2h1 = fft2(X2, axes=(0, 1))
X3h1 = fft2(X3, axes=(0, 1))
X4h1 = fft2(X4, axes=(0, 1))

# first concatenation + transpose
X1h1_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(X1h1[:sp2, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(X2h1[:sp2, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(X3h1[:sp2, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(X4h1[:sp2, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

X2h1_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(X1h1[sp2:, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(X2h1[sp2:, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(X3h1[sp2:, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(X4h1[sp2:, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

X3h1_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(X1h1[:sp2, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(X2h1[:sp2, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(X3h1[:sp2, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(X4h1[:sp2, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

X4h1_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(X1h1[sp2:, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(X2h1[sp2:, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(X3h1[sp2:, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(X4h1[sp2:, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

# second 2D-FFT
X1h2 = fft2(X1h1_t, axes=(0, 1))
X2h2 = fft2(X2h1_t, axes=(0, 1))
X3h2 = fft2(X3h1_t, axes=(0, 1))
X4h2 = fft2(X4h1_t, axes=(0, 1))

# second concatenation + transpose (reverse)
X1h2_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(X1h2[:sp2, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(X2h2[:sp2, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(X3h2[:sp2, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(X4h2[:sp2, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

X2h2_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(X1h2[sp2:, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(X2h2[sp2:, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(X3h2[sp2:, :sp3, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(X4h2[sp2:, :sp3, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

X3h2_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(X1h2[:sp2, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(X2h2[:sp2, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(X3h2[:sp2, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(X4h2[:sp2, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

X4h2_t = np.concatenate(
    (
        np.concatenate(
            (
                np.transpose(X1h2[sp2:, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(X2h2[sp2:, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        ), 
        np.concatenate(
            (
                np.transpose(X3h2[sp2:, sp3:, :, :], axes=(2, 3, 0, 1)), 
                np.transpose(X4h2[sp2:, sp3:, :, :], axes=(2, 3, 0, 1))
            ), 
            axis=0
        )
    ), 
    axis=1
)

# final concatenation
XS12 = np.concatenate((X1h2_t, X2h2_t), axis=2)
XS34 = np.concatenate((X3h2_t, X4h2_t), axis=2)
XS1234 = np.concatenate((XS12, XS34), axis=3)
print(f"4D error in split: {np.linalg.norm(X_hat - XS1234):e}")