import numpy as np

def pinv_diag(S):
    nonzero = S != 0
    pinv = S.copy()
    pinv[nonzero] = 1 / pinv[nonzero]
    return pinv

def nystroem_flip(A, B):
    # TODO: implement with eigh?
    U, S, V = np.linalg.svd(A)
    del V
    sqrt_S = np.sqrt(S)

    return np.vstack((
        U * sqrt_S,
        B.T.dot(U * pinv_diag(sqrt_S))
    ))

def nystroem_clip(A, B):
    lam, U = np.linalg.eigh(A)
    sqrt_lam = np.sqrt(np.maximum(0, lam))

    return np.vstack((
        U * sqrt_lam,
        B.T.dot(U * pinv_diag(sqrt_lam))
    ))
