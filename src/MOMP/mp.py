import numpy as np

def MP(Y, proj, stop):
    bool_flat = len(Y.shape) == 1   # Check if Y is a vector
    if bool_flat:
        Y = Y[..., np.newaxis]
    # Initialization
    I = []
    AX_I = np.array([])
    Y_res = Y
    # Loop
    while True:
        ii, AX_ii = proj(Y_res)
        I.append(ii)
        AX_I = np.contatenate([AX_I, AX_ii[:, np.newaxis]], axis=1)
        alpha_ii = np.dot(AX_ii.conj(), Y_res)
        alpha = np.concatenate([alpha, alpha_ii[np.newaxis]], axis=0)
        Y_res = Y_res - AX_ii*alpha_ii
        if stop(Y, Y_res, AX_I, alpha):
            break
    # Undo flatness
    if bool_flat:
        alpha = alpha[:, 0]
    return alpha, np.array(I)

class OMP:
    def __init__(self, proj, stop):
        self.proj = proj
        self.stop = stop
    def __call__(self, Y):
        bool_flat = len(Y.shape) == 1   # Check if Y is a vector
        if bool_flat:
            Y = Y[..., np.newaxis]
        # Initialization
        I = []
        AX_I = np.zeros((Y.shape[0], 0))
        alpha = np.zeros((0, Y.shape[1]))
        Y_res = Y.copy()
        # Loop
        while not self.stop(Y, Y_res, AX_I, alpha):
            ii, AX_ii = self.proj(Y_res)
            I.append(ii)
            AX_I = np.concatenate([AX_I, AX_ii[:, np.newaxis]], axis=1)
            alpha = np.linalg.lstsq(AX_I, Y, rcond=None)[0]
            Y_res = Y - np.dot(AX_I, alpha)
        # Undo flatness
        if bool_flat:
            alpha = alpha[:, 0]
        return np.array(I), alpha
