import numpy as np

class General:
    def __init__(self, maxIter=None):
        self.maxIter = maxIter
    def __call__(self, Y, Y_res, AX_I, *args, **kwargs):
        if self.maxIter is not None and AX_I.shape[-1] >= self.maxIter:
            return True
        return False
