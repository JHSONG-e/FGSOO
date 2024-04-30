import numpy as np
import torch
from Gtopx import gtopx



def cassini2_gtopx(x):
    
    res = torch.zeros(x.shape[0], 1, dtype = torch.float64)
    for i in range(x.shape[0]):
        [f, g] = gtopx(2, x[i].tolist(), 1, x.shape[1], 0)
        res[i] = f[0]
    
    return -res






