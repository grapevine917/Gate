import numpy as np
import os

path = 'data/sift/sift_base.fvecs'
if not os.path.exists(path):
    path = 'sift/sift_base.fvecs'

data = np.fromfile(path, dtype=np.int32)
dim = data[0]
print('dim =', dim)
print('num vectors =', data.size // (dim + 1))
