import numpy as np
import os

base = "data/fixed"
for g in os.listdir(base):
    f = os.listdir(os.path.join(base, g))[0]
    x = np.load(os.path.join(base, g, f))
    print(g, x.shape)
