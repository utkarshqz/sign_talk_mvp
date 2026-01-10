import numpy as np
import json

data = np.load("reference_db/thanks.npy")
print(json.dumps(data[0].tolist()))
