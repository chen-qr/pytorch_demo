import torch
import numpy as np

data = [[1, 2],[3, 4]]

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(x_np)