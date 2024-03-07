import torch
import numpy as np

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data) # 直接从数组生成Tensor

print(x_data)