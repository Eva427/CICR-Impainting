import torch
a = [[1,2],[3,4]]
a = torch.Tensor(a)
print(a.shape)
a = torch.unsqueeze(a, dim=0)
print(a.shape)