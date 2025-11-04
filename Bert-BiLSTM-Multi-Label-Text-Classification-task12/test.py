import torch
t1 = torch.Tensor([[1,2,3,4],[2,2,3,4]])
index = torch.argmax(t1,axis=1)
print(index)

