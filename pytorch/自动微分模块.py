import torch

x = torch.tensor(5)
y = torch.tensor(0.)
w = torch.tensor(1., requires_grad=True,dtype=torch.float32)
b = torch.tensor(3., requires_grad=True,dtype=torch.float32)
z = w * x + b

loss = torch.nn.MSELoss()
loss = loss(y,z)
loss.backward()

print(w.grad)
print(b.grad)