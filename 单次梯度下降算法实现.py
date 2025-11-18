import torch

#定义张量x
x = torch.tensor(10,requires_grad=True,dtype=torch.float32)
print(f'x={x}')

#定义函数
loss = 2 * x ** 2

#定义起始值 w0 = 20

w0 = 20

#定义步长 lr
lr = 0.01
w1 = w0 = lr * loss.sum.backward()

print('w1 =',w1)
#损失函数

#求梯度

#