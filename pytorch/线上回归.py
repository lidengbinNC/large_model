#导入模块
import matplotlib
matplotlib.use('TkAgg')  # 避免 PyCharm 自带后端冲突

import torch
from torch.utils.data import DataLoader,TensorDataset
from torch.nn import Linear,MSELoss
from torch.optim import SGD

from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


#构建数据集
def create_dataset():
    x,y,coef = make_regression(n_samples=100,n_features=1,noise=10,coef=True,bias=1.5,random_state=0)
    x = torch.tensor(x,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)
    return x,y,coef

x,y,coef = create_dataset()

dataset = TensorDataset(x,y)
dataloader = DataLoader(dataset=dataset,batch_size=16,shuffle=True)
model = Linear(in_features=1,out_features=1)

#损失和优化器
loss = MSELoss()

optimizer = SGD(params=model.parameters(),lr=0.002)

epochs = 150
#损失的变化
loss_epoch = []
total_loss = 0.0
train_simple = 0.0

for _ in range(epochs):
    for train_x ,train_y in dataloader:
        y_pred = model(train_x.type(torch.float32))
        #计算损失值
        loss_value = loss(y_pred,train_y.reshape(-1,1).type(torch.float32))
        total_loss +=loss_value
        train_simple += len(train_y)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
    loss_epoch.append(total_loss/train_simple)
plt.show()

plt.scatter(x,y)
x1 = torch.linspace(x.min(),x.max(),1000)
y0 = torch.tensor([v*model.weight+model.bias for v in x1])
y1 = torch.tensor([v*coef+1.5 for v in x1])

plt.plot(x1,y0,label='预测值')
plt.plot(x1,y1,label= '真实值')
plt.grid()
plt.show()
