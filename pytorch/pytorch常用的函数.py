import torch

# mydata = torch.tensor([1,2,3,4,5,6])
#
# print(mydata)
#
# print(mydata.shape)
#
# mydata1 = mydata.reshape(2,-1)
#
# print(mydata1.shape)
# print(mydata1.unsqueeze(-1).squeeze().shape)

# transpose 和 permute  --- 交换一次 和  一次交换多次

#view() 和 contiguous() ---- view 只能修改在 存在整块内存的张量

#cat() ---将两个张量的维度 拼接起来

data1 = torch.randint(0,10,[1,2,3])
data2 = torch.randint(0,10,[1,2,3])

print(torch.cat([data1, data2], dim=1).shape)