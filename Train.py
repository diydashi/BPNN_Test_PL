import pickle
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 输入参数数量
InputNum = 6
TrainNum = 20000
LR = 2

# 提取预处理训练数据
with open('DataSet_IndexTable.bin', 'rb') as file:
    InputIndex = pickle.load(file)
    file.close()

with open('Train_X.bin', 'rb') as file:
    InputArray = pickle.load(file)
    file.close()

with open('Train_Y.bin', 'rb') as file:
    OutputArray = pickle.load(file)
    file.close()


# 处理数据 创建网络
def normalization(data):
    _range = data.max(axis=0) - data.min(axis=0)
    return (data - data.min(axis=0)) / _range


InputArray = InputArray[:, 0:InputNum]
InputArray = normalization(InputArray)  # 归一化
OutputArray = normalization(OutputArray)  # 归一化

InputArray = torch.tensor(InputArray, dtype=torch.float32)
OutputArray = torch.tensor(OutputArray, dtype=torch.float32)

Net = nn.Sequential(nn.Linear(InputNum, 50), nn.ReLU(),
                    nn.Linear(50, 16), nn.ReLU(),
                    nn.Linear(16, 1), nn.Softplus())

Loss = nn.MSELoss()
optim = torch.optim.SGD(params=Net.parameters(), lr=LR)  # 随机梯度下降

# 训练 使用CUDA
Net_cuda = Net.to(device)
Loss_cuda = Loss.to(device)
InputArray_cuda = InputArray.to(device)
OutputArray_cuda = OutputArray.to(device)

for i in range(TrainNum):
    yp = Net_cuda(InputArray_cuda)  # 前向传递的预测值
    loss = Loss_cuda(yp, OutputArray_cuda)  # 预测值与实际值的差别
    optim.zero_grad()
    loss.backward()  # 反向传递
    optim.step()  # 更新参数

    if i == TrainNum * 0.5:
        optim = torch.optim.SGD(params=Net.parameters(), lr=LR * 0.5)

    if i == TrainNum * 0.25:
        optim = torch.optim.SGD(params=Net.parameters(), lr=LR * 0.25)

    print(i, loss)

torch.save(Net_cuda, 'Net.pt')
