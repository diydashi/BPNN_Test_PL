import pickle
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# 输入参数数量
InputNum = 12
TrainNum = 200000
LR_Initial = 4

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
def apply_normalization(data, _min, _range):
    return (data - _min) / _range


def get_normalization_factor(data):
    _min = data.min(axis=0)
    _range = data.max(axis=0) - _min
    return _min, _range


InputArray = InputArray[:, 0:InputNum]
x_normalization_min, x_normalization_range = get_normalization_factor(InputArray)
y_normalization_min, y_normalization_range = get_normalization_factor(OutputArray)
InputArray = apply_normalization(InputArray, x_normalization_min, x_normalization_range)  # 归一化
OutputArray = apply_normalization(OutputArray, y_normalization_min, y_normalization_range)  # 归一化

InputArray = torch.tensor(InputArray, dtype=torch.float32)
OutputArray = torch.tensor(OutputArray, dtype=torch.float32)

Net = nn.Sequential(nn.Linear(InputNum, 64),    nn.LeakyReLU(),
                    nn.Linear(64, 48),          nn.LeakyReLU(),
                    nn.Linear(48, 24),          nn.LeakyReLU(),
                    nn.Linear(24, 1),           nn.Softplus())

Loss = nn.MSELoss()
optim = torch.optim.SGD(params=Net.parameters(), lr=LR_Initial)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim,
                                                       mode='min',
                                                       factor=0.9995,
                                                       threshold=1e-7,
                                                       patience=0,
                                                       cooldown=1,
                                                       min_lr=0.1)

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
    scheduler.step(loss)

    print(i, loss, optim.param_groups[0]['lr'])

torch.save(Net_cuda, 'Net.pt')

with open('x_normalization_factor.bin', 'wb') as file:
    pickle.dump([x_normalization_min, x_normalization_range], file)
    file.close()

with open('y_normalization_factor.bin', 'wb') as file:
    pickle.dump([y_normalization_min, y_normalization_range], file)
    file.close()
