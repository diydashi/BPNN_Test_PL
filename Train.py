import pickle
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import copy

# 设定参数
InputNum = 12
TrainNum = 100000
LR_Initial = 4

# 运行平台
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


# 读取
ExtractPath = 'WorkPath.bin'
with open(ExtractPath, 'rb') as file:
    WorkPath = pickle.load(file)
    file.close()

ExtractPath = WorkPath + "/output/Train_Xs.bin"
with open(ExtractPath, 'rb') as file:
    InputArrays = pickle.load(file)
    file.close()

ExtractPath = WorkPath + "/output/Train_Ys.bin"
with open(ExtractPath, 'rb') as file:
    OutputArrays = pickle.load(file)
    file.close()


# 处理数据 创建网络
def apply_normalization(data, _min, _range):
    return (data - _min) / _range


def get_normalization_factor(data):
    _min = data.min(axis=0)
    _range = data.max(axis=0) - _min
    return _min, _range


Nets = []
Nor_Factors = {"x_min": [], "x_range": [], "y_min": [], "y_range": []}  # 归一化参数表
LOSSs = [np.nan for i in range(0, len(InputArrays))]
for index in range(0, len(InputArrays)):
    InputArrays[index] = InputArrays[index][:, 0:InputNum]
    x_normalization_min, x_normalization_range = get_normalization_factor(InputArrays[index])
    y_normalization_min, y_normalization_range = get_normalization_factor(OutputArrays[index])
    InputArrays[index] = apply_normalization(InputArrays[index], x_normalization_min, x_normalization_range)  # 归一化
    OutputArrays[index] = apply_normalization(OutputArrays[index], y_normalization_min, y_normalization_range)  # 归一化

    InputArrays[index] = torch.tensor(InputArrays[index], dtype=torch.float32)
    OutputArrays[index] = torch.tensor(OutputArrays[index], dtype=torch.float32)

    Net = nn.Sequential(nn.Linear(InputNum, 64), nn.LeakyReLU(),
                        nn.Linear(64, 48), nn.LeakyReLU(),
                        nn.Linear(48, 24), nn.LeakyReLU(),
                        nn.Linear(24, 1), nn.Softplus())

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
    InputArray_cuda = InputArrays[index].to(device)
    OutputArray_cuda = OutputArrays[index].to(device)

    for i in range(TrainNum):
        yp = Net_cuda(InputArray_cuda)  # 前向传递的预测值
        loss = Loss_cuda(yp, OutputArray_cuda)  # 预测值与实际值的差别
        optim.zero_grad()
        loss.backward()  # 反向传递
        optim.step()  # 更新参数
        scheduler.step(loss)

        LOSSs[index] = loss.data.to('cpu').item()
        print(i, LOSSs, optim.param_groups[0]['lr'])

    Net_cpu = Net_cuda.to('cpu')
    Nets.append(copy.deepcopy(Net_cpu))

    Nor_Factors["x_min"].append(copy.deepcopy(x_normalization_min))
    Nor_Factors["x_range"].append(copy.deepcopy(x_normalization_range))
    Nor_Factors["y_min"].append(copy.deepcopy(y_normalization_min))
    Nor_Factors["y_range"].append(copy.deepcopy(y_normalization_range))

# 保存
SavePath = WorkPath + "/Net/Nets.bin"
with open(SavePath, 'wb') as file:
    pickle.dump(Nets, file)
    file.close()

SavePath = WorkPath + "/Net/Nor_Factors.bin"
with open(SavePath, 'wb') as file:
    pickle.dump(Nor_Factors, file)
    file.close()

SavePath = WorkPath + "/Net/InputNum.bin"
with open(SavePath, 'wb') as file:
    pickle.dump(InputNum, file)
    file.close()
