import pickle
import numpy as np
import torch
from torch import nn
import torch.utils.data as TorchData
import matplotlib.pyplot as plt
import copy

# 设定参数
InputNum = 12
TrainNum = 5
BatchSize = 100
LR_Initial = 1

# 运行平台
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


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
LOSSs = [copy.deepcopy(np.nan) for i in range(0, len(InputArrays))]
LOSS_Record = [copy.deepcopy(LOSSs) for i in range(0, TrainNum)]  # 记录损失
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

    # 训练 使用CUDA
    Net_cuda = Net.to(device)
    Loss_cuda = Loss.to(device)
    InputArray_cuda = InputArrays[index].to(device)
    OutputArray_cuda = OutputArrays[index].to(device)

    optim = torch.optim.SGD(params=Net_cuda.parameters(), lr=LR_Initial)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim,
                                                           mode='min',
                                                           factor=0.9995,
                                                           threshold=1e-7,
                                                           patience=0,
                                                           cooldown=0,
                                                           min_lr=0.1)

    for epoch in range(TrainNum):
        count = 0
        loss_average = 0
        BatchStartIndex = 0
        epoch_end = False
        while True:
            if BatchStartIndex + BatchSize < len(InputArray_cuda):
                batch_x = InputArray_cuda[BatchStartIndex:(BatchStartIndex + BatchSize)]
                batch_y = OutputArray_cuda[BatchStartIndex:(BatchStartIndex + BatchSize)]
            else:
                batch_x = InputArray_cuda[BatchStartIndex:len(InputArray_cuda)]
                batch_y = OutputArray_cuda[BatchStartIndex:len(InputArray_cuda)]
                epoch_end = True

            yp = Net_cuda(batch_x)  # 前向传递的预测值
            loss = Loss_cuda(yp, batch_y)  # 预测值与实际值的差别
            optim.zero_grad()
            loss.backward()  # 反向传递
            optim.step()  # 更新参数

            loss_average += loss.to('cpu').data.item()
            count += 1

            if epoch_end:
                break
            BatchStartIndex += BatchSize


        loss_average /= count
        scheduler.step(loss_average)
        LOSSs[index] = loss_average
        LOSS_Record[epoch][index] = LOSSs[index]
        print(epoch, LOSS_Record[epoch], optim.param_groups[0]['lr'])

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

SavePath = WorkPath + "/Net/TrainNum.bin"
with open(SavePath, 'wb') as file:
    pickle.dump(TrainNum, file)
    file.close()

SavePath = WorkPath + "/Net/LOSS_Record.bin"
with open(SavePath, 'wb') as file:
    pickle.dump(LOSS_Record, file)
    file.close()
