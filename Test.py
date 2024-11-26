import pickle
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 提取预处理测试数据
ExtractPath = 'WorkPath.bin'
with open(ExtractPath, 'rb') as file:
    WorkPath = pickle.load(file)
    file.close()

ExtractPath = WorkPath + "/output/Test_Xs.bin"
with open(ExtractPath, 'rb') as file:
    InputArrays = pickle.load(file)
    file.close()

ExtractPath = WorkPath + "/output/Test_Ys.bin"
with open(ExtractPath, 'rb') as file:
    OutputArrays = pickle.load(file)
    file.close()

ExtractPath = WorkPath + "/Net/Nets.bin"
with open(ExtractPath, 'rb') as file:
    Nets = pickle.load(file)
    file.close()

ExtractPath = WorkPath + "/Net/Nor_Factors.bin"
with open(ExtractPath, 'rb') as file:
    Nor_Factors = pickle.load(file)
    file.close()

ExtractPath = WorkPath + "/Net/InputNum.bin"
with open(ExtractPath, 'rb') as file:
    InputNum = pickle.load(file)
    file.close()

ExtractPath = WorkPath + "/Net/TrainNum.bin"
with open(ExtractPath, 'rb') as file:
    TrainNum = pickle.load(file)
    file.close()

ExtractPath = WorkPath + "/Net/LOSS_Record.bin"
with open(ExtractPath, 'rb') as file:
    LOSS_Record = pickle.load(file)
    file.close()
LOSS_Record = list(map(list, zip(*LOSS_Record)))

# 测试集 处理
def apply_normalization(data, _min, _range):
    return (data - _min) / _range


def apply_de_normalization(data, _min, _range):
    return data * _range + _min


def mape(true, pre):
    e = np.abs((true - pre) / true)
    re = e.mean() * 100
    return re


# 归一
for index in range(0, len(InputArrays)):
    InputArray = InputArrays[index][:, 0:InputNum]
    InputArray_Nor = apply_normalization(InputArray, Nor_Factors["x_min"][index], Nor_Factors["x_range"][index])
    OutputArray_Nor = apply_normalization(OutputArrays[index], Nor_Factors["y_min"][index], Nor_Factors["y_range"][index])

    # Test 使用CUDA
    Net_cuda = Nets[index].to(device)
    InputArray_Nor = torch.tensor(InputArray_Nor, dtype=torch.float32)
    InputArray_Nor_cuda = InputArray_Nor.to(device)

    Yp_Nor_cuda = Net_cuda(InputArray_Nor_cuda)
    Yp_Nor = Yp_Nor_cuda.cpu().detach().numpy()
    Yp = apply_de_normalization(Yp_Nor, Nor_Factors["y_min"][index], Nor_Factors["y_range"][index])

    loss_Nor = mape(Yp_Nor, OutputArray_Nor)
    loss = mape(Yp, OutputArrays[index])

    if index + 1 == len(InputArrays):
        print("NetPooled", "\tNormalized\tMAPE\t", loss_Nor, "%")
        print("NetPooled", "\tOriginal\tMAPE\t", loss, "%", "\n")
    else:
        print("Net" + str(index + 1), "\tNormalized\tMAPE\t", loss_Nor, "%")
        print("Net" + str(index + 1), "\tOriginal\tMAPE\t", loss, "%", "\n")

    # 绘图
    a = np.hstack((OutputArrays[index], Yp, OutputArrays[index] - Yp)).tolist()
    a.sort(key=(lambda x: x[0]))
    a = list(map(list, zip(*a)))

    plt.figure()
    plt.plot(np.arange(len(OutputArrays[index])), a[1])
    plt.plot(np.arange(len(OutputArrays[index])), a[0])
    plt.plot(np.arange(len(OutputArrays[index])), a[2])

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2fdB'))
    plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    plt.xlabel('测试样本序号')
    plt.ylabel('损耗')

    plt.figure()
    plt.plot(np.arange(TrainNum), np.log10(LOSS_Record[index]))

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('10^%.2f'))
    plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    plt.xlabel('轮次')
    plt.ylabel('误差')


for index in range(0, len(InputArrays) - 1):
    InputArray = InputArrays[index][:, 0:InputNum]
    InputArray_Nor = apply_normalization(InputArray, Nor_Factors["x_min"][len(InputArrays) - 1], Nor_Factors["x_range"][len(InputArrays) - 1])
    OutputArray_Nor = apply_normalization(OutputArrays[index], Nor_Factors["y_min"][len(InputArrays) - 1], Nor_Factors["y_range"][len(InputArrays) - 1])

    # Test 使用CUDA
    Net_cuda = Nets[len(InputArrays) - 1].to(device)
    InputArray_Nor = torch.tensor(InputArray_Nor, dtype=torch.float32)
    InputArray_Nor_cuda = InputArray_Nor.to(device)

    Yp_Nor_cuda = Net_cuda(InputArray_Nor_cuda)
    Yp_Nor = Yp_Nor_cuda.cpu().detach().numpy()
    Yp = apply_de_normalization(Yp_Nor, Nor_Factors["y_min"][len(InputArrays) - 1], Nor_Factors["y_range"][len(InputArrays) - 1])

    loss_Nor = mape(Yp_Nor, OutputArray_Nor)
    loss = mape(Yp, OutputArrays[index])
    print("NetPooled Data" + str(index + 1), "Normalized\tMAPE\t", loss_Nor, "%")
    print("NetPooled Data" + str(index + 1), "Original\tMAPE\t", loss, "%", "\n")

    # 绘图
    a = np.hstack((OutputArrays[index], Yp, OutputArrays[index] - Yp)).tolist()
    a.sort(key=(lambda x: x[0]))
    a = list(map(list, zip(*a)))

    plt.figure()
    plt.plot(np.arange(len(OutputArrays[index])), a[1])
    plt.plot(np.arange(len(OutputArrays[index])), a[0])
    plt.plot(np.arange(len(OutputArrays[index])), a[2])

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2fdB'))
    plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    plt.xlabel('测试样本序号')
    plt.ylabel('预测偏差')



plt.show(block=True)
