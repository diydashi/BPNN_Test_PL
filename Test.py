import pickle
import torch
from torch import nn
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

InputNum = 12

# 提取预处理测试数据
with open('DataSet_IndexTable.bin', 'rb') as file:
    InputIndex = pickle.load(file)
    file.close()

with open('Test_X.bin', 'rb') as file:
    InputArray = pickle.load(file)
    file.close()

with open('Test_Y.bin', 'rb') as file:
    OutputArray = pickle.load(file)
    file.close()

with open('x_normalization_factor.bin', 'rb') as file:
    x_nor = pickle.load(file)
    file.close()

with open('y_normalization_factor.bin', 'rb') as file:
    y_nor = pickle.load(file)
    file.close()


Net = torch.load('Net.pt')


# 测试集 处理
def apply_normalization(data, _min, _range):
    return (data - _min) / _range


def apply_de_normalization(data, _min, _range):
    return data * _range + _min


def mape(true, pre):
    e = np.abs((true - pre) / true)
    re = e.mean() * 100
    return re


InputArray = InputArray[:, 0:InputNum]
InputArray_Nor = apply_normalization(InputArray, x_nor[0], x_nor[1])
OutputArray_Nor = apply_normalization(OutputArray, y_nor[0], y_nor[1])  # 归一化

# Test 使用CUDA
Net_cuda = Net.to(device)
InputArray_Nor = torch.tensor(InputArray_Nor, dtype=torch.float32)
InputArray_Nor_cuda = InputArray_Nor.to(device)

Yp_Nor_cuda = Net(InputArray_Nor_cuda)
Yp_Nor = Yp_Nor_cuda.cpu().detach().numpy()
Yp = apply_de_normalization(Yp_Nor, y_nor[0], y_nor[1])

loss_Nor = mape(Yp_Nor, OutputArray_Nor)
loss = mape(Yp, OutputArray)
print("Normalized MAPE\t", loss_Nor, "%")
print("Original MAPE\t", loss, "%")



# 绘图
a = np.hstack((OutputArray, Yp)).tolist()
a.sort(key=(lambda x: x[0]))
a = list(map(list, zip(*a)))

b = (OutputArray - Yp).tolist()

plt.plot(np.arange(len(OutputArray)), a[1])
plt.plot(np.arange(len(OutputArray)), a[0])
plt.plot(np.arange(len(OutputArray)), b)
plt.show(block=True)
