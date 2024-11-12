import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import pickle
from sklearn.model_selection import train_test_split

# defines
# RawDataFile = 'PL_Data.mat'
# RawDataArrayName = 'data_pl_241105'  # mat文件中的数组名
RawDataFile = 'PL_Data2.mat'
RawDataArrayName = 'data_pl_241108'  # mat文件中的数组名

RawData_UsedInput_Table = (  # 提取有用输入列,索引&注释 按原始数据顺序 下标从0开始
    [0, "纬度"],
    [1, "经度"],
    [2, "频率"],
    [3, "仰角"],
    [4, "星地距离"],
    [5, "地面站高度"],
    [6, "地表温度"],
    [7, "地表气压"],
    [8, "地表水蒸气密度"],
    [9, "综合水蒸气含量"],
    [10, "液态云含水量"],
    [11, "降雨量"],
    [12, "降雨高度"],
    [13, "湿表面反射率"],
)

RawData_UsedOutput_Table = (  # 提取有用输出列,索引&注释 按原始数据顺序 下标从0开始
    # [18, "总衰减"],
    [19, "总衰减"],
)

RawData_UsedInput_Table = list(map(list, zip(*RawData_UsedInput_Table)))  # 转置
RawData_UsedOutput_Table = list(map(list, zip(*RawData_UsedOutput_Table)))

# 提取
RawData = scio.loadmat(RawDataFile)  # dict 字典
RawArray = RawData[RawDataArrayName]

UsedInputArray = RawArray[:, RawData_UsedInput_Table[0]]
UsedOutputArray = RawArray[:, RawData_UsedOutput_Table[0]]
# print(UsedInputArray)
# print(UsedOutputArray)

# PCA
ModelPCA = PCA(n_components=len(RawData_UsedInput_Table[0]))
ModelPCA.fit(UsedInputArray)

RawData_UsedInput_Table.append(ModelPCA.mean_.tolist())
# print("PCA维数", ModelPCA.n_components_)
# print("原始参量权重", RawData_UsedInput_Table[2])

# 排序索引表和希望使用的输入数据
RawData_UsedInput_Table = list(map(list, zip(*RawData_UsedInput_Table)))  # 转置
Sorted_UsedInput_Table = sorted(RawData_UsedInput_Table, key=(lambda x: x[2]), reverse=True)  # 排序
Sorted_UsedInput_Table = list(map(list, zip(*Sorted_UsedInput_Table)))  # 转置
Sorted_UsedInputArray = RawArray[:, Sorted_UsedInput_Table[0]]

Sorted_UsedInput_Table.append(ModelPCA.explained_variance_ratio_.tolist())  # 方差和原始权重排序不相关 仅参考

Sorted_UsedInput_Table = list(map(list, zip(*Sorted_UsedInput_Table)))  # 转置
print("PCA排序的索引表", Sorted_UsedInput_Table)
print("PCA排序的输入数据", Sorted_UsedInputArray)

# 随机划分
DataSet = np.hstack((Sorted_UsedInputArray, UsedOutputArray))
train_set, test_set = train_test_split(DataSet, test_size=0.05, random_state=78)

# 保存排序的索引表 输入数据 标签数据
train_x, train_y = np.hsplit(train_set, [-1])
test_x, test_y = np.hsplit(test_set, [-1])

with open('DataSet_IndexTable.bin', 'wb') as file:
    pickle.dump(Sorted_UsedInput_Table, file)
    file.close()

with open('Train_X.bin', 'wb') as file:
    pickle.dump(train_x, file)
    file.close()

with open('Train_Y.bin', 'wb') as file:
    pickle.dump(train_y, file)
    file.close()

with open('Test_X.bin', 'wb') as file:
    pickle.dump(test_x, file)
    file.close()

with open('Test_Y.bin', 'wb') as file:
    pickle.dump(test_y, file)
    file.close()
