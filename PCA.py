import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt  # 加载matplotlib用于数据的可视化
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# defines
RawDataFile = 'PL_Data.mat'
RawDataArrayName = 'data_pl_241105'  # mat文件中的数组名

RawData_UsedInput_Table = (  # 提取有用输入列,索引&注释 按原始数据顺序 下标从0开始
    #[2, "频率"],
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

RawData_UsedOutput_Table = (  # 提取有用输出列,索引&注释 按适当顺序 下标从0开始
    [18, "总衰减"],
)





# 提取
RawData = scio.loadmat(RawDataFile)  # dict 字典
RawArray = RawData[RawDataArrayName]

UsedInputIndex = []
for i in RawData_UsedInput_Table[:]:
    UsedInputIndex.append(i[0])
UsedInputArray = RawArray[:, UsedInputIndex]

UsedOutputIndex = []
for i in RawData_UsedOutput_Table[:]:
    UsedOutputIndex.append(i[0])
UsedOutputArray = RawArray[:, UsedOutputIndex]
#print(UsedInputArray)
#print(UsedOutputArray)

# PCA
ModelPCA = PCA(n_components=0.99)
ModelPCA.fit(UsedInputArray)
PCA_Mean = ModelPCA.mean_.tolist()
#print("PCA维数", ModelPCA.n_components_)
#print("PCA方差", ModelPCA.explained_variance_ratio_)
#print("原始参量权重", PCA_Mean)

# 排序索引表和希望使用的输入数据
Sorted_UsedInput_Table = list(RawData_UsedInput_Table)
for i in range(len(Sorted_UsedInput_Table)):
    Sorted_UsedInput_Table[i].append(PCA_Mean[i])
Sorted_UsedInput_Table.sort(key=(lambda x: x[2]), reverse=True)

Sorted_UsedInputIndex = []
for i in Sorted_UsedInput_Table[:]:
    Sorted_UsedInputIndex.append(i[0])
Sorted_UsedInputArray = RawArray[:, Sorted_UsedInputIndex]

for i in Sorted_UsedInput_Table:
    del i[0]
print("PCA排序的索引表", Sorted_UsedInput_Table)
print("PCA排序的输入数据", Sorted_UsedInputArray)

# 保存排序的索引表 输入数据 标签数据
Sorted_UsedInput_Table = np.array(Sorted_UsedInput_Table)
np.save("PreprocessedInputData_IndexTable.npy", Sorted_UsedInput_Table)
np.save("PreprocessedInputData.npy", Sorted_UsedInputArray)
np.save("PreprocessedOutputData.npy", UsedOutputArray)
