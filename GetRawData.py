import os
import numpy as np
import scipy.io as scio
import pickle
import shutil

# 定义
WorkPath = "./WorkFile/241120"

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
    #[18, "总衰减"],
    [19, "总衰减"],
)

RawData_UsedInput_Table = list(map(list, zip(*RawData_UsedInput_Table)))  # 转置
RawData_UsedOutput_Table = list(map(list, zip(*RawData_UsedOutput_Table)))

# 提取
RawArrays = []
UsedInputArrays = []
UsedOutputArrays = []
RawFiles = os.listdir(WorkPath + "/input/")
for file in RawFiles:
    RawData = scio.loadmat(WorkPath + "/input/" + file)  # 提取 dict 字典
    for value in RawData.values():
        if type(value) == np.ndarray:
            RawArrays.append(value)
            break

for RawArray in RawArrays:
    UsedInputArrays.append(RawArray[:, RawData_UsedInput_Table[0]])
    UsedOutputArrays.append(RawArray[:, RawData_UsedOutput_Table[0]])

# 合并集
PooledUsedInputArray = 0
PooledUsedOutputArray = 0
for UsedInputArray in UsedInputArrays:
    if type(PooledUsedInputArray) != np.ndarray:
        PooledUsedInputArray = UsedInputArray
    else:
        PooledUsedInputArray = np.append(PooledUsedInputArray, UsedInputArray, axis=0)

for UsedOutputArray in UsedOutputArrays:
    if type(PooledUsedOutputArray) != np.ndarray:
        PooledUsedOutputArray = UsedOutputArray
    else:
        PooledUsedOutputArray = np.append(PooledUsedOutputArray, UsedOutputArray, axis=0)

UsedInputArrays.append(PooledUsedInputArray)
UsedOutputArrays.append(PooledUsedOutputArray)

RawData_UsedInput_Table = list(map(list, zip(*RawData_UsedInput_Table)))  # 转置
RawData_UsedOutput_Table = list(map(list, zip(*RawData_UsedOutput_Table)))


# 清空
SavePath = WorkPath + "/output/"
shutil.rmtree(SavePath)
os.mkdir(SavePath)
# git 不同步输出文件
GitIgnorePath = WorkPath + "/output/.gitignore"
with open(GitIgnorePath, 'w') as file:
    file.write('*')


# 保存
SavePath = 'WorkPath.bin'
with open(SavePath, 'wb') as file:
    pickle.dump(WorkPath, file)
    file.close()

SavePath = WorkPath + "/output/RawData_UsedInput_Table.bin"
with open(SavePath, 'wb') as file:
    pickle.dump(RawData_UsedInput_Table, file)
    file.close()

SavePath = WorkPath + "/output/RawData_UsedOutput_Table.bin"
with open(SavePath, 'wb') as file:
    pickle.dump(RawData_UsedOutput_Table, file)
    file.close()

SavePath = WorkPath + "/output/RawFileNames.bin"
with open(SavePath, 'wb') as file:
    pickle.dump(RawFiles, file)
    file.close()

SavePath = WorkPath + "/output/UsedInputArrays.bin"
with open(SavePath, 'wb') as file:
    pickle.dump(UsedInputArrays, file)
    file.close()

SavePath = WorkPath + "/output/UsedOutputArrays.bin"
with open(SavePath, 'wb') as file:
    pickle.dump(UsedOutputArrays, file)
    file.close()

print("files", len(RawFiles))
