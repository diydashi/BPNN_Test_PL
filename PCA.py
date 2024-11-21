from sklearn.decomposition import PCA
import pickle
import copy


# 读取
ExtractPath = 'WorkPath.bin'
with open(ExtractPath, 'rb') as file:
    WorkPath = pickle.load(file)
    file.close()

ExtractPath = WorkPath + "/output/RawData_UsedInput_Table.bin"
with open(ExtractPath, 'rb') as file:
    RawData_UsedInput_Table = pickle.load(file)
    file.close()

ExtractPath = WorkPath + "/output/RawData_UsedOutput_Table.bin"
with open(ExtractPath, 'rb') as file:
    RawData_UsedOutput_Table = pickle.load(file)
    file.close()

ExtractPath = WorkPath + "/output/RawFileNames.bin"
with open(ExtractPath, 'rb') as file:
    RawFiles = pickle.load(file)
    file.close()

ExtractPath = WorkPath + "/output/UsedInputArrays.bin"
with open(ExtractPath, 'rb') as file:
    UsedInputArrays = pickle.load(file)
    file.close()

ExtractPath = WorkPath + "/output/UsedOutputArrays.bin"
with open(ExtractPath, 'rb') as file:
    UsedOutputArrays = pickle.load(file)
    file.close()

RawData_UsedInput_Table = list(map(list, zip(*RawData_UsedInput_Table)))  # 转置
RawData_UsedOutput_Table = list(map(list, zip(*RawData_UsedOutput_Table)))


# PCA
DataSetCount = 0
Sorted_UsedInputTables = []
Sorted_UsedInputArrays = []
for UsedInputArray in UsedInputArrays:
    ModelPCA = PCA(n_components=len(RawData_UsedInput_Table[0]))
    ModelPCA.fit(UsedInputArray)

    Sorted_UsedInputTables.append(copy.deepcopy(RawData_UsedInput_Table))
    Sorted_UsedInputTables[DataSetCount].append(copy.deepcopy(ModelPCA.mean_.tolist()))
    # print("PCA维数", ModelPCA.n_components_)
    # print("原始参量权重", RawData_UsedInput_Table[2])

    # 排序索引表和希望使用的输入数据
    Sorted_UsedInputTables[DataSetCount] = list(map(list, zip(*Sorted_UsedInputTables[DataSetCount])))  # 转置
    Sorted_UsedInputTables[DataSetCount] = sorted(Sorted_UsedInputTables[DataSetCount], key=(lambda x: x[2]), reverse=True)  # 排序
    Sorted_UsedInputTables[DataSetCount] = list(map(list, zip(*Sorted_UsedInputTables[DataSetCount])))  # 转置
    Sorted_UsedInputArrays.append(copy.deepcopy(UsedInputArray[:, Sorted_UsedInputTables[DataSetCount][0]]))

    Sorted_UsedInputTables[DataSetCount].append(copy.deepcopy(ModelPCA.explained_variance_ratio_.tolist()))  # 方差和原始权重排序不相关 仅参考

    Sorted_UsedInputTables[DataSetCount] = list(map(list, zip(*Sorted_UsedInputTables[DataSetCount])))  # 转置
    print("PCA排序的索引表", Sorted_UsedInputTables[DataSetCount])
    print("PCA排序的输入数据", Sorted_UsedInputArrays[DataSetCount])

    DataSetCount += 1


# 保存排序的索引表 输入数据 标签数据

SavePath = WorkPath + "/output/Sorted_UsedInputTables.bin"
with open(SavePath, 'wb') as file:
    pickle.dump(Sorted_UsedInputTables, file)
    file.close()

SavePath = WorkPath + "/output/Sorted_UsedInputArrays.bin"
with open(SavePath, 'wb') as file:
    pickle.dump(Sorted_UsedInputArrays, file)
    file.close()






