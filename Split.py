from sklearn.model_selection import train_test_split
import pickle
import numpy as np


# 定义
TestProportion = 0.05
RandomSeed = 28


# 读取
ExtractPath = 'WorkPath.bin'
with open(ExtractPath, 'rb') as file:
    WorkPath = pickle.load(file)
    file.close()

ExtractPath = WorkPath + "/output/Sorted_UsedInputTables.bin"
with open(ExtractPath, 'rb') as file:
    Sorted_UsedInputTables = pickle.load(file)
    file.close()

ExtractPath = WorkPath + "/output/Sorted_UsedInputArrays.bin"
with open(ExtractPath, 'rb') as file:
    Sorted_UsedInputArrays = pickle.load(file)
    file.close()

ExtractPath = WorkPath + "/output/UsedOutputArrays.bin"
with open(ExtractPath, 'rb') as file:
    UsedOutputArrays = pickle.load(file)
    file.close()


# 随机划分
Train_Xs = []
Train_Ys = []
Test_Xs = []
Test_Ys = []
for index in range(0, len(Sorted_UsedInputArrays)):
    DataSet = np.hstack((Sorted_UsedInputArrays[index], UsedOutputArrays[index]))
    train_set, test_set = train_test_split(DataSet, test_size=TestProportion, random_state=RandomSeed)
    train_x, train_y = np.hsplit(train_set, [-1])
    test_x, test_y = np.hsplit(test_set, [-1])

    Train_Xs.append(train_x)
    Train_Ys.append(train_y)
    Test_Xs.append(test_x)
    Test_Ys.append(test_y)


# 保存
SavePath = WorkPath + "/output/Train_Xs.bin"
with open(SavePath, 'wb') as file:
    pickle.dump(Train_Xs, file)
    file.close()

SavePath = WorkPath + "/output/Train_Ys.bin"
with open(SavePath, 'wb') as file:
    pickle.dump(Train_Ys, file)
    file.close()

SavePath = WorkPath + "/output/Test_Xs.bin"
with open(SavePath, 'wb') as file:
    pickle.dump(Test_Xs, file)
    file.close()

SavePath = WorkPath + "/output/Test_Ys.bin"
with open(SavePath, 'wb') as file:
    pickle.dump(Test_Ys, file)
    file.close()
