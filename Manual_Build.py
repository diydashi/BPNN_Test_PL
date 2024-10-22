# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

# 库的导入
import numpy as np
import pandas as pd

maxepochs = 20000  # 训练次数
learnrate = 0.01  # 学习率
errorfinal = 0.65 * 10 ** (-3)  # 停止训练误差阈值
samnum = 72  # 输入数据数量
indim = 4  # 输入层节点数
outdim = 1  # 输出层节点数
hiddenunitnum = 8  # 隐含层节点数

# 激活函数tanh
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


# 反激活函数
def de_tanh(x):
    return (1 - x ** 2)


# 输入数据的导入
df = pd.read_csv("train.csv")
df.columns = ["Co", "Cr", "Mg", "Pb", "Ti"]
Co = df["Co"]
Co = np.array(Co)
Cr = df["Cr"]
Cr = np.array(Cr)
Mg = df["Mg"]
Mg = np.array(Mg)
Pb = df["Pb"]
Pb = np.array(Pb)
Ti = df["Ti"]
Ti = np.array(Ti)
Sample_IN = np.matrix([Co, Cr, Mg, Pb])
#print(Sample_IN)
# 数据归一化，将输入数据压缩至1到-1之间，便于计算，后续通过反归一化恢复原始值
sampleinminmax = np.array([Sample_IN.min(axis=1).T.tolist()[0], Sample_IN.max(axis=1).T.tolist()[0]]).transpose()
print(sampleinminmax)
sampleinnorm = (2 * (np.array(Sample_IN.T) - sampleinminmax.transpose()[0]) / (sampleinminmax.transpose()[1] - sampleinminmax.transpose()[0]) - 1).transpose()
print(sampleinnorm)

sampleout = np.matrix([Ti])
sampleoutminmax = np.array([sampleout.min(axis=1).T.tolist()[0], sampleout.max(axis=1).T.tolist()[0]]).transpose()
sampleoutnorm = (2 * (np.array(sampleout.T) - sampleoutminmax.transpose()[0]) / (sampleoutminmax.transpose()[1] - sampleoutminmax.transpose()[0]) - 1).transpose()

noise = 0.03 * np.random.rand(sampleoutnorm.shape[0], sampleoutnorm.shape[1])   #注入噪声
sampleoutnorm += noise



# 随机生成隐含层与输出层的权值w和阈值b
scale = np.sqrt(3 / ((indim + outdim) * 0.5))  # 最大值最小值范围为-1.44~1.44
w1 = np.random.uniform(low=-scale, high=scale, size=[hiddenunitnum, indim])
b1 = np.random.uniform(low=-scale, high=scale, size=[hiddenunitnum, 1])
w2 = np.random.uniform(low=-scale, high=scale, size=[outdim, hiddenunitnum])
b2 = np.random.uniform(low=-scale, high=scale, size=[outdim, 1])

# errhistory存储误差
errhistory = np.matrix(np.zeros((1, maxepochs)))

# 开始训练
for i in range(maxepochs):
    print("The iteration is : ", i)
    # 前向传播，计算隐含层、输出层输出
    hiddenout = tanh((np.dot(w1, sampleinnorm).transpose() + b1.transpose())).transpose()
    networkout = tanh((np.dot(w2, hiddenout).transpose() + b2.transpose())).transpose() # X
    # 计算误差值
    err = sampleoutnorm - networkout #dLoss / dY = Y - Ygt
    loss = np.sum(err ** 2) / 2
    print("the loss is :", loss)
    errhistory[:, i] = loss
    # 判断是否停止训练
    if loss < errorfinal:
        break
    # 反向传播，利用结果误差进行误差项的计算
    delta2 = err * de_tanh(networkout) # dLoss / dW = dLoss / dY  *  dY / dW = (Y - Ygt) * X
    delta1 = np.dot(w2.transpose(), delta2) * de_tanh(hiddenout) #中间层 dLoss / dW
    # 计算输出层的误差项
    dw2 = np.dot(delta2, hiddenout.transpose())
    dw2 = dw2 / samnum
    db2 = np.dot(delta2, np.ones((samnum, 1)))
    db2 = db2 / samnum
    # 计算隐含层的误差项
    dw1 = np.dot(delta1, sampleinnorm.transpose())
    dw1 = dw1 / samnum
    db1 = np.dot(delta1, np.ones((samnum, 1)))
    db1 = db1 / samnum

    # 对权值、阈值进行更新
    w2 += learnrate * dw2
    b2 += learnrate * db2
    w1 += learnrate * dw1
    b1 += learnrate * db1
print('更新的权重w1:', w1)
print('更新的偏置b1:', b1)
print('更新的权重w2:', w2)
print('更新的偏置b2:', b2)
print("The loss after iteration is ：", loss)

# 保持训练结束后的权值、阈值，用于测试
np.save("w1.npy", w1)
np.save("b1.npy", b1)
np.save("w2.npy", w2)
np.save("b2.npy", b2)
