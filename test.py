# 库的导入
import numpy as np
import pandas as pd


# 激活函数tanh
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


# 输入数据的导入，用于测试数据的归一化与返归一化
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
samplein = np.matrix([Co, Cr, Mg, Pb])

sampleinminmax = np.array([samplein.min(axis=1).T.tolist()[0], samplein.max(axis=1).T.tolist()[0]]).transpose()
sampleout = np.matrix([Ti])
sampleoutminmax = np.array([sampleout.min(axis=1).T.tolist()[0], sampleout.max(axis=1).T.tolist()[0]]).transpose()

# 导入训练的权值、阈值
w1 = np.load('w1.npy')
w2 = np.load('w2.npy')
b1 = np.load('b1.npy')
b2 = np.load('b2.npy')

# 测试数据的导入
df = pd.read_csv("test.csv")
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
input = np.matrix([Co, Cr, Mg, Pb])

# 测试数据数量
testnum = 24

# 测试数据中输入数据的归一化
inputnorm = (2 * (np.array(input.T) - sampleinminmax.transpose()[0]) / (
            sampleinminmax.transpose()[1] - sampleinminmax.transpose()[0]) - 1).transpose()
# 隐含层、输出层的计算
hiddenout = tanh((np.dot(w1, inputnorm).transpose() + b1.transpose())).transpose()
networkout = tanh((np.dot(w2, hiddenout).transpose() + b2.transpose())).transpose()
# 对输出结果进行反归一化
diff = sampleoutminmax[:, 1] - sampleoutminmax[:, 0]
networkout2 = (networkout + 1) / 2
networkout2 = networkout2 * diff + sampleoutminmax[0][0]
output1 = networkout2.flatten()
output1 = output1.tolist()
for i in range(testnum):
    output1[i] = float('%.2f' % output1[i])
print("the prediction is:", output1)

# 将输出结果与真实值进行对比，计算误差
output = Ti
rmse = (np.sum(np.square(output - output1)) / len(output)) ** 0.5
mae = np.sum(np.abs(output - output1)) / len(output)
average_loss1 = np.sum(np.abs((output - output1) / output)) / len(output)
mape = "%.2f%%" % (average_loss1 * 100)
f1 = 0
for m in range(testnum):
    f1 = f1 + np.abs(output[m] - output1[m]) / ((np.abs(output[m]) + np.abs(output1[m])) / 2)
f2 = f1 / testnum
smape = "%.2f%%" % (f2 * 100)
print("the MAE is :", mae)
print("the RMSE is :", rmse)
print("the MAPE is :", mape)
print("the SMAPE is :", smape)

# 计算预测值与真实值误差与真实值之比的分布
A = 0
B = 0
C = 0
D = 0
E = 0
for m in range(testnum):
    y1 = np.abs(output[m] - output1[m]) / np.abs(output[m])
    if y1 <= 0.1:
        A = A + 1
    elif y1 > 0.1 and y1 <= 0.2:
        B = B + 1
    elif y1 > 0.2 and y1 <= 0.3:
        C = C + 1
    elif y1 > 0.3 and y1 <= 0.4:
        D = D + 1
    else:
        E = E + 1
print("Ratio <= 0.1 :", A)
print("0.1< Ratio <= 0.2 :", B)
print("0.2< Ratio <= 0.3 :", C)
print("0.3< Ratio <= 0.4 :", D)
print("Ratio > 0.4 :", E)
