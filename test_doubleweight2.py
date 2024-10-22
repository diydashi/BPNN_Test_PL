import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 读取数据
data1 = pd.read_excel('data1.xlsx')  # 仿真数据
data2 = pd.read_excel('data2.xlsx')  # 实测数据

# 提取输入特征（前三列）和标签（第四列）
X_sim = data1.iloc[:, :-1].values  # 仿真数据输入
y_sim = data1.iloc[:, -1].values  # 仿真数据标签

X_real = data2.iloc[:, :-1].values  # 实测数据输入
y_real = data2.iloc[:, -1].values  # 实测数据标签

# 数据标准化处理
scaler = StandardScaler()
X_sim = scaler.fit_transform(X_sim)
X_real = scaler.transform(X_real)

# 转换为 PyTorch 张量
X_sim = torch.tensor(X_sim, dtype=torch.float32)
y_sim = torch.tensor(y_sim, dtype=torch.float32).view(-1, 1)
X_real = torch.tensor(X_real, dtype=torch.float32)
y_real = torch.tensor(y_real, dtype=torch.float32).view(-1, 1)


# 定义自定义双权重神经元层
class DualWeightNeuron(nn.Module):
    def __init__(self, input_size, output_size, alpha=0.4):
        super(DualWeightNeuron, self).__init__()
        self.w1 = nn.Parameter(torch.randn(input_size, output_size))  # 仿真数据的权重
        self.w2 = nn.Parameter(torch.randn(input_size, output_size))  # 实测数据的权重
        self.b = nn.Parameter(torch.zeros(output_size))  # 偏置项
        self.alpha = alpha  # 权重比重

    def forward(self, x):
        # 结合 w1 和 w2，按照 alpha 进行加权求和
        output = self.alpha * (x @ self.w1) + (1 - self.alpha) * (x @ self.w2) + self.b
        return output


# 定义包含三层隐藏层的模型
class DualWeightModel(nn.Module):
    def __init__(self, input_size, alpha=0.4):
        super(DualWeightModel, self).__init__()
        # 第一隐藏层：输入 -> 32 个神经元
        self.fc1 = DualWeightNeuron(input_size, 32, alpha)
        # 第二隐藏层：32 -> 128 个神经元
        self.fc2 = DualWeightNeuron(32, 128, alpha)
        # 第三隐藏层：128 -> 256 个神经元
        self.fc3 = DualWeightNeuron(128, 256, alpha)
        # 输出层：256 个神经元 -> 1 个输出（传播损耗）
        self.out = DualWeightNeuron(256, 1, alpha)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # 第一层 + 激活
        x = self.relu(self.fc2(x))  # 第二层 + 激活
        x = self.relu(self.fc3(x))  # 第三层 + 激活
        x = self.out(x)  # 输出层
        return x


# 实例化模型
model = DualWeightModel(input_size=3, alpha=0.4)  # 输入是3个参数（距离、俯仰角度、频率）

# 定义损失函数和优化器
criterion = nn.MSELoss()

# 训练 w1，用仿真数据
optimizer_w1 = optim.Adam(
    [model.fc1.w1, model.fc2.w1, model.fc3.w1, model.out.w1, model.fc1.b, model.fc2.b, model.fc3.b, model.out.b],
    lr=0.01)  # 只训练 w1 和 b
num_epochs = 100

# 阶段1: 训练 w1
for epoch in range(num_epochs):
    model.train()
    optimizer_w1.zero_grad()
    outputs = model(X_sim)
    loss = criterion(outputs, y_sim)
    loss.backward()
    optimizer_w1.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 冻结 w1，训练 w2
optimizer_w2 = optim.Adam(
    [model.fc1.w2, model.fc2.w2, model.fc3.w2, model.out.w2, model.fc1.b, model.fc2.b, model.fc3.b, model.out.b],
    lr=0.01)  # 只训练 w2 和 b

# 阶段2: 训练 w2
for epoch in range(num_epochs):
    model.train()
    optimizer_w2.zero_grad()
    outputs = model(X_real)
    loss = criterion(outputs, y_real)
    loss.backward()
    optimizer_w2.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 预测
model.eval()
with torch.no_grad():
    # 示例输入：距离200米，俯仰角30度，频率2.4GHz
    test_input = torch.tensor([[200, 30, 2.4]], dtype=torch.float32)
    test_input = scaler.transform(test_input)  # 记得标准化
    test_input = torch.tensor(test_input, dtype=torch.float32)

    predicted_loss = model(test_input)
    print(f'Predicted Loss: {predicted_loss.item():.4f}')