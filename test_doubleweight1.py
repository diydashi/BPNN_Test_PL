import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler


# 1. 读取和预处理数据
def load_and_preprocess_data(simulation_file, real_file):
    # 读取 Excel 数据
    data1 = pd.read_excel(simulation_file)  # 仿真数据
    data2 = pd.read_excel(real_file)  # 实测数据

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

    return X_sim, y_sim, X_real, y_real, scaler


# 2. 定义双权重神经网络层
class DualWeightNeuron(nn.Module):
    def __init__(self, input_size, alpha=0.4):
        super(DualWeightNeuron, self).__init__()
        self.w1 = nn.Parameter(torch.randn(input_size, 1))  # 仿真数据的权重
        self.w2 = nn.Parameter(torch.randn(input_size, 1))  # 实测数据的权重
        self.b = nn.Parameter(torch.zeros(1))  # 偏置项
        self.alpha = alpha  # 权重比重

    def forward(self, x):
        # 结合 w1 和 w2，按照 alpha 进行加权求和
        output = self.alpha * (x @ self.w1) + (1 - self.alpha) * (x @ self.w2) + self.b
        return output


# 3. 定义包含双权重神经元的模型
class DualWeightModel(nn.Module):
    def __init__(self, input_size):
        super(DualWeightModel, self).__init__()
        self.fc = DualWeightNeuron(input_size)  # 使用自定义双权重神经元

    def forward(self, x):
        return self.fc(x)


# 4. 训练模型的函数
def train_model(model, X_train, y_train, optimizer, criterion, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# 5. 预测函数
def predict(model, test_input, scaler):
    model.eval()
    with torch.no_grad():
        test_input = scaler.transform(test_input)  # 标准化处理
        test_input = torch.tensor(test_input, dtype=torch.float32)
        predicted_loss = model(test_input)
        return predicted_loss.item()


# 6. 主程序
def main():
    # 读取并预处理数据
    X_sim, y_sim, X_real, y_real, scaler = load_and_preprocess_data('data1.xlsx', 'data2.xlsx')

    # 定义模型
    model = DualWeightModel(input_size=3)  # 输入是3个参数

    # 定义损失函数
    criterion = nn.MSELoss()

    # 阶段1：训练 w1，用仿真数据
    print("开始训练 w1（仿真数据）...")
    optimizer_w1 = optim.Adam([model.fc.w1, model.fc.b], lr=0.01)  # 只训练 w1 和 b
    train_model(model, X_sim, y_sim, optimizer_w1, criterion, num_epochs=100)

    # 阶段2：冻结 w1，训练 w2
    print("开始训练 w2（实测数据）...")
    optimizer_w2 = optim.Adam([model.fc.w2, model.fc.b], lr=0.01)  # 只训练 w2 和 b
    train_model(model, X_real, y_real, optimizer_w2, criterion, num_epochs=100)

    # 示例预测
    test_input = [[200, 30, 2.4]]  # 示例输入
    predicted_loss = predict(model, test_input, scaler)
    print(f'Predicted Loss: {predicted_loss:.4f}')


if __name__ == '__main__':
    main()