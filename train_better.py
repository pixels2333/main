import numpy as np
import matplotlib.pyplot as plt
import torch.optim
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader
from torchvision import transforms

from CNN_model import *
from MyData import *


# 自定义MAPE损失函数
class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, predictions, targets):
        # 为了避免除以零，添加一个小的常数epsilon
        epsilon = 1e-8
        targets = targets.clamp(min=epsilon)

        # 计算MAPE
        loss = torch.mean(torch.abs(predictions - targets) / targets)
        return loss


# 定义R2决定系数
def r2_score(predictions, targets):
    predictions_mean = targets.mean()
    ss_total = torch.sum((targets - predictions_mean) ** 2)
    ss_residual = torch.sum((targets - predictions) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2.item()


# 定义NRMSE
def nrmse_score(predictions, targets, data_range=None):
    if data_range is None:
        data_range = targets.max() - targets.min()
    nrmse = torch.sqrt(torch.mean((predictions - targets) ** 2)) / data_range
    return nrmse.item()


def calculate_r2_and_nrmse(predictions, targets):
    # 计算MSE
    mse = ((predictions - targets) ** 2).mean()
    # 计算NRMSE
    nrmse = np.sqrt(mse) / (targets.max() - targets.min())
    # 计算R2分数
    ss_res = ((predictions - targets) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)

    return r2, nrmse


# 调用图片数据集
transform = transforms.ToTensor()
# 创建自定义Dataset实例
batch_size = 32
# 创建训练数据集实例
train_dataset = MyData(image_folder='T_V_img_data/train_183_4758',
                       excel_file='T_V_img_data/train_enlarge.xlsx', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = MyData(image_folder='T_V_img_data/val_46_1196',
                     excel_file='T_V_img_data/val_enlarge.xlsx', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
# 创建测试数据集实例
test_dataset = MyData(image_folder='Test',
                      excel_file='test.xlsx', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 引用神经网络并设定有关参数
cnn = CNN()
cnn = cnn.cuda()

# 损失函数
loss_fn = MAPELoss()  # 损失函数调整为MAPE
loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 1.5548648368932752e-05  # 学习率调整为0.01
optimizer = torch.optim.Adam(cnn.parameters(
), lr=learning_rate, weight_decay=2.9527008730298823e-05)  # 优化器调整为Adam算法并加入L1正则化

# 设置训练参数
total_train_step = 0
total_val_step = 0
epoch = 300

# 训练过程利用tensorboard实现可视化
writer = SummaryWriter("logs")


# 训练与验证部分
for i in range(epoch):
    print("----------第{}轮训练开始-----------".format(i+1))

    # 训练步骤开始

    cnn.train()
    for data in train_loader:
        imgs, train_targets = data
        imgs = imgs.cuda()
        train_targets = train_targets.cuda()

        outputs = cnn(imgs)
        loss = loss_fn(outputs, train_targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        total_train_step += 1

        if total_train_step % 20 == 0:
            print(f"Step: {total_train_step}, Train Loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    cnn.eval()
    with torch.no_grad():
        for data in val_loader:
            imgs, val_targets = data
            imgs = imgs.cuda()
            val_targets = val_targets.cuda()
            outputs = cnn(imgs)
            loss = loss_fn(outputs, val_targets)
            total_val_step += 1

            if total_val_step % 5 == 0:
                print(f"Step: {total_val_step}, Val Loss: {loss.item()}")
                writer.add_scalar("val_loss", loss.item(), total_val_step)


# 测试部分
print("----------开始测试-----------")

# 初始化三个预测值和真实值的列表
predictions_list = [[], [], []]  # 存储三个预测值
targets_list = [[], [], []]      # 存储三个真实值

# 测试步骤开始
cnn.eval()

with torch.no_grad():
    for data in test_loader:
        imgs, test_targets = data
        imgs = imgs.cuda()
        test_targets = test_targets.cuda()
        outputs = cnn(imgs)

        # 将预测值和真实值按列分别存储
        for j in range(outputs.size(1)):
            predictions_list[j].extend(outputs[:, j].cpu().numpy().flatten())
            targets_list[j].extend(test_targets[:, j].cpu().numpy().flatten())

# 对每组数据进行线性回归拟合并作图
for i in range(3):
    # 将列表转换为numpy数组
    predictions = np.array(predictions_list[i]).reshape(-1, 1)
    targets = np.array(targets_list[i]).squeeze()

    # 线性回归拟合
    regressor = LinearRegression()
    regressor.fit(predictions, targets)
    coefficients = regressor.coef_
    intercept = regressor.intercept_

    # 预测
    predicted_values = regressor.predict(predictions)

    # 转换为torch张量
    predictions_tensor = torch.tensor(predictions.flatten())
    targets_tensor = torch.tensor(targets)

    # 使用你的函数计算R2和NRMSE
    r2 = r2_score(predictions_tensor, targets_tensor)
    nrmse = nrmse_score(predictions_tensor, targets_tensor,
                        data_range=targets_tensor.max() - targets_tensor.min())

    # 作图
    plt.figure(figsize=(8, 6))
    plt.scatter(predictions, targets, color='blue', label='Actual data')
    plt.plot(predictions, predicted_values, color='red', linewidth=2,
             label=f'Fit: y={coefficients[0]:.2f}x+{intercept:.2f}')
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.title(f'Output {i+1} - R2: {r2:.4f}, NRMSE: {nrmse:.4f}')
    plt.legend()
    plt.show()
# 关闭TensorBoard writer
writer.close()
