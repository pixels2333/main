import numpy as np
import optuna
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from CNN_model import *  # 假设这是你的CNN模型定义
from MyData import *  # 假设这是你的自定义数据集定义


# 自定义MAPE损失函数和评估指标函数保持不变
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


# 定义超参数调优的目标函数
def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义超参数搜索空间
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)

    # 实例化模型、损失函数和优化器
    cnn = CNN().to(device)
    loss_fn = MAPELoss().to(device)
    optimizer = optim.Adam(
        cnn.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 数据加载
    transform = transforms.ToTensor()
    train_dataset = MyData(image_folder='T_V_img_data/train_183_4758',
                           excel_file='T_V_img_data/train_enlarge.xlsx',
                           transform=transform)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = MyData(image_folder='T_V_img_data/val_46_1196',
                         excel_file='T_V_img_data/val_enlarge.xlsx',
                         transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    total_train_step = 0
    total_val_step = 0

    # 训练和验证循环
    for epoch in range(300):  # 假设我们只训练10个epoch作为示例
        print(f"----------Epoch {epoch + 1}----------")
        cnn.train()
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = cnn(imgs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 20 == 0:
                loss_value = loss.item()
                print(f"Step: {total_train_step}, Train Loss: {loss_value}")

        cnn.eval()
        with torch.no_grad():
            val_loss = 0
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = cnn(imgs)
                val_loss += loss_fn(outputs, targets).item()

            val_loss /= len(val_loader)
            print(f"Epoch {epoch + 1}, Val Loss: {val_loss}")

    # 返回验证损失作为优化目标
    return val_loss


# 创建Optuna study对象
study = optuna.create_study(direction="minimize")

# 执行优化，n_trials表示要进行的试验次数
n_trials = 10
study.optimize(objective, n_trials=n_trials)

# 打印最佳超参数
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
params = trial.params
for key, value in params.items():
    print(f"    {key}: {value}")
