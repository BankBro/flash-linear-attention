import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os


# 导入你写的 KANActivation
from fla.modules.kan_activation import KANActivation


# 简单的网络结构，使用 KANActivation 作为隐藏层
class SimpleKANModel(nn.Module):
    def __init__(self):
        super(SimpleKANModel, self).__init__()
        self.kan1 = KANActivation(grid_size=8, spline_order=3, grid_range=[-np.pi * 1.5, np.pi * 1.5], init_strategy='linear_with_noise')
        # self.kan2 = KANActivation(grid_size=8, spline_order=3, grid_range=[-np.pi * 1.5, np.pi * 1.5], init_strategy='linear_with_noise')

    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        x = self.kan1(x)  # 应用 KAN 激活
        # x = self.kan2(x)
        return x


def rand_tensor(shape, a=-1.0, b=1.0):
    return (b - a) * torch.rand(shape) + a


# 生成数据：y = sin(x)
def generate_data(B, L, D):
    x = rand_tensor((B, L, D), -np.pi, np.pi)
    y = torch.sin(x)
    return x, y

def collect_kan_loss(model):
    total_kan_loss = 0.0
    for layer in model.children():
        if isinstance(layer, KANActivation):
            total_kan_loss += layer.current_loss
    return total_kan_loss


# 训练函数
def train(model, x_train, y_train, epochs=500, lr=1e-2, save_interval=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(x_train)
        loss = criterion(output, y_train)

        kan_loss = collect_kan_loss(model)

        # loss.backward()
        kan_loss.backward() 
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

        if (epoch + 1) % save_interval == 0:
            plot_results(model, x_train, y_train, save_path=f"./kan_plots/epoch_{epoch+1}.png")
            plot_activation(model.kan1, device, save_path=f"./kan_plots/activation_epoch_{epoch+1}_layer1.png")
            # plot_activation(model.kan2, device, save_path=f"./kan_plots/activation_epoch_{epoch+1}_layer2.png")

    return model


# 可视化预测结果
def plot_results(model, x_train, y_train, save_path):
    model.eval()
    with torch.no_grad():
        pred = model(x_train)

    x_np = x_train[0].flatten().cpu().numpy()
    y_true = y_train[0].flatten().cpu().numpy()
    y_pred = pred[0].flatten().cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.scatter(x_np, y_true, label="True Function (sin(x))", s=5, alpha=0.7)
    plt.scatter(x_np, y_pred, label="KAN Approximation", s=5, alpha=0.7, marker='x')
    plt.legend()
    plt.title("KANActivation Fitting sin(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    # print(f"Plot saved to {save_path}")


# 可视化激活函数
def plot_activation(kan_layer, device, save_path):
    B, L, D = 1, 100, 1
    num_points = B * L * D

    x = (
        torch.linspace(-np.pi, np.pi, num_points)
        .unsqueeze(0)
        .unsqueeze(-1)
        .reshape(B, L, D)
        .to(device)
    )

    with torch.no_grad():
        y = kan_layer(x)

    x_np = x.flatten().cpu().numpy()
    y_np = y.flatten().cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(x_np, y_np, label="KAN Activation")
    plt.title("Learned KAN Activation Function")
    plt.xlabel("Input x")
    plt.ylabel("Output")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    # print(f"Activation plot saved to {save_path}")


if __name__ == "__main__":
    os.makedirs("kan_plots", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleKANModel().to(device)
    print(model)

    plot_activation(model.kan1, device, save_path="./kan_plots/activation_epoch_0_layer1.png")
    # plot_activation(model.kan2, device, save_path="./kan_plots/activation_epoch_0_layer2.png")

    # 生成数据
    B, L, D = 4, 500, 16
    x_train, y_train = generate_data(B, L, D)
    x_train, y_train = x_train.to(device), y_train.to(device)

    # 训练模型
    trained_model = train(model, x_train, y_train, epochs=3000, lr=1e-2, save_interval=500)

    # 最终绘图展示结果
    plot_results(trained_model, x_train, y_train, save_path="./kan_plots/final_result.png")