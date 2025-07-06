import torch
import torch.nn as nn
from layer_gelu import GELU

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        # 定义多层网络，包含5层线性层和激活函数
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        # 遍历每一层
        for layer in self.layers:
            # 当前层的输出
            layer_output = layer(x)
            # 检查是否可以应用残差连接
            if self.use_shortcut and x.shape == layer_output.shape:
                # 如果可以，添加残差连接
                x = x + layer_output
            else:
                x = layer_output
        return x
    

def print_gradients(model, x):
    # 打印模型中每个参数的梯度
    output = model(x)
    target = torch.tensor([0.])
    loss = nn.MSELoss()
    loss = loss(output, target)

    # 反向传播
    loss.backward()
    # 打印梯度
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

if __name__ == "__main__":
    layer_sizes = [3,3,3,3,3,1]
    simple_input = torch.tensor([1., 0., -1.])

    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
    print_gradients(model_without_shortcut, simple_input)

    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
    print_gradients(model_with_shortcut, simple_input)