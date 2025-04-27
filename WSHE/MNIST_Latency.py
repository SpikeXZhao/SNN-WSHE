import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmcls.models.layers.lif import FcLif, Conv2dLif
import wandb
from argparse import Namespace
from spikingjelly.activation_based import encoding


def parse_args():
    parser = argparse.ArgumentParser(description='Train MINIST SNN network')

    parser.add_argument('--dataset-path', type=str, default='../data',
                        help='Path to dataset')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optmizer for the training. (adam or SGD)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Leaning rate for the optimizer.')
    parser.add_argument('--step-size', type=int, default=50,
                        help='step size for the optimizer.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma')
    args = parser.parse_args()
    return args

class MNIST_Conv(nn.Module):
    def __init__(self, timestep=10, fmode='spike', cmode='spike', amode='mean', soma_params='all_share', noise=1e-3,
                 neuron='lif'):
        super(MNIST_Conv, self).__init__()
        neuron = neuron.lower()
        assert neuron in ['lif']
        self.clif1 = Conv2dLif(1, 64, kernel_size=3, padding=1, mode=cmode, soma_params=soma_params, noise=noise)
        self.clif2 = Conv2dLif(64, 128, kernel_size=3, padding=1, mode=cmode, soma_params=soma_params, noise=noise)
        self.clif3 = Conv2dLif(128, 256, kernel_size=3, padding=1, mode=cmode, soma_params=soma_params, noise=noise)
        self.flatten = nn.Flatten()
        self.fclif1 = FcLif(256 * 28 * 28, 256, mode=fmode, soma_params=soma_params, noise=noise)

        self.fclif2 = FcLif(256, 10, mode=fmode, soma_params=soma_params, noise=noise)

        assert amode == 'mean'
        self.tempAdd = None
        self.timestep = timestep

    def reset(self, xi):
        self.tempAdd = torch.zeros_like(xi)

    def forward(self, xis: torch.Tensor) -> torch.Tensor:
        xo = 0
        for i in range(self.timestep):
            if i == 0 : encoder.reset()
            x = encoder(xis)  # (B, D)
            if i == 0: self.clif1.reset(x)
            x = self.clif1(x)
            if i == 0: self.clif2.reset(x)
            x = self.clif2(x)
            if i == 0: self.clif3.reset(x)
            x = self.clif3(x)
            x = torch.flatten(x, start_dim=1)
            if i == 0: self.fclif1.reset(x)
            x = self.fclif1(x)
            if i == 0: self.fclif2.reset(x)
            x = self.fclif2(x)
            xo = xo + x / self.timestep
        return xo

class MINIST_FC(nn.Module):
    def __init__(self, timestep=10, fmode='spike', cmode='spike', amode='mean', soma_params='all_share', noise=1e-3,
                 neuron='lif'):
        super(MINIST_FC, self).__init__()
        neuron = neuron.lower()
        assert neuron in ['lif']
        self.flatten = nn.Flatten()
        self.fclif1 = FcLif(28 * 28, 256, mode=fmode, soma_params=soma_params, noise=noise)
        self.fclif2 = FcLif(256, 10, mode=fmode, soma_params=soma_params, noise=noise)

        assert amode == 'mean'
        self.tempAdd = None
        self.timestep = timestep

    def reset(self, xi):
        self.tempAdd = torch.zeros_like(xi)

    def forward(self, xis: torch.Tensor) -> torch.Tensor:
        xo = 0
        for i in range(self.timestep):
            if i == 0 : encoder.reset()
            x = encoder(xis)  # (B, D)
            x = torch.flatten(x, start_dim=1)
            if i == 0: self.fclif1.reset(x)
            x = self.fclif1(x)
            if i == 0: self.fclif2.reset(x)
            x = self.fclif2(x)
            xo = xo + x / self.timestep
        return xo


# 训练函数
def train(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        label_onehot = F.one_hot(labels, 10).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.mse_loss(outputs, label_onehot)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

# 验证函数
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            label_onehot = F.one_hot(labels, 10).float()
            outputs = model(inputs)
            loss = F.mse_loss(outputs, label_onehot)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

if __name__ == '__main__':
    args = parse_args()

    config = Namespace(
        project_name = 'MINIST',
        batch_size = args.batch_size,
        epochs = args.epochs,
        optimizer = args.optim,
        learning_rate = args.learning_rate,
        step_size = args.step_size,
        gamma = args.gamma
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.clamp(0.0, 1.0)),
    ])

    # 加载数据集
    train_dataset = torchvision.datasets.MNIST(
        root=args.dataset_path,
        train=True,
        transform=transform,
        download=False
    )

    test_dataset = torchvision.datasets.MNIST(
        root=args.dataset_path,
        train=False,
        transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False
    )

    # 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # encoder = encoding.PoissonEncoder()
    encoder = encoding.LatencyEncoder(T=10, enc_function='linear')
    model = MNIST_Conv().to(device)
    # model = MINIST_FC().to(device)
    criterion = nn.CrossEntropyLoss()
    if config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    wandb.init(
        project=config.project_name,
        config=config.__dict__
    )
    print(config)

    # 训练循环
    best_acc = 0.0
    best_vacc = 0.0
    best_epoch = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, test_loader, criterion)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        wandb.log({'Epoch':epoch, 'Train Loss': train_loss, 'Train Acc': train_acc, 'Val Loss': val_loss, 'Val Acc': val_acc})

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./trained_model/MINIST_Latency_Conv_%02d_%0.2f.pth"%(epoch, val_acc))
            best_epoch, best_vacc = epoch, val_acc
            print("------------------------------------Saved best model!-----------------------------------------")

    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load("./trained_model/MINIST_Latency_Conv_%02d_%0.2f.pth"%(best_epoch, best_vacc)))
    final_val_loss, final_val_acc = validate(model, test_loader, criterion)
    print(f"\nFinal Test Accuracy: {final_val_acc:.2f}%")