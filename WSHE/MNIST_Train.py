import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from argparse import Namespace
from MNIST_models import model_choice




def parse_args():
    parser = argparse.ArgumentParser(description='Train MINIST SNN network')

    parser.add_argument('--model', type=str, default='latency',
                        help='Model type, you can choice WSHE, poisson, phase, latency , RGB and ann')
    parser.add_argument('--dataset-path', type=str, default='../data',
                        help='Path to dataset')
    parser.add_argument('--is-FC', type=bool, default=False,
                        help='Use MLP')
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


# 训练函数
def train(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        inputs, target = data.to(device), target.to(device)
        label_onehot = F.one_hot(target, 10).float()
        if args.model == 'poisson':
            inputs = data.repeat(10, 1, 1, 1, 1).transpose(0, 1)
            inputs = encoder(inputs)
        elif args.model in ['WSHE', 'RGB']:
            inputs = encoder(inputs).to(device)

        outputs = model(inputs)
        # loss = criterion(outputs, labels)
        loss = F.mse_loss(outputs, label_onehot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

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
            if args.model == 'poisson':
                inputs = inputs.repeat(10, 1, 1, 1, 1).transpose(0, 1)
                inputs = encoder(inputs)
            elif args.model in ['WSHE', 'RGB']:
                inputs = encoder(inputs)
            outputs = model(inputs)
            # loss = criterion(outputs, labels)
            loss = F.mse_loss(outputs, label_onehot)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def transform_choice(encoder):
    if encoder in ('WSHE', 'ann', 'poisson'):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
        ])
        return transform
    elif encoder == 'phase':
        max_val = 1.0 - 1 / (2 ** 10)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.clamp(0.0, 1.0)),  # 先约束到标准[0,1]
            transforms.Lambda(lambda x: x * max_val),  # 线性缩放到目标范围
        ])
        return transform
    elif encoder == 'latency':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.clamp(0.0, 1.0))
        ])
        return transform



if __name__ == '__main__':
    args = parse_args()

    config = Namespace(
        project_name = 'MINIST',
        encoder = args.model,
        batch_size = args.batch_size,
        epochs = args.epochs,
        optimizer = args.optim,
        learning_rate = args.learning_rate,
        step_size = args.step_size,
        gamma = args.gamma
    )

    # 数据预处理
    transform = transform_choice(args.model)
    # 加载数据集
    train_dataset = torchvision.datasets.MNIST(
        root='../data',
        train=True,
        transform=transform,
        download=False
    )

    test_dataset = torchvision.datasets.MNIST(
        root='../data',
        train=False,
        transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, encoder = model_choice(args.model, device, is_FC=args.is_FC)

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
            torch.save(model.state_dict(), f"./trained_model/MINIST_{args.model}_{epoch}_{val_acc}.pth")
            torch.save(model, f"./trained_model/Full_MINIST_{args.model}_best.pth")
            best_epoch, best_vacc = epoch, val_acc
            print("------------------------------------Saved best model!-----------------------------------------")

    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load(f"./trained_model/MINIST_{args.model}_{epoch}_{val_acc}.pth"))
    final_val_loss, final_val_acc = validate(model, test_loader, criterion)
    print(f"\nFinal Test Accuracy: {final_val_acc:.2f}%")