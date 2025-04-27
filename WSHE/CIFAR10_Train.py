import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from argparse import Namespace
from CIFAR10_models import model_choice
from transforms_choice import transforms_choice



def parse_args():
    parser = argparse.ArgumentParser(description='Train CIFAR10 network')

    parser.add_argument('--model', type=str, default='poisson',
                        help='Model type, you can choice WSHE, poisson, phase, latency , RGB and ann')
    parser.add_argument('--dataset-path', type=str, default='../data',
                        help='Path to dataset')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Training epochs')
    parser.add_argument('--optim', type=str, default='sgd',
                        help='Optmizer for the training. (adam or SGD)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Leaning rate for the optimizer.')
    parser.add_argument('--step-size', type=int, default=250,
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

    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        label_onehot = F.one_hot(labels, 10).float()
        if args.model == 'poisson':
            inputs = inputs.repeat(10, 1, 1, 1, 1).transpose(0, 1)
            inputs = encoder(inputs)
        elif args.model in ['WSHE', 'RGB']:
            inputs = encoder(inputs)
        optimizer.zero_grad()
        outputs = model(inputs)
        # loss = criterion(outputs, labels)
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

if __name__ == '__main__':
    args = parse_args()

    config = Namespace(
        project_name = 'CIFAR10',
        encoder = args.model,
        batch_size = args.batch_size,
        epochs = args.epochs,
        optimizer = args.optim,
        learning_rate = args.learning_rate,
        step_size = args.step_size,
        gamma = args.gamma
    )

    train_transform, test_transform = transforms_choice(args.model)

    # 加载数据集
    train_set = torchvision.datasets.CIFAR10(
        root=args.dataset_path,
        train=True,
        download=True,
        transform=train_transform
    )

    test_set = torchvision.datasets.CIFAR10(
        root=args.dataset_path,
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8
    )

    # 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, encoder = model_choice(args.model, device)

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
            torch.save(model.state_dict(), f"./trained_model/CIFAR10_{args.model}_{epoch}_{val_acc}.pth")
            torch.save(model, f"./trained_model/Full_CIFAR10_{args.model}_best.pth")
            best_epoch, best_vacc = epoch, val_acc
            print("------------------------------------Saved best model!-----------------------------------------")

    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load(f"./trained_model/CIFAR10_{args.model}_{epoch}_{val_acc}.pth"))
    final_val_loss, final_val_acc = validate(model, test_loader, criterion)
    print(f"\nFinal Test Accuracy: {final_val_acc:.2f}%")