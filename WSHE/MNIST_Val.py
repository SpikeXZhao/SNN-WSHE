import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import CIFAR10_models
from SpikeEncoder_GPU import TorchSpikeEncoder
from spikingjelly.activation_based import encoding
from transforms_choice import transforms_choice
import numpy as np
import matplotlib.pyplot as plt
import random
from utils import globals
from MNIST_models import model_choice
globals._init()


def parse_args():
    parser = argparse.ArgumentParser(description='MNIST SNN Inference')
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['ann', 'WSHE', 'poisson', 'phase', 'latency', 'RGB'],
                        help='Model type used for inference')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model file')
    parser.add_argument('--is-FC', type=bool, default=False,
                        help='Use MLP')
    parser.add_argument('--dataset-path', type=str, default='../data',
                        help='Path to MNIST dataset')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for inference')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of sample predictions to display')
    return parser.parse_args()


def load_model(model_type, model_path):
    # 加载模型参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, encoder = model_choice(model_type, device, args.is_FC)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model.to(device)
    model.eval()
    return model, encoder

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

def load_dataset(model_type, args):
    # 加载MNIST测试集

    transform = transform_choice(model_type)
    test_dataset = torchvision.datasets.MNIST(
        root='../data',
        train=False,
        transform=transform
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    return test_loader, test_dataset  # 同时返回Loader和原始数据集


def dataset_inference(model, encoder, test_loader, model_type, num_samples=10):
    device = next(model.parameters()).device
    classes = ('0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9')

    correct = 0
    total = 0
    sample_count = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Dataset Inference"):
            inputs, labels = inputs.to(device), labels.to(device)

            # 应用编码器
            if model_type in ['WSHE', 'RGB']:
                inputs = encoder(inputs)
            elif model_type == 'poisson':
                inputs = inputs.repeat(10, 1, 1, 1, 1).transpose(0, 1)
                inputs = encoder(inputs)

            outputs = model(inputs)

            # 处理时序输出
            if model_type != 'ann':
                outputs = outputs.mean(1) if outputs.ndim == 5 else outputs  # 时间维度平均

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 显示部分样本结果
            if sample_count < num_samples:
                for i in range(min(inputs.size(0), num_samples - sample_count)):
                    print(f"\nSample {sample_count + 1}:")
                    print(f"True label: {classes[labels[i]]}")
                    print(f"Predicted: {classes[predicted[i]]}")
                    sample_count += 1

    accuracy = 100 * correct / total
    print(f"\nTotal Accuracy on Test Set: {accuracy:.2f}%")
    return accuracy


def single_inference(model, encoder, test_dataset, model_type):
    device = next(model.parameters()).device
    classes = ('0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9')

    # 随机选择样本
    index = random.randint(0, len(test_dataset) - 1)
    raw_image, true_label = test_dataset[index]


    # 复制预处理流程
    input_tensor = raw_image.unsqueeze(0).to(device)  # 添加batch维度
    print(input_tensor.shape)
    # 应用编码器（与dataset_inference保持一致）
    if model_type in ['WSHE', 'RGB']:
        input_tensor = encoder(input_tensor)  # 添加时间维度
    elif model_type == 'poisson':
        input_tensor = input_tensor.repeat(10, 1, 1, 1, 1).transpose(0, 1)
        input_tensor = encoder(input_tensor)

    # 执行推理
    with torch.no_grad():
        outputs = model(input_tensor)
        if model_type != 'ann':
            outputs = outputs
            print(outputs)
        _, predicted = torch.max(outputs, 1)

    # 可视化结果
    plt.figure(figsize=(6, 6))

    # 反标准化处理
    mean = 0.1307
    std = 0.3081
    image = raw_image.numpy().transpose((1, 2, 0))
    image = std * image + mean
    image = np.clip(image, 0, 1)

    plt.imshow(image)
    plt.title(f"True: {classes[true_label]}\nPredicted: {classes[predicted.item()]}",
              fontsize=12, pad=10)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return predicted.item()


if __name__ == '__main__':
    args = parse_args()

    # 加载模型和数据集
    model, encoder = load_model(args.model_type, args.model_path)
    test_loader, test_dataset = load_dataset(args.model_type, args)
    single_inference(model, encoder, test_dataset, args.model_type)
    # 执行数据集推理
    dataset_inference(model, encoder, test_loader, args.model_type, args.num_samples)