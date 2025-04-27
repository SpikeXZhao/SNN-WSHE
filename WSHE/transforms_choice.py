import torchvision.transforms as transforms
import torch
import numpy as np


def transforms_choice(encoder='WSHE'):
    if encoder in ('WSHE', 'ann', 'poisson'):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, shear=15),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        return train_transform, test_transform
    elif encoder == 'phase':
        max_val = 1.0 - 1 / (2 ** 10)  # 0.9990234375
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, shear=15),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),

            # 新增：将数据缩放到 [0, max_val] 范围（适配K=10的相位编码）
            transforms.Lambda(lambda x: x.clamp(0.0, 1.0)),  # 先约束到标准[0,1]
            transforms.Lambda(lambda x: x * max_val),  # 线性缩放到目标范围
            # transforms.Lambda(lambda x: x.clamp(0.0, max_val))      # 更安全的二次约束（可选）
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),

            # 新增：将数据缩放到 [0, max_val] 范围
            transforms.Lambda(lambda x: x.clamp(0.0, 1.0)),  # 先约束到标准[0,1]
            transforms.Lambda(lambda x: x * max_val),  # 线性缩放到目标范围
            # transforms.Lambda(lambda x: x.clamp(0.0, max_val))      # 更安全的二次约束（可选）
        ])
        return train_transform, test_transform
    elif encoder == 'latency':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, shear=15),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),

            # 新增：将数据压缩回[0,1]范围
            transforms.Lambda(lambda x: x.clamp(0.0, 1.0))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.clamp(0.0, 1.0)),
        ])
        return train_transform, test_transform

    elif encoder == 'RGB':
        train_transform = transforms.Compose([
            # 空间增强（在PIL图像阶段处理）
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, shear=15),

            # 颜色增强（在PIL图像阶段处理）
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            ),

            # 转换为张量（保持0-255整数）
            transforms.Lambda(
                lambda img: torch.ByteTensor(np.array(img).transpose(2, 0, 1))
            ),

            # 张量增强（处理uint8类型）
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value=0  # 擦除值为0（黑色）
            )
        ])

        # 测试集预处理
        test_transform = transforms.Compose([
            # 直接转换为整数张量
            transforms.Lambda(
                lambda img: torch.ByteTensor(np.array(img).transpose(2, 0, 1))
            )
        ])
        return train_transform, test_transform
    else:
        raise ValueError(
            f"不支持的编码器类型: {encoder}。支持的编码器类型为 'HSHE', 'ann', 'poisson', 'phase', 'latency', 'RGB'。")