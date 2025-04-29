# SNN-WSHE: Spiking Neural Network - Weighted Spatiotemporal Hybrid Encoding

> 用于脉冲神经网络时空编码研究的开源框架
> 北京交通大学 BIR

## 核心功能
 **加权时空混合编码**
   - 将浮点值张量编码为离散脉冲张量以适配SNN网络输入
   - 在有限时间步下实现编码范围-精度自适应调节
   - 最大程度保留原始信息，在输入阶段使用以提升SNN模型整体性能
   - 支持二维张量、一般特征图、原始RGB图像编码

二维张量编码：
![二维张量编码](.img/tensor_code.jpeg)

图编码：
![图编码](.img/graph_code.jpeg)
## 快速开始
### 安装依赖
编码器在pytorch2.0.0+cu118下开发：
```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```
SNN框架使用了商业库Lynbidl，获取请联系sales@lynxi.com。 该库也存在开源版本，仓库地址为：https://github.com/LynxiTech/BIDL

### 编码器使用示例
```python
from SpikeEncoder_GPU import TorchSpikeEncoder
import torch
# 初始化编码器
encoder = TorchSpikeEncoder(num_trains=9, num_steps=10).to('cuda')
# 示例输入 (batch=4, channel=3, 224x224)
input_tensor = torch.randn(4, 3, 224, 224).to('cuda') * 100000 # 验证不同范围下的编码
spike_seq = encoder(input_tensor).cpu()  # 编码
value = input_tensor[1, 1, 17, 14] # 选择一个值解码验证
print(f'origin value: {value}')
encoded_train = spike_seq[1, :, 1, 17 * 3: 17 * 3 + 3, 14 * 3: 14 * 3 + 3]
print(f'graph encoded shape: {spike_seq.shape}') # 输出形状: (4, 10, 3, 672, 672)
print(f'encoded value shape: {encoded_train.shape}\nencoded value: \n{encoded_train}')
decoded_value = encoder.ValueDecoder(encoded_train) # 解码
print(f'decoded value: {decoded_value}')
encoder.PlotPulseSequence(encoded_train) # 编码结果可视化
```

### 训练示例
```bash
python CIFAR10_Train.py --model WSHE
```
注意：为了在输入的卷积操作时将卷积核与脉冲块严格对其，需要将第一层卷积层的卷积核大小设置为脉冲块一样的大小，卷积步长等于脉冲块的边长。
