# SNN-WSHE: Spiking Neural Network - Weighted Spatiotemporal Hybrid Encoding

> It is used for the study of weighted spatiotemporal hybrid coding of spiking neural networks
> 
> Beijing Jiaotong University BIR

## Core Features
 **Weighted spatiotemporal hybrid Encoding**
   - The floating-point value tensor is encoded as a discrete pulse tensor to fit the SNN network input
   - Adaptive adjustment of coding range-accuracy is realized in a limited time step
   - The original information is retained to the greatest extent and used in the input stage to improve the overall performance of the SNN model
   - Supports 2D tensors, general feature maps, and native RGB image encoding

2D tensor encoding:
![2D tensor encoding](.img/tensor_code.jpeg)

Graph encoding:
![Graph encoding](.img/graph_code.jpeg)
## Getting Started
### Install dependencies
The encoder was developed under pytorch2.0.0+cu118:
```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```
The SNN framework uses the commercial library Lynbidl, please contact sales@lynxi.com to obtain it. 
There is also an open source version of the library, and the repository address is: https://github.com/LynxiTech/BIDL

### Encoder usage example
```python
from SpikeEncoder_GPU import TorchSpikeEncoder
import torch
# Initialize the encoder
encoder = TorchSpikeEncoder(num_trains=9, num_steps=10).to('cuda')
# Sample input (batch=4, channel=3, 224x224)
input_tensor = torch.randn(4, 3, 224, 224).to('cuda') * 100000 # Verify the encoding under different ranges
spike_seq = encoder(input_tensor).cpu()  # Encoding
value = input_tensor[1, 1, 17, 14] # Select a value to decode for verification
print(f'origin value: {value}')
encoded_train = spike_seq[1, :, 1, 17 * 3: 17 * 3 + 3, 14 * 3: 14 * 3 + 3]
print(f'graph encoded shape: {spike_seq.shape}') # Output shapes: (4, 10, 3, 672, 672)
print(f'encoded value shape: {encoded_train.shape}\nencoded value: \n{encoded_train}')
decoded_value = encoder.ValueDecoder(encoded_train) # Decoding
print(f'decoded value: {decoded_value}')
encoder.PlotPulseSequence(encoded_train) # Visualization of coding results
```

### Training examples
```bash
python CIFAR10_Train.py --model WSHE
```
Note: In order to use the convolution kernel exactly against the pulse block during the input convolution operation, the convolution kernel size of the first convolution layer needs to be set to the same size as the pulse block, and the convolution step size is equal to the side length of the pulse block.
