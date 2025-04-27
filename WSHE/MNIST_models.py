import torch
import torch.nn as nn
from utils import globals
from lynadapter.warp_load_save import load,save,load_kernel,save_kernel
from mmcls.models.layers.lif import FcLif, Conv2dLif
from spikingjelly.activation_based import encoding
import SpikeEncoder_GPU


class MNIST_FC_WSHE(nn.Module):

    def __init__(self,
                 timestep=10, c0=1, fclass=10, cmode='spike', fmode='spike', amode='mean',
                 noise=1e-3, neurons_num=7
                 ):
        super(MNIST_FC_WSHE, self).__init__()
        self.neurons_num = neurons_num
        self.fclass = fclass
        self.norm = nn.BatchNorm2d(c0)
        self.flat = nn.Flatten(1, -1)
        self.flif1 = FcLif(28 * 28 * 9, 256, mode=fmode, noise=noise)
        self.flif2 = FcLif(256, 10, mode=fmode, noise=noise)
        assert amode == 'mean'
        self.timestep = timestep
        self.ON_APU = globals.get_value('ON_APU')
        self.FIT = globals.get_value('FIT')


    def reset(self, xi):
        self.tempAdd = torch.zeros_like(xi)

    def forward(self, xis: torch.Tensor) -> torch.Tensor:
        if self.ON_APU:
            x = self.norm(xis)
            x = self.flat(x)
            self.flif1.reset(x)
            x = self.flif1(x)
            self.flif2.reset()
            x = self.flif2(x)
            x = x.unsqueeze(2).unsqueeze(3)
            self.reset(x)
            self.tempAdd = load(self.tempAdd, f'tempAdd')
            self.tempAdd = self.tempAdd + x / self.timestep
            output = self.tempAdd.clone()
            save(self.tempAdd, f'tempAdd')
            return output.squeeze(-1).squeeze(-1)

        else:
            t = xis.size(1)
            xo = 0
            for i in range(t):
                x = xis[:, i, ...]
                x = self.norm(x)
                x = self.flat(x)
                if i == 0: self.flif1.reset(x)
                x = self.flif1(x)
                if i == 0: self.flif2.reset(x)
                x = self.flif2(x)
                xo = xo + x / t
            return xo

class MNIST_CSNN_WSHE(nn.Module):

    def __init__(self,
                 timestep=10, c0=1, fclass=10, cmode='spike', fmode='spike', amode='mean',
                 noise=1e-3, neurons_num=7
                 ):
        super(MNIST_CSNN_WSHE, self).__init__()
        self.neurons_num = neurons_num
        self.fclass = fclass
        self.norm = nn.BatchNorm2d(c0)
        self.clif1 = Conv2dLif(c0, 64, 3, stride=3, feed_back=False, mode=cmode, noise=noise)
        self.clif2 = Conv2dLif(64, 128, 3, padding=1, feed_back=False, mode=cmode, noise=noise)
        self.clif3 = Conv2dLif(128, 256, 3, padding=1, feed_back=False, mode=cmode, noise=noise)
        self.flat = nn.Flatten(1, -1)
        self.flif1 = FcLif(28 * 28 * 256, 256, mode=fmode, noise=noise)
        self.flif2 = FcLif(256, 10, mode=fmode, noise=noise)
        assert amode == 'mean'
        self.timestep = timestep
        self.ON_APU = globals.get_value('ON_APU')
        self.FIT = globals.get_value('FIT')


    def reset(self, xi):
        self.tempAdd = torch.zeros_like(xi)

    def forward(self, xis: torch.Tensor) -> torch.Tensor:
        if self.ON_APU:
            x0 = self.norm(xis)
            # x0 = xis
            self.clif1.reset(x0)
            x1 = self.clif1(x0)
            self.clif2.reset(x1)
            x2 = self.clif2(x1)
            self.clif3.reset(x2)
            x3 = self.clif3(x2)
            x4 = self.flat(x3)
            self.flif1.reset(x4)
            x5 = self.flif1(x4)
            self.flif2.reset(x5)
            x6 = self.flif2(x5)
            x6 = x6.unsqueeze(2).unsqueeze(3)
            self.reset(x6)
            self.tempAdd = load(self.tempAdd, f'tempAdd')
            self.tempAdd = self.tempAdd + x6 / self.timestep
            output = self.tempAdd.clone()
            save(self.tempAdd, f'tempAdd')
            return output.squeeze(-1).squeeze(-1)

        else:
            t = xis.size(1)
            xo_list = []
            xo = 0
            for i in range(t):
                x0 = xis[:, i, ...]
                x0 = self.norm(x0)
                if i == 0: self.clif1.reset(x0)
                x1 = self.clif1(x0)
                if i == 0: self.clif2.reset(x1)
                x2 = self.clif2(x1)
                if i == 0: self.clif3.reset(x2)
                x3 = self.clif3(x2)
                x4 = self.flat(x3)
                if i == 0: self.flif1.reset(x4)
                x5 = self.flif1(x4)
                if i == 0: self.flif2.reset(x5)
                x6 = self.flif2(x5)
                xo = xo + x6 / t
            return xo

class MINIST_CSNN(nn.Module):

    def __init__(self,
                 timestep=10, c0=1, fclass=10, cmode='spike', fmode='spike', amode='mean',
                 noise=1e-3, neurons_num=7, encoder=None
                 ):
        super(MINIST_CSNN, self).__init__()
        self.neurons_num = neurons_num
        self.fclass = fclass
        self.norm = nn.BatchNorm2d(c0)
        self.clif1 = Conv2dLif(c0, 64, 3, padding=1, feed_back=False, mode=cmode, noise=noise)
        self.clif2 = Conv2dLif(64, 128, 3, padding=1, feed_back=False, mode=cmode, noise=noise)
        self.clif3 = Conv2dLif(128, 256, 3, padding=1, feed_back=False, mode=cmode, noise=noise)
        self.flat = nn.Flatten(1, -1)
        self.flif1 = FcLif(28 * 28 * 256, 256, mode=fmode, noise=noise)
        self.flif2 = FcLif(256, 10, mode=fmode, noise=noise)
        assert amode == 'mean'
        self.timestep = timestep
        self.ON_APU = globals.get_value('ON_APU')
        self.FIT = globals.get_value('FIT')
        if self.ON_APU is not True:
            if encoder == 'latency':
                self.encoder = encoding.LatencyEncoder(T=10, enc_function='linear')
            elif encoder == 'phase':
                self.encoder = encoding.WeightedPhaseEncoder(K=10)
            else:
                self.encoder = None

    def reset(self, xi):
        self.tempAdd = torch.zeros_like(xi)

    def forward(self, xis: torch.Tensor) -> torch.Tensor:
        if self.ON_APU:
            x0 = self.norm(xis)
            # x0 = xis
            self.clif1.reset(x0)
            x1 = self.clif1(x0)
            self.clif2.reset(x1)
            x2 = self.clif2(x1)
            self.clif3.reset(x2)
            x3 = self.clif3(x2)
            x4 = self.flat(x3)
            self.flif1.reset(x4)
            x5 = self.flif1(x4)
            self.flif2.reset(x5)
            x6 = self.flif2(x5)
            x6 = x6.unsqueeze(2).unsqueeze(3)
            self.reset(x6)
            self.tempAdd = load(self.tempAdd, f'tempAdd')
            self.tempAdd = self.tempAdd + x6 / self.timestep
            output = self.tempAdd.clone()
            save(self.tempAdd, f'tempAdd')
            return output.squeeze(-1).squeeze(-1)

        else:
            xo = 0
            for i in range(self.timestep):
                if self.encoder is not None:
                    if i == 0: self.encoder.reset()
                    x = self.encoder(xis)  # (B, D)
                else:
                    x = xis[:, i, ...]
                x = self.norm(x)
                if i == 0: self.clif1.reset(x)
                x1 = self.clif1(x)
                if i == 0: self.clif2.reset(x1)
                x2 = self.clif2(x1)
                if i == 0: self.clif3.reset(x2)
                x3 = self.clif3(x2)
                x4 = self.flat(x3)
                if i == 0: self.flif1.reset(x4)
                x5 = self.flif1(x4)
                if i == 0: self.flif2.reset(x5)
                x6 = self.flif2(x5)
                xo = xo + x6 / self.timestep
            return xo

class MINIST_FC(nn.Module):

    def __init__(self,
                 timestep=10, c0=1, fclass=10, cmode='spike', fmode='spike', amode='mean',
                 noise=1e-3, neurons_num=7, encoder=None
                 ):
        super(MINIST_FC, self).__init__()
        self.neurons_num = neurons_num
        self.fclass = fclass
        self.norm = nn.BatchNorm2d(c0)
        self.flat = nn.Flatten(1, -1)
        self.flif1 = FcLif(28 * 28, 256, mode=fmode, noise=noise)
        self.flif2 = FcLif(256, 10, mode=fmode, noise=noise)
        assert amode == 'mean'
        self.timestep = timestep
        self.ON_APU = globals.get_value('ON_APU')
        self.FIT = globals.get_value('FIT')
        if self.ON_APU is not True:
            if encoder == 'latency':
                self.encoder = encoding.LatencyEncoder(T=10, enc_function='linear')
            elif encoder == 'phase':
                self.encoder = encoding.WeightedPhaseEncoder(K=10)
            else:
                self.encoder = None

    def reset(self, xi):
        self.tempAdd = torch.zeros_like(xi)

    def forward(self, xis: torch.Tensor) -> torch.Tensor:
        if self.ON_APU:
            x = self.norm(xis)
            x = self.flat(x)
            self.flif1.reset(x)
            x = self.flif1(x)
            self.flif2.reset(x)
            x = self.flif2(x)
            x = x.unsqueeze(2).unsqueeze(3)
            self.reset(x)
            self.tempAdd = load(self.tempAdd, f'tempAdd')
            self.tempAdd = self.tempAdd + x / self.timestep
            output = self.tempAdd.clone()
            save(self.tempAdd, f'tempAdd')
            return output.squeeze(-1).squeeze(-1)

        else:
            xo = 0
            for i in range(self.timestep):
                if self.encoder is not None:
                    if i == 0: self.encoder.reset()
                    x = self.encoder(xis)  # (B, D)
                else:
                    x = xis[:, i, ...]
                x = self.norm(x)
                x = self.flat(x)
                if i == 0: self.flif1.reset(x)
                x = self.flif1(x)
                if i == 0: self.flif2.reset(x)
                x = self.flif2(x)
                xo = xo + x / self.timestep
            return xo


class MINIST_ANN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def model_choice(model_type, device, is_FC=False):
    if model_type == 'WSHE':
        print(f'Training SNN use WSHE! ')
        print(is_FC)
        if is_FC:
            model = MNIST_FC_WSHE().to(device)
        else:
            model = MNIST_CSNN_WSHE().to(device)
        encoder = SpikeEncoder_GPU.TorchSpikeEncoder().to(device)
        return model, encoder
    elif model_type == 'poisson':
        print(f'Training SNN use Poisson Encoder! ')
        if is_FC:
            model = MINIST_FC(encoder=None).to(device)
        else:
            model = MINIST_CSNN(encoder=None).to(device)
        encoder = encoding.PoissonEncoder()
        return model, encoder
    elif model_type == 'phase':
        print(f'Training SNN use Phase Encoder! ')
        if is_FC:
            model = MINIST_FC(encoder='phase').to(device)
        else:
            model = MINIST_CSNN(encoder='phase').to(device)
        encoder = None
        return model, encoder
    elif model_type == 'latency':
        print(f'Training SNN use latency Encoder! ')
        if is_FC:
            model = MINIST_FC(encoder='latency').to(device)
        else:
            model = MINIST_CSNN(encoder='latency').to(device)
        encoder = None
        return model, encoder
    elif model_type == 'ann':
        print(f'Training ANN model!')
        model = MINIST_ANN.to(device)
        encoder = None
        return model, encoder
    else:
        raise ValueError(
            f"Unsupported model: {model_type}. The supported models are: 'WSHE', 'ann', 'poisson', 'phase', 'latency'。")