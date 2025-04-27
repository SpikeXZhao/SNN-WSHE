import torch
import torch.nn as nn
from utils import globals
from lynadapter.warp_load_save import load,save,load_kernel,save_kernel
from mmcls.models.layers.lif import FcLif, Conv2dLif
from spikingjelly.activation_based import encoding
import SpikeEncoder_GPU


class CIFAR10CSNN_WSHE(nn.Module):
    def __init__(self, timestep=10, fmode='spike', cmode='spike', amode='mean', soma_params='all_share', noise=1e-3,
                 neuron='lif'):
        super(CIFAR10CSNN_WSHE, self).__init__()
        neuron = neuron.lower()
        assert neuron in ['lif']
        self.ON_APU = globals.get_value('ON_APU')
        self.clif1 = Conv2dLif(3, 64, kernel_size=3, stride=3, mode=cmode, soma_params=soma_params, noise=noise)
        self.clif2 = Conv2dLif(64, 64, kernel_size=3, padding=1, mode=cmode, soma_params=soma_params, noise=noise)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.clif3 = Conv2dLif(64, 128, kernel_size=3, padding=1, mode=cmode, soma_params=soma_params, noise=noise)
        self.clif4 = Conv2dLif(128, 128, kernel_size=3, padding=1, mode=cmode, soma_params=soma_params, noise=noise)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.clif5 = Conv2dLif(128, 256, kernel_size=3, padding=1, mode=cmode, soma_params=soma_params, noise=noise)
        self.clif6 = Conv2dLif(256, 256, kernel_size=3, padding=1, mode=cmode, soma_params=soma_params, noise=noise)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fclif1 = FcLif(256 * 4 * 4, 512, mode=fmode, soma_params=soma_params, noise=noise)
        self.dropout = nn.Dropout(0.3)

        self.fclif2 = FcLif(512, 10, mode=fmode, soma_params=soma_params, noise=noise)

        assert amode == 'mean'
        self.tempAdd = None
        self.timestep = timestep
        self.FIT = globals.get_value('FIT')
        self.MULTINET = globals.get_value('MULTINET')
        self.MODE = globals.get_value('MODE')
        print(f'apu: {self.ON_APU}, FIT : {self.FIT}, MULTINET : {self.MULTINET}, MODE : {self.MODE}')

    def reset(self, xi):
        self.tempAdd = torch.zeros_like(xi)

    def forward(self, xis: torch.Tensor) -> torch.Tensor:
        if self.ON_APU:
            assert len(xis.shape) == 4 # (B, C, H, W)
            x = xis
            self.clif1.reset(x)
            x = self.clif1(x)
            self.clif2.reset(x)
            x = self.clif2(x)
            x = self.pool1(x)
            self.clif3.reset(x)
            x = self.clif3(x)
            self.clif4.reset(x)
            x = self.clif4(x)
            x = self.pool2(x)
            self.clif5.reset(x)
            x = self.clif5(x)
            self.clif6.reset(x)
            x = self.clif6(x)
            x = self.pool3(x)
            x = torch.flatten(x, start_dim=1)
            self.fclif1.reset(x)
            x = self.fclif1(x)
            x = self.dropout(x)
            self.fclif2.reset(x)
            x = self.fclif2(x)
            self.reset(x)

            if self.MULTINET:
                self.tempAdd = load_kernel(self.tempAdd, f'tempAdd', uselookup=True, mode=self.MODE)
            else:
                self.tempAdd = load_kernel(self.tempAdd, f'tempAdd', mode=self.MODE)

            self.tempAdd = self.tempAdd + x / self.timestep
            # self.tempAdd = x
            output = self.tempAdd.clone()
            if self.MULTINET:
                save_kernel(self.tempAdd, f'tempAdd', uselookup=True, mode=self.MODE)
            else:
                save_kernel(self.tempAdd, f'tempAdd', mode=self.MODE)
            return output.squeeze(-1).squeeze(-1)

        else:
            t = xis.size(1)
            xo_list = []
            xo = 0
            for i in range(t):
                x = xis[:, i, ...]  # (B, D)
                if i == 0: self.clif1.reset(x)
                x = self.clif1(x)
                # x = self.dropout2d1(x)
                if i == 0: self.clif2.reset(x)
                x = self.clif2(x)
                x = self.pool1(x)
                if i == 0: self.clif3.reset(x)
                x = self.clif3(x)
                if i == 0: self.clif4.reset(x)
                x = self.clif4(x)
                x = self.pool2(x)
                # x = self.dropout2d2(x)
                if i == 0: self.clif5.reset(x)
                x = self.clif5(x)
                if i == 0: self.clif6.reset(x)
                x = self.clif6(x)
                x = self.pool3(x)
                x = torch.flatten(x, start_dim=1)
                if i == 0: self.fclif1.reset(x)
                x = self.fclif1(x)
                x = self.dropout(x)
                if i == 0: self.fclif2.reset(x)
                x = self.fclif2(x)
                xo = xo + x / t
                # xo = x
            return xo

class CIFAR10CSNN(nn.Module):
    def __init__(self, timestep=10, fmode='spike', cmode='spike', amode='mean', soma_params='all_share', noise=1e-3,
                 neuron='lif', encoder=None, isRGB=False):
        super(CIFAR10CSNN, self).__init__()
        neuron = neuron.lower()
        assert neuron in ['lif']
        self.ON_APU = globals.get_value('ON_APU')
        if isRGB:
            self.clif1 = Conv2dLif(3, 64, kernel_size=2, stride=2, mode=cmode, soma_params=soma_params, noise=noise)
        else:
            self.clif1 = Conv2dLif(3, 64, kernel_size=3, padding=1, mode=cmode, soma_params=soma_params, noise=noise)
        self.clif2 = Conv2dLif(64, 64, kernel_size=3, padding=1, mode=cmode, soma_params=soma_params, noise=noise)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.clif3 = Conv2dLif(64, 128, kernel_size=3, padding=1, mode=cmode, soma_params=soma_params, noise=noise)
        self.clif4 = Conv2dLif(128, 128, kernel_size=3, padding=1, mode=cmode, soma_params=soma_params, noise=noise)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.clif5 = Conv2dLif(128, 256, kernel_size=3, padding=1, mode=cmode, soma_params=soma_params, noise=noise)
        self.clif6 = Conv2dLif(256, 256, kernel_size=3, padding=1, mode=cmode, soma_params=soma_params, noise=noise)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fclif1 = FcLif(256 * 4 * 4, 512, mode=fmode, soma_params=soma_params, noise=noise)
        self.dropout = nn.Dropout(0.3)

        self.fclif2 = FcLif(512, 10, mode=fmode, soma_params=soma_params, noise=noise)

        assert amode == 'mean'
        self.tempAdd = None
        self.timestep = timestep
        self.FIT = globals.get_value('FIT')
        self.MULTINET = globals.get_value('MULTINET')
        self.MODE = globals.get_value('MODE')
        print(f'apu: {self.ON_APU}, FIT : {self.FIT}, MULTINET : {self.MULTINET}, MODE : {self.MODE}')

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
            assert len(xis.shape) == 4  # (B, C, H, W)
            x = xis
            self.clif1.reset(x)
            x = self.clif1(x)
            self.clif2.reset(x)
            x = self.clif2(x)
            x = self.pool1(x)
            self.clif3.reset(x)
            x = self.clif3(x)
            self.clif4.reset(x)
            x = self.clif4(x)
            x = self.pool2(x)
            self.clif5.reset(x)
            x = self.clif5(x)
            self.clif6.reset(x)
            x = self.clif6(x)
            x = self.pool3(x)
            x = torch.flatten(x, start_dim=1)
            self.fclif1.reset(x)
            x = self.fclif1(x)
            x = self.dropout(x)
            self.fclif2.reset(x)
            x = self.fclif2(x)
            self.reset(x)

            if self.MULTINET:
                self.tempAdd = load_kernel(self.tempAdd, f'tempAdd', uselookup=True, mode=self.MODE)
            else:
                self.tempAdd = load_kernel(self.tempAdd, f'tempAdd', mode=self.MODE)

            self.tempAdd = self.tempAdd + x / self.timestep
            output = self.tempAdd.clone()
            if self.MULTINET:
                save_kernel(self.tempAdd, f'tempAdd', uselookup=True, mode=self.MODE)
            else:
                save_kernel(self.tempAdd, f'tempAdd', mode=self.MODE)
            return output.squeeze(-1).squeeze(-1)

        else:
            xo = 0
            for i in range(self.timestep):
                if self.encoder is not None:
                    if i == 0: self.encoder.reset()
                    x = self.encoder(xis)  # (B, D)
                else:
                    x = xis[:, i, ...]
                if i == 0: self.clif1.reset(x)
                x = self.clif1(x)
                if i == 0: self.clif2.reset(x)
                x = self.clif2(x)
                x = self.pool1(x)
                if i == 0: self.clif3.reset(x)
                x = self.clif3(x)
                if i == 0: self.clif4.reset(x)
                x = self.clif4(x)
                x = self.pool2(x)
                if i == 0: self.clif5.reset(x)
                x = self.clif5(x)
                if i == 0: self.clif6.reset(x)
                x = self.clif6(x)
                x = self.pool3(x)
                x = torch.flatten(x, start_dim=1)
                if i == 0: self.fclif1.reset(x)
                x = self.fclif1(x)
                x = self.dropout(x)
                if i == 0: self.fclif2.reset(x)
                x = self.fclif2(x)
                xo = xo + x / self.timestep
            return xo

class CIFAR10ANNModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def model_choice(model_type, device):
    if model_type == 'WSHE':
        print(f'Training SNN use Hybrid Encoder! ')
        model = CIFAR10CSNN_WSHE().to(device)
        encoder = SpikeEncoder_GPU.TorchSpikeEncoder().to(device)
        return model, encoder
    elif model_type == 'poisson':
        print(f'Training SNN use Poisson Encoder! ')
        model = CIFAR10CSNN(encoder=None).to(device)
        encoder = encoding.PoissonEncoder()
        return model, encoder
    elif model_type == 'phase':
        print(f'Training SNN use Phase Encoder! ')
        model = CIFAR10CSNN(encoder='phase').to(device)
        encoder = None
        return model, encoder
    elif model_type == 'latency':
        print(f'Training SNN use Latency Encoder! ')
        model = CIFAR10CSNN(encoder='latency').to(device)
        encoder = None
        return model, encoder
    elif model_type == 'ann':
        print(f'Training ANN model!')
        model = CIFAR10ANNModel().to(device)
        encoder = None
        return model, encoder
    elif model_type == 'RGB':
        print(f'Training SNN use RGB Encoder! ')
        model = CIFAR10CSNN(encoder=None, isRGB=True).to(device)
        encoder = SpikeEncoder_GPU.TorchRGBEncoder().to(device)
        return model, encoder
    else:
        raise ValueError(
            f"Unsupported model: {model_type}. The supported models are: 'WSHE', 'ann', 'poisson', 'phase', 'latency', 'RGB'。")