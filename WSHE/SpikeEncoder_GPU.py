# Digital Signature: bb140e06
"""
WSHE Encoder Implementation
Copyright (c) 2025 Xin Zhao, Beijing Jiaotong University
"""
import torch
import matplotlib.pyplot as plt
import math



class TorchSpikeEncoder(torch.nn.Module):
    """
    加权时空混合编码,用于编码特征图像信息。

    Weighted Spatiotemporal Hybrid Encoding, which is used to encode feature image information.

    """

    def __init__(self, num_trains=9, num_steps=10, point_pos=1,
                 sign_channel=True, pos_channel=False):
        """
        初始化方法。
        参数：
            num_trains(int)： 脉冲序列数，默认取N=9即可应对大多数情况。
            num_steps(int): 时间步数，默认取T=10即可应对大多数情况。
            point_pos(int): 权重位，程序会自动判断权重大小，默认值为1。
            sign_channel(bool): 是否启用符号通道。
            pos_channel(bool): 是否启用权重通道。

        Initialization method.
        Param：
            num_trains(int)： The number of pulse sequences, the default is N=9 to cope with most cases.
            num_steps(int): The number of time steps, the default is T=10 to deal with most situations.
            point_pos(int): The weight bit, the program will automatically judge the weight of the weight, and the default value is 1.
            sign_channel(bool): Whether the symbolic channel is enabled.
            pos_channel(bool): Whether to enable the weight channel.

        """
        super().__init__()
        # Parameter validation
        assert num_trains >= 3, "num_trains requires at least 3 channels (symbol + weight + at least 1 digit)"

        self.num_trains = num_trains
        self.num_steps = num_steps
        self.point_pos = point_pos
        self.sign_channel = sign_channel
        self.pos_channel = pos_channel

        # Calculate the output channel
        self.ochannel = 0
        if self.sign_channel: self.ochannel += 1
        if self.pos_channel: self.ochannel += 1

        self.num_digits = self.num_trains - self.ochannel
        # Precomputed constants
        self.register_buffer('time_grid', torch.arange(num_steps))
        self.register_buffer('digit_powers', torch.tensor(
            [10 ** i for i in range(num_trains - self.ochannel - 1, -1, -1)],
            dtype=torch.int64
        ))
        self.register_buffer('exponents', 10 ** torch.arange(self.num_digits - 1, -1, -1))

    @torch.no_grad()
    def forward(self, x):
        """
        主要编码方法。

        参数：
            x(tensor)：输入的图张量，形状(B, C, H, W)。
        返回：
            encoded_graph(tensor)：输出的图张量，形状(B, num_steps, C, H*3, W*3)。

        Primary coding methods.

        Param：
            x(tensor): the input graph tensor, shape(B, C, H, W).
        return:
            encoded_graph (tensor): the output graph tensor, shape(B, num_steps, C, H*3, W*3).

        """
        B, C, H, W = x.shape
        N = B * C * H * W  # Total number of elements
        device = x.device
        x_flat = x.flatten()
        abs_values = torch.abs(x_flat)
        num_digits = self.num_trains - (1 if self.sign_channel else 0)  # Adjust according to other channels

        # Sign processing ----------------------------------------------------------
        if self.sign_channel:
            # Determining Positive or Negative
            positive_mask = x >= 0

            # Generate Symbolic Pulse Train (All 1 for Positive, 0 for Negative)
            symbol_sequences = torch.full(
                size=(positive_mask.numel(), self.num_steps),
                fill_value=0,
                dtype=torch.float32,
                device=x.device
            )
            # Positive positions are padded with 1
            symbol_sequences[positive_mask.flatten()] = 1.0

        # Weights processing ----------------------------------------------------------
        if self.pos_channel:
            # Calculate the weights
            mask = abs_values >= 1.0
            log_values = torch.where(
                mask,
                torch.log10(abs_values),
                torch.tensor(1e-10, device=x.device)
            )
            point_pos = torch.floor(log_values).int() + 1
            point_pos[~mask] = 1
            point_pos = torch.clamp(point_pos, 1, num_digits)

            # Generate pulse trains
            pos_mask = self.time_grid < point_pos.view(-1, 1)
            pos_sequences = pos_mask.float()  # (N, T)
        else:
            point_pos = torch.full((N,), self.point_pos, device=device)

        # Numeric sequence encoding --------------------------------------------------
        num_digits = self.num_trains - self.ochannel

        scales = 10 ** (num_digits - point_pos)
        values_scaled = (abs_values * scales).round().long()

        # Decompose the digits
        # digits = (values_scaled.view(-1, 1) // self.exponents) % 10  # (N, D)

        div_result = torch.div(values_scaled.view(-1, 1),
                               self.exponents,
                               rounding_mode='trunc')
        digits = (div_result) % 10

        # Generate pulse trains
        # time_grid = torch.arange(self.num_steps, device=x.device)
        digit_sequences = (self.time_grid < digits.unsqueeze(2)).float()  # (N, D, T)


        all_sequences = torch.zeros(N, self.num_steps, self.num_trains, device=device) # Initialize the pulse train


        # Merge pulse trains -------------------------------------------------------
        if self.sign_channel:
            all_sequences[:, :, 0] = symbol_sequences
        if self.pos_channel:
            all_sequences[:, :, 1 if self.sign_channel else 0] = pos_sequences
        all_sequences[:, :, self.ochannel:] = digit_sequences.transpose(2, 1)

        # Reshape and expand spatial dimensions ---------------------------------------------
        encoded_sequences = all_sequences.view(B, C, H, W, self.num_steps, 3, 3)
        # Adjust the dimension order: (B, C, T, H, 3, W, 3)
        encoded_graph = encoded_sequences.permute(0, 1, 4, 2, 5, 3, 6)
        encoded_graph = encoded_graph.reshape(B, C, self.num_steps, H * 3, W * 3).transpose(1, 2)
        return encoded_graph


    def ValueDecoder(self, pulse_sequence):
        """
        脉冲解码方法，解码单个数值的脉冲序列，用于验证编码准确性而非网络输出。

        参数:
        pulse_sequences (list): 一个脉冲序列列表。

        返回:
        value (float): 解码出的数字。

        Pulse decoding method, which decodes a pulse train of a single value, is used to verify the encoding accuracy rather than the network output.

        Param:
            pulse_sequences (list): A list of spike trains.
        return:
            value (float): The number decoded.
        """
        if len(pulse_sequence.shape)==3:
            pulse_sequence = pulse_sequence.reshape((self.num_steps, 9)).T
        # Decode the symbol bits
        if self.sign_channel:
            symbol_sequence = pulse_sequence[0]
            num_positive_spikes = torch.sum(symbol_sequence)  # 统计脉冲数量
            sign = 1 if num_positive_spikes >= self.num_steps / 2 else -1  # 判断符号

        # Decode weights
        point_position = 1
        if self.pos_channel:
            point_position = torch.sum(pulse_sequence[1])

        # Decode digital sequences
        decoded_digits = []
        for integer_sequence in pulse_sequence[self.ochannel: self.num_trains]:
            spike_index = torch.sum(integer_sequence)  # 找到脉冲发射的位置
            # integer_value = round((1 - spike_index / (self.num_steps - 1)) * 9)  # 映射到 [0, 9]
            integer_value = spike_index  # 映射到 [0, 9]
            decoded_digits.append(integer_value)
        # print(decoded_digits)
        # Combine the decoded quantile values into a float value
        value = sum([digit * (10 ** (-i + point_position - 1)) for i, digit in enumerate(decoded_digits)])

        # Restore symbols based on symbol bits
        if self.sign_channel:
            value = value * sign

        return value

    def PlotPulseSequence(self, pulse_sequence):
        """
        脉冲序列可视化。

        参数:
        pulse_sequence (list): 一个脉冲序列。

        Pulse train visualization.

        Param：
            pulse_sequence (list): A list of spike trains.

        """
        if len(pulse_sequence.shape)==3:
            pulse_sequence = pulse_sequence.reshape((self.num_steps, 9)).T

        # Decode the sequence
        value = self.ValueDecoder(pulse_sequence)

        # Visualize the pulse train
        plt.figure(figsize=(10, 6))

        # Plot each pulse train
        for i, pulse_sequence in enumerate(pulse_sequence):
            plt.step(torch.arange(self.num_steps), pulse_sequence + i, where='post', label=f'Sequence {i + 1}')

        plt.title(f'Encoding for Value {value}')
        plt.xlabel('Time Step')
        plt.ylabel('Pulse Sequence')
        plt.yticks(torch.arange(0, len(pulse_sequence) + 1))
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.show()


class TorchRGBEncoder(torch.nn.Module):
    """
    加权时空混合编码，专为RGB整数数据设计。

    Weighted spatiotemporal hybrid encoding, designed for RGB integer data.
    """

    def __init__(self, num_steps=10):
        """
        初始化方法。
        参数：
            num_steps(int): 时间步数，默认取T=10即可应对大多数情况。

        Initialization method.
        Param：
            num_steps(int): The number of time steps, the default is T=10 to deal with most situations.
        """

        super().__init__()
        self.num_steps = num_steps
        self.num_trains = 4  # Fixed use of 2x2 blocks

        # Precomputed Number Factorization Base [100, 10, 1]
        self.register_buffer('exponents', torch.tensor([100, 10, 1], dtype=torch.int64))
        self.register_buffer('time_grid', torch.arange(num_steps))

    @torch.no_grad()
    def forward(self, x):
        """
        主要编码方法。

        参数：
            x(tensor)：输入的图张量，形状(B, C, H, W) 值域[0,255]
        返回：
            encoded_graph(tensor)：输出的图张量，形状(B, num_steps, C, H*2, W*2)

        Primary coding methods.

        Param：
            x(tensor): the input graph tensor, shape(B, C, H, W) range[0,255].
        return:
            encoded_graph (tensor): the output graph tensor, shape: (B, num_steps, C, H*2, W*2).

        """
        B, C, H, W = x.shape
        N = B * C * H * W
        device = x.device
        x = torch.clamp(x, 0, 255).long()  # Force integer ranges
        x = x.flatten()

        # Decompose digits (hundreds, tens, units)
        digits = (x.view(-1, 1) // self.exponents) % 10

        # Expands to 2x2 blocks (4th train fills)
        padding = torch.zeros_like(digits[..., :1])  # 空通道
        digits_padded = torch.cat([digits, padding], dim=-1)

        # Generate pulse trains
        spike_grid = (self.time_grid < digits_padded.unsqueeze(2)).float()
        all_sequences = torch.zeros(N, self.num_steps, self.num_trains, device=device)

        all_sequences[:, :, :] = spike_grid.transpose(2, 1)

        # Reshape and expand spatial dimensions ---------------------------------------------
        encoded_sequences = all_sequences.view(B, C, H, W, self.num_steps, 2, 2)

        encoded_graph = encoded_sequences.permute(0, 1, 4, 2, 5, 3, 6)
        encoded_graph = encoded_graph.reshape(B, C, self.num_steps, H * 2, W * 2)
        return encoded_graph.transpose(1, 2)

    def ValueDecoder(self, pulse_sequence):
        """
        脉冲解码方法，解码单个数值的脉冲序列，用于验证编码准确性而非网络输出。

        参数:
        pulse_sequences (list): 一个脉冲序列列表。

        返回:
        value (float): 解码出的数字。

        Pulse decoding method, which decodes a pulse train of a single value, is used to verify the encoding accuracy rather than the network output.

        Param:
            pulse_sequences (list): A list of spike trains.
        return:
            value (float): The number decoded.
        """
        if len(pulse_sequence.shape)==3:
            pulse_sequence = pulse_sequence.reshape((self.num_steps, 4)).T
        decoded_digits = []
        for integer_sequence in pulse_sequence[: 3]:
            spike_index = torch.sum(integer_sequence)
            integer_value = spike_index
            decoded_digits.append(integer_value)
        # print(decoded_digits)
        # Combine the decoded quantile values into a floating value
        value = sum([digit * (10 ** (-i + 2)) for i, digit in enumerate(decoded_digits)])
        return value


class TorchTensorEncoder(torch.nn.Module):
    """
    加权时空混合编码，用于编码二维张量。

    Weighted spatiotemporal hybrid encoding for encoding two-dimensional tensors.
    """
    def __init__(self, num_trains=9, num_steps=10, point_pos=1,
                 sign_channel=True, pos_channel=True):
        """
        初始化方法。
        参数：
            num_trains(int)： 脉冲序列数，默认取N=9即可应对大多数情况。
            num_steps(int): 时间步数，默认取T=10即可应对大多数情况。
            point_pos(int): 权重位，程序会自动判断权重大小，默认值为1。
            sign_channel(bool): 是否启用符号通道。
            pos_channel(bool): 是否启用权重通道。

        Initialization method.
        Param：
            num_trains(int)： The number of pulse sequences, the default is N=9 to cope with most cases.
            num_steps(int): The number of time steps, the default is T=10 to deal with most situations.
            point_pos(int): The weight bit, the program will automatically judge the weight of the weight, and the default value is 1.
            sign_channel(bool): Whether the symbolic channel is enabled.
            pos_channel(bool): Whether to enable the weight channel.

                """
        super().__init__()
        self.num_trains = num_trains
        self.num_steps = num_steps
        self.point_pos = point_pos
        self.sign_channel = sign_channel
        self.pos_channel = pos_channel

        # 预计算常量
        self.ochannel = 0
        if self.sign_channel: self.ochannel += 1
        if self.pos_channel: self.ochannel += 1

        self.register_buffer('exponents', 10 ** torch.arange(
            num_trains - self.ochannel - 1, -1, -1, dtype=torch.int64
        ))
        self.register_buffer('time_grid', torch.arange(num_steps))

    @torch.no_grad()
    def forward(self, x):
        """
        主要编码方法。

        参数：
            x(tensor)：输入二维张量，形状(B, N)。
        返回：
            encoded_graph(tensor)：输出的脉冲张量，形状(B, num_steps, N, num_trains)

        Primary coding methods.

        Param：
            x(tensor): the input tensor, shape(B, N).
        return:
            encoded_graph (tensor): the output tensor, shape(B, num_steps, N, num_trains)

        """
        B, N = x.shape
        device = x.device

        # Sign processing ------------------------------------------------------
        sign_mask = torch.ones_like(x, dtype=torch.bool)
        if self.sign_channel:
            sign_mask = x >= 0  # (B, N)
        abs_vals = torch.abs(x)
        # else:
        #     abs_vals = x

        # Weight processing ------------------------------------------------------
        point_pos = torch.full((B, N), self.point_pos, device=device)
        if self.pos_channel:
            for i in range(self.num_trains):
                lower = 10.0 ** i
                upper = 10.0 ** (i + 1)
                mask = (abs_vals > lower) & (abs_vals <= upper)
                point_pos[mask] = i + 1

            # Values less than 1 are processed
            point_pos[abs_vals < 1.0] = 1
            point_pos.clamp_(1, self.num_trains - self.ochannel)

        # Numeric encoding core logic ------------------------------------------------
        num_digits = self.num_trains - self.ochannel
        scale = 10 ** (num_digits - point_pos)  # (B, N)
        scaled_vals = (abs_vals * scale).round().long()  # (B, N)

        # Decompose the digits (B, N, D)
        digits = (scaled_vals.unsqueeze(2) // self.exponents) % 10

        # Generate pulse trains (B, N, D, T)
        time_grid = self.time_grid.view(1, 1, 1, -1)  # (1, 1, 1, T)
        digit_sequences = (time_grid < digits.unsqueeze(3)).float()  # 广播比较

        # Merge all channels --------------------------------------------------
        # Initialize the output tensor (B, N, T, C)
        output = torch.zeros(B, N, self.num_steps, self.num_trains, device=device)

        # Populate the sequence of symbols
        if self.sign_channel:
            output[..., 0] = sign_mask.unsqueeze(2).expand(-1, -1, self.num_steps)

        # Populate the weight sequence
        if self.pos_channel:
            pos_channel_idx = 1 if self.sign_channel else 0
            pos_seq = (self.time_grid < point_pos.unsqueeze(2)).float()  # (B, N, T)
            output[..., pos_channel_idx] = pos_seq

        # Populate the sequence of numbers
        output[..., self.ochannel:] = digit_sequences.permute(0, 1, 3, 2)  # (B, N, T, D)

        # Adjust dimension order to (B, T, N, C)
        return output.permute(0, 2, 1, 3)

    def ValueDecoder(self, pulse_sequence):
        """
        脉冲解码方法，解码单个数值的脉冲序列，用于验证编码准确性而非网络输出。

        参数:
        pulse_sequences (list): 一个脉冲序列列表。

        返回:
        value (float): 解码出的数字。

        Pulse decoding method, which decodes a pulse train of a single value, is used to verify the encoding accuracy rather than the network output.

        Param:
            pulse_sequences (list): A list of spike trains.
        return:
            value (float): The number decoded.
        """
        if len(pulse_sequence.shape)==3:
            pulse_sequence = pulse_sequence.reshape((self.num_steps, 9)).T
        # Decode the sequence of symbols
        if self.sign_channel:
            symbol_sequence = pulse_sequence[0]
            num_positive_spikes = torch.sum(symbol_sequence)
            sign = 1 if num_positive_spikes >= self.num_steps / 2 else -1
            index = 1
        else:
            index = 0

        # Decode the weight sequence

        point_position = 1
        if self.pos_channel:
            point_position = torch.sum(pulse_sequence[index])

        # Decode digital sequences
        decoded_digits = []
        for integer_sequence in pulse_sequence[self.ochannel: self.num_trains]:
            spike_index = torch.sum(integer_sequence)
            # integer_value = round((1 - spike_index / (self.num_steps - 1)) * 9)
            integer_value = spike_index
            decoded_digits.append(integer_value)
        # print(decoded_digits)
        # Combine the decoded quantile values into a floating value
        value = sum([digit * (10 ** (-i + point_position - 1)) for i, digit in enumerate(decoded_digits)])

        # Restore symbols based on symbol bits
        if self.sign_channel:
            value = value * sign

        return value

if __name__ == '__main__':
    ################################Graph encoding verification#######################################
    # Initialize the encoder
    encoder = TorchSpikeEncoder(num_trains=9, num_steps=10, pos_channel=True).to('cuda')

    # Sample input (batch=4, channel=3, 224x224)
    input_tensor = torch.randn(4, 3, 224, 224).to('cuda') * 100000
    # Verify encoding speed
    import time

    spike_seq = encoder(input_tensor).cpu()  # Output shape: (4, 10, 3, 672, 672)
    value = input_tensor[1, 1, 17, 14]
    print(f'origin value: {value}')
    # graph_encoded = encoder.GraphEncoder(image_tensor)
    start_time = time.time()
    print(f'encode time: {time.time() - start_time}')
    encoded_train = spike_seq[1, :, 1, 17 * 3: 17 * 3 + 3, 14 * 3: 14 * 3 + 3]
    # encoded_train = spike_seq[0, :, 0, 0:3, 0:3]
    print(f'graph encoded shape: {spike_seq.shape}')

    print(f'encoded value shape: {encoded_train.shape}\nencoded value: \n{encoded_train}')
    decoded_value = encoder.ValueDecoder(encoded_train)
    print(f'decoded value: {decoded_value}')
    encoder.PlotPulseSequence(encoded_train)


    ##############################2D tensor encoding verification######################################
    # encoder = TorchTensorEncoder(sign_channel=True, pos_channel=True).to('cuda')
    #
    # input_tensor = torch.randn(64, 200).to('cuda') * 1000
    #
    # B = 0
    # N = 0
    # # input_tensor[B][N] = -1.694097638130188
    # pulse_sequences = encoder(input_tensor)
    # print(f"pulse_sequences shape: {pulse_sequences.shape}")
    # value = input_tensor[B][N]
    # print(f'original value: {value}')
    # encoded_spike_train = pulse_sequences[B, :, N, :].T
    # print(f'encoded_spike_train shape: {encoded_spike_train.shape}\nencoded_spike_train: {encoded_spike_train}')
    # decoded_value = encoder.ValueDecoder(encoded_spike_train)
    # print(f'decoded value: {decoded_value}')

    #############################RGB image encoding verification##############################################
    # random_tensor = torch.randint(low=0, high=256, size=(128, 3, 32, 32), dtype=torch.uint8).to('cuda')
    # # print(random_tensor.shape)
    # B, C, H, W = 0, 0, 0, 0
    #
    # encoder = TorchRGBEncoder().to('cuda')
    #
    # encoded_seq = encoder(random_tensor)
    # print(f'encoded shape: {encoded_seq.shape}')
    # print(f'original value:{random_tensor[B, C, H, W]}')
    # value_seq = encoded_seq[B, :, C, H*2: H*2 + 2, W*2: W*2 + 2]
    # print(f'encoded value:{value_seq.shape}\n encoded seq:\n{value_seq}')
    # decoded = encoder.ValueDecoder(value_seq)
    # print(f'decoded value:{decoded}')








