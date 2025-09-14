import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch, time, pdb, os, random, math
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from utils import load_CIFAR10, get_RRC, args_parser, progress_bar
from resnet import ResNetTx, ResNetRx
from scipy.io import savemat
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set seeds
torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
np.random.seed(0)

class AFDMParameterNet(nn.Module):
    """神经网络用于生成AFDM参数c1和c2"""
    def __init__(self, input_size=64*64*2, hidden_size=256, output_size=2):
        super(AFDMParameterNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, H):
        # 将复数信道矩阵转换为实数表示
        H_real = H.real.reshape(H.shape[0], -1)
        H_imag = H.imag.reshape(H.shape[0], -1)
        x = torch.cat([H_real, H_imag], dim=1)
        
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        # 使用sigmoid将输出限制在[0,1]范围内
        c_params = self.sigmoid(x)
        return c_params[:, 0], c_params[:, 1]  # c1, c2

class SemanticComm(nn.Module):
    def __init__(self, args):
        super(SemanticComm, self).__init__()
        self.args = args
        self.N = args.N
        self.Enc = ResNetTx()
        self.Dec = ResNetRx()
        
        # AFDM参数生成网络
        self.afdm_param_net = AFDMParameterNet(input_size=self.N*self.N*2, hidden_size=256, output_size=2)
        
        # 固定信道参数 - 将在每5个epoch开始时随机生成
        self.register_buffer('h', torch.zeros(3, dtype=torch.complex64))
        self.register_buffer('normDelays', torch.zeros(3, dtype=torch.float32))
        self.register_buffer('normDopplers', torch.zeros(3, dtype=torch.float32))
        self.P = 3
        
        # 信道矩阵缓存
        self.register_buffer('channel_matrix_cache', None)
        self.register_buffer('cached_c1', None)
        
        # 记录当前使用的信道参数组
        self.current_channel_group = -1
        
        # 固定AFDM参数（第一阶段使用）
        self.fixed_c1 = torch.tensor(0.1, dtype=torch.float32)
        self.fixed_c2 = torch.tensor(0.0, dtype=torch.float32)

    def set_channel_parameters(self, epoch):
        """为当前epoch设置随机信道参数，每5个epoch变化一次"""
        # 计算信道参数组号
        channel_group = epoch // 5
        
        # 如果信道参数组号没有变化，则不需要更新
        if channel_group == self.current_channel_group:
            return
            
        # 更新当前信道参数组号
        self.current_channel_group = channel_group
        
        # 设置随机种子以确保每个信道参数组的信道参数相同
        torch.manual_seed(channel_group)
        
        # 随机生成信道参数
        # 信道系数 (复数) 
        h_real = torch.normal(0, 1, (self.P,))  
        h_imag = torch.normal(0, 1, (self.P,))
        h = torch.complex(h_real, h_imag)
        # 归一化信道系数
        h = h / torch.sqrt(torch.sum(torch.abs(h)**2))
        
        # 随机延迟 
        max_delay = 5
        normDelays = torch.randint(0, max_delay+1, (self.P,)).float()
        
        # 随机多普勒 (在[0, 1]范围内)
        normDopplers = torch.rand(self.P)
        
        # 更新缓冲区
        self.h.data.copy_(h)
        self.normDelays.data.copy_(normDelays)
        self.normDopplers.data.copy_(normDopplers)
        
        # 清除缓存
        self.channel_matrix_cache = None
        self.cached_c1 = None
        
        print(f"更新信道参数 (组 {channel_group}):")
        print(f"  信道系数: {h}")
        print(f"  延迟: {normDelays}")
        print(f"  多普勒: {normDopplers}")

    def power_norm(self, feature):
        in_shape = feature.shape
        sig_in = feature.reshape(in_shape[0], -1)
        # each pkt in a batch, compute the mean and var
        sig_mean = torch.mean(sig_in, dim=1)
        sig_std = torch.std(sig_in, dim=1)
        # normalize
        sig_out = (sig_in - sig_mean.unsqueeze(dim=1)) / (sig_std.unsqueeze(dim=1) + 1e-8)
        return sig_out.reshape(in_shape)

    def generate_channel_matrix(self, N, c1):
        # 检查缓存
        if not self.training and (self.channel_matrix_cache is not None and 
            self.cached_c1 is not None and 
            torch.abs(self.cached_c1 - c1) < 1e-6 and
            self.channel_matrix_cache.shape[0] == N):
            return self.channel_matrix_cache
        
        # 创建W矩阵
        n = torch.arange(N, device=self.args.device, dtype=torch.float32)
        W = torch.diag(torch.exp(-1j * 2 * np.pi * n / N))
        
        # 创建Pi矩阵
        Pi = torch.zeros(N, N, dtype=torch.complex64, device=self.args.device)
        Pi[0, -1] = 1
        Pi[1:, :-1] = torch.eye(N-1, dtype=torch.complex64, device=self.args.device)
        
        # 预计算所有路径的矩阵
        H_p = torch.zeros(N, N, self.P, dtype=torch.complex64, device=self.args.device)
        
        for p in range(self.P):
            delay = int(self.normDelays[p].item())
            doppler = self.normDopplers[p].item()
            
            # 计算CPP矩阵
            cpp_diag = torch.ones(N, dtype=torch.complex64, device=self.args.device)
            for i in range(delay):
                exponent = -1j * 2 * np.pi * c1 * (N**2 - 2 * N * (delay - i))
                cpp_diag[i] = torch.exp(exponent)
            
            CPP_p = torch.diag(cpp_diag)
            
            # 计算W^doppler - 使用向量化方法
            W_diag = torch.diag(W)
            W_doppler = torch.diag(W_diag ** doppler)
            
            # 计算Pi^delay - 使用矩阵幂运算
            Pi_delay = torch.matrix_power(Pi, delay)
            
            # 累加每个路径的贡献
            H_p[:, :, p] = self.h[p] * CPP_p @ W_doppler @ Pi_delay
        
        # 求和所有路径
        H = torch.sum(H_p, dim=2)
        
        # 添加小的正则化项以确保矩阵非奇异
        H += 1e-8 * torch.eye(N, dtype=torch.complex64, device=self.args.device)
        
        # 仅在评估模式下更新缓存
        if not self.training:
            self.register_buffer('channel_matrix_cache', H)
            self.register_buffer('cached_c1', c1)
        
        return H

    def AFDM_modulation(self, X, c1, c2):
        """
        向量化的AFDM调制函数
        """
        batch_size, num_afdm, N = X.shape
        
        # 创建L1和L2矩阵
        n = torch.arange(N, device=X.device, dtype=torch.float32)
        L1 = torch.diag(torch.exp(-1j * 2 * np.pi * c1 * (n ** 2)))
        L2 = torch.diag(torch.exp(-1j * 2 * np.pi * c2 * (n ** 2)))
        
        # DFT矩阵
        F = torch.fft.fft(torch.eye(N, device=X.device, dtype=torch.complex64), dim=0) / np.sqrt(N)
        
        # AFDM调制矩阵
        IA = L1.conj().T @ F.conj().T @ L2.conj().T
        
        # 重塑X以便进行批量矩阵乘法
        X_reshaped = X.reshape(-1, N)  # [batch_size * num_afdm, N]
        
        # 应用调制
        S_reshaped = X_reshaped @ IA.T
        
        # 重塑回原始形状
        S = S_reshaped.reshape(batch_size, num_afdm, N)
        
        return S

    def AFDM_demodulation(self, S, c1, c2):
        """
        向量化的AFDM解调函数
        """
        batch_size, num_afdm, N = S.shape
        
        # 创建L1和L2矩阵
        n = torch.arange(N, device=S.device, dtype=torch.float32)
        L1 = torch.diag(torch.exp(-1j * 2 * np.pi * c1 * (n ** 2)))
        L2 = torch.diag(torch.exp(-1j * 2 * np.pi * c2 * (n ** 2)))
        
        # DFT矩阵
        F = torch.fft.fft(torch.eye(N, device=S.device, dtype=torch.complex64), dim=0) / np.sqrt(N)
        
        # AFDM解调矩阵
        A = L2 @ F @ L1
        
        # 重塑S以便进行批量矩阵乘法
        S_reshaped = S.reshape(-1, N)  # [batch_size * num_afdm, N]
        
        # 应用解调
        X_reshaped = S_reshaped @ A.T
        
        # 重塑回原始形状
        X = X_reshaped.reshape(batch_size, num_afdm, N)
        
        return X
    
    def get_afdm_params(self):
        """获取当前AFDM参数值"""
        return self.c1.item(), self.c2.item()

    def forward(self, x, stage=1, return_symbols=False):
        inputBS = x.shape[0]
        
        # =================================================================================== Encoding
        x_enc = self.Enc(x)
        # power norm
        x_norm = self.power_norm(x_enc)
        # reshape to a vector BS*256*2 (for construting complex symbols)
        x_reshaped = x_norm.view(inputBS, 256, 2)
        # (512 real symbols) to (256 complex symbols); power of real = 1; power of complex = 2
        x_symbols = torch.view_as_complex(x_reshaped)
        
        # =================================================================================== AFDM参数选择
        if stage == 1:
            # 第一阶段：使用固定AFDM参数
            c1 = self.fixed_c1.to(self.args.device)
            c2 = self.fixed_c2.to(self.args.device)
        else:
            # 第二阶段：使用神经网络生成的AFDM参数
            # 首先生成一个初始信道矩阵用于参数网络
            H_init = self.generate_channel_matrix(self.N, torch.tensor(0.1, device=self.args.device))
            
            # 使用神经网络生成AFDM参数c1和c2
            # 为批次中的每个样本生成参数
            c1_batch, c2_batch = self.afdm_param_net(H_init.unsqueeze(0).repeat(inputBS, 1, 1))
            
            # 使用批次中第一个样本的参数作为代表
            c1 = c1_batch[0]
            c2 = c2_batch[0]
        
        # =================================================================================== AFDM modulation
        numAFDM = int(256/self.N)
        x_afdm = x_symbols.view(inputBS, numAFDM, self.N)
        
        # AFDM调制
        data_t = self.AFDM_modulation(x_afdm, c1, c2)
        
        # 生成当前参数下的信道矩阵
        H = self.generate_channel_matrix(self.N, c1)
        
        # 对每个AFDM符号应用多径和多普勒
        data_t_channel = torch.zeros_like(data_t)
        for b in range(inputBS):
            for t in range(numAFDM):
                data_t_channel[b, t] = H @ data_t[b, t]
        
        # reshape back to a packet
        data_t = data_t_channel.view(inputBS, numAFDM * self.N)

        # =================================================================================== Channel
        # 添加噪声到基带信号
        signal_power = torch.mean(torch.abs(data_t)**2)
        snr_linear = 10**(self.args.snr / 10)
        noise_power = signal_power / snr_linear
        
        # 生成复高斯噪声
        noise_std = torch.sqrt(noise_power/2).item()  
        noise_real = torch.normal(0.0, noise_std, data_t.shape, device=self.args.device)
        noise_imag = torch.normal(0.0, noise_std, data_t.shape, device=self.args.device)
        noise = torch.complex(noise_real, noise_imag)
        data_r = data_t + noise
        
        # =================================================================================== demodulation
        # reshape to AFDM symbols
        data_r = data_r.view(inputBS, numAFDM, self.N)
        
        # AFDM解调
        y_demod = self.AFDM_demodulation(data_r, c1, c2)
        
        # =================================================================================== MMSE均衡 (时频域均衡)
        # 计算有效信道矩阵
        n = torch.arange(self.N, device=self.args.device, dtype=torch.float32)
        L1 = torch.diag(torch.exp(-1j * 2 * np.pi * c1 * (n ** 2)))
        L2 = torch.diag(torch.exp(-1j * 2 * np.pi * c2 * (n ** 2)))
        F = torch.fft.fft(torch.eye(self.N, device=self.args.device, dtype=torch.complex64), dim=0) / np.sqrt(self.N)
        
        # 调制和解调矩阵
        IA = L1.conj().T @ F.conj().T @ L2.conj().T  # 调制矩阵
        A = L2 @ F @ L1  # 解调矩阵
        
        # 计算等效信道矩阵
        H_eff = A @ H @ IA
        
        # 计算MMSE均衡器
        sigma2 = noise_power
        I = torch.eye(self.N, dtype=torch.complex64, device=self.args.device)
        W_mmse = torch.linalg.solve(
            H_eff.conj().T @ H_eff + sigma2 * I, 
            H_eff.conj().T
        )
        
        # 应用MMSE均衡
        y_reshaped = y_demod.reshape(-1, self.N)  # [inputBS * numAFDM, N]
        y_eq_reshaped = y_reshaped @ W_mmse.T
        y_eq = y_eq_reshaped.reshape(inputBS, numAFDM, self.N)
        
        # reshape to a packet, BS*4*64 -> BS*256
        y_symbols = y_eq.view(inputBS, numAFDM * self.N)
        
        if return_symbols:
            return x_symbols, y_symbols, c1, c2
        
        # complex to real, BS*256*2
        y_real = torch.view_as_real(y_symbols)
        # reshape to a compressed image
        y_reshaped_dec = y_real.view(y_real.size(0), 8, 8, 8)
        y_decoded = self.Dec(y_reshaped_dec)

        return y_decoded, torch.tensor(0.0), torch.tensor(0.0), c1, c2

def calculate_ber(x_symbols, y_symbols, M=16):
    """
    计算发射符号和接收符号之间的BER
    x_symbols: 发射符号 [batch_size, num_symbols]
    y_symbols: 接收符号 [batch_size, num_symbols]
    M: QAM调制阶数
    """
    # 将符号映射到最近的QAM星座点
    x_constellation = qam_modulate(qam_demodulate(x_symbols, M), M)
    y_constellation = qam_modulate(qam_demodulate(y_symbols, M), M)
    
    # 计算误码率
    errors = torch.sum(x_constellation != y_constellation)
    total_bits = x_symbols.numel() * int(math.log2(M))
    ber = errors.float() / total_bits
    
    return ber.item()

def qam_demodulate(symbols, M):
    """
    QAM解调，将符号映射到比特
    symbols: 复数符号
    M: 调制阶数
    """
    # 计算星座点并确保在正确的设备上
    constellation = create_qam_constellation(M).to(symbols.device)
    
    # 找到每个符号最近的星座点
    symbols_flat = symbols.view(-1)
    symbols_reshaped = symbols_flat.unsqueeze(1).expand(-1, len(constellation))
    distances = torch.abs(symbols_reshaped - constellation)
    indices = torch.argmin(distances, dim=1)
    
    return indices

def qam_modulate(bits, M):
    """
    QAM调制，将比特映射到符号
    bits: 比特序列
    M: 调制阶数
    """
    constellation = create_qam_constellation(M).to(bits.device)
    return constellation[bits]

def create_qam_constellation(M):
    """
    创建QAM星座点
    M: 调制阶数
    """
    # 计算星座点数量
    k = int(math.sqrt(M))
    if k**2 != M:
        raise ValueError("M must be a perfect square")
    
    # 创建星座点
    real_values = torch.linspace(-1, 1, k)
    imag_values = torch.linspace(-1, 1, k)
    constellation = torch.complex(
        real_values.repeat_interleave(k),
        imag_values.repeat(k)
    )
    
    # 归一化星座点功率
    power = torch.mean(torch.abs(constellation)**2)
    constellation = constellation / torch.sqrt(power)
    
    return constellation

def train_stage1(epoch, args, model, trainloader, best_PSNR, writer=None):
    """第一阶段训练：固定信道参数和AFDM参数，训练语义通信部分"""
    print('\nStage 1 - Epoch: %d' % epoch)
    model.train()
    
    # 固定信道参数（使用组0）
    model.set_channel_parameters(0)
    
    # 只训练编码器和解码器，冻结AFDM参数生成网络
    for name, param in model.named_parameters():
        if 'afdm_param_net' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    total_loss = 0.0
    batch_count = 0
    
    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(args.device)
        
        # 清零梯度
        args.optimizer_stage1.zero_grad()
        
        # 前向传播（使用阶段1）
        outputs, _, _, c1_val, c2_val = model(inputs, stage=1)
        
        # 计算损失
        loss = args.loss(outputs, inputs)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 优化器步进
        args.optimizer_stage1.step()
        
        # 累积统计信息
        total_loss += loss.item()
        batch_count += 1
        
        # 记录到TensorBoard (每个batch)
        if writer is not None:
            global_step = epoch * len(trainloader) + batch_idx
            writer.add_scalar('Stage1/Train/Batch_Loss', loss.item(), global_step)
            writer.add_scalar('Stage1/Train/Batch_MSE', loss.item(), global_step)
            writer.add_scalar('Stage1/Train/c1', c1_val.item(), global_step)
            writer.add_scalar('Stage1/Train/c2', c2_val.item(), global_step)
            
            # 记录图片输入输出 (每10个batch记录一次，避免日志过大)
            if batch_idx % 10 == 0:
                # 记录输入图片
                writer.add_images('Stage1/Images/Input', inputs[:4], global_step)
                # 记录输出图片
                writer.add_images('Stage1/Images/Output', outputs[:4], global_step)
                # 记录重建误差
                error_images = torch.abs(inputs[:4] - outputs[:4])
                writer.add_images('Stage1/Images/Error', error_images, global_step)
        
        # 显示进度
        progress_bar(batch_idx, len(trainloader), 'Stage1 - bestPSNR: %.2f, MSE: %.4f, c1: %.4f, c2: %.4f, SNR: %.1fdB'%(best_PSNR, loss.item(), c1_val.item(), c2_val.item(), args.snr))
    
    # 打印epoch总结
    avg_loss = total_loss / batch_count
    print(f'Stage1 - Epoch {epoch} Summary:')
    print(f'  Avg MSE: {format_number(avg_loss, 4)}')
    print(f'  c1: {format_number(c1_val.item(), 4)}')
    print(f'  c2: {format_number(c2_val.item(), 4)}')
    
    # 记录到TensorBoard (每个epoch)
    if writer is not None:
        writer.add_scalar('Stage1/Train/Epoch_Loss', avg_loss, epoch)
        writer.add_scalar('Stage1/Train/Epoch_c1', c1_val.item(), epoch)
        writer.add_scalar('Stage1/Train/Epoch_c2', c2_val.item(), epoch)

def train_stage2(epoch, args, model, trainloader, best_PSNR, writer=None):
    """第二阶段训练：固定语义编码器解码器，训练AFDM参数生成网络"""
    print('\nStage 2 - Epoch: %d' % epoch)
    model.train()
    
    # 每5个epoch变化一次信道参数
    model.set_channel_parameters(epoch)
    
    # 只训练AFDM参数生成网络，冻结编码器和解码器
    for name, param in model.named_parameters():
        if 'afdm_param_net' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    total_loss = 0.0
    batch_count = 0
    
    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(args.device)
        
        # 清零梯度
        args.optimizer_stage2.zero_grad()
        
        # 前向传播（使用阶段2）
        outputs, _, _, c1_val, c2_val = model(inputs, stage=2)
        
        # 计算损失
        loss = args.loss(outputs, inputs)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 优化器步进
        args.optimizer_stage2.step()
        
        # 累积统计信息
        total_loss += loss.item()
        batch_count += 1
        
        # 记录到TensorBoard (每个batch)
        if writer is not None:
            global_step = epoch * len(trainloader) + batch_idx
            writer.add_scalar('Stage2/Train/Batch_Loss', loss.item(), global_step)
            writer.add_scalar('Stage2/Train/Batch_MSE', loss.item(), global_step)
            writer.add_scalar('Stage2/Train/c1', c1_val.item(), global_step)
            writer.add_scalar('Stage2/Train/c2', c2_val.item(), global_step)
            
            # 记录图片输入输出 (每10个batch记录一次，避免日志过大)
            if batch_idx % 10 == 0:
                # 记录输入图片
                writer.add_images('Stage2/Images/Input', inputs[:4], global_step)
                # 记录输出图片
                writer.add_images('Stage2/Images/Output', outputs[:4], global_step)
                # 记录重建误差
                error_images = torch.abs(inputs[:4] - outputs[:4])
                writer.add_images('Stage2/Images/Error', error_images, global_step)
        
        # 显示进度
        progress_bar(batch_idx, len(trainloader), 'Stage2 - bestPSNR: %.2f, MSE: %.4f, c1: %.4f, c2: %.4f, SNR: %.1fdB'%(best_PSNR, loss.item(), c1_val.item(), c2_val.item(), args.snr))
    
    # 打印epoch总结
    avg_loss = total_loss / batch_count
    print(f'Stage2 - Epoch {epoch} Summary:')
    print(f'  Avg MSE: {format_number(avg_loss, 4)}')
    print(f'  c1: {format_number(c1_val.item(), 4)}')
    print(f'  c2: {format_number(c2_val.item(), 4)}')
    
    # 记录到TensorBoard (每个epoch)
    if writer is not None:
        writer.add_scalar('Stage2/Train/Epoch_Loss', avg_loss, epoch)
        writer.add_scalar('Stage2/Train/Epoch_c1', c1_val.item(), epoch)
        writer.add_scalar('Stage2/Train/Epoch_c2', c2_val.item(), epoch)

def test(epoch, args, model, testloader, best_PSNR, saveflag = 1, writer=None, stage=1):
    model.eval()
    psnr_all_list = []
    MSEnoAvg = nn.MSELoss(reduction = 'none')
    ber_list = []
    
    # 为测试设置固定的信道参数（使用组0）
    model.set_channel_parameters(0)
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(testloader):
            b,c,h,w=inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]
            inputs = inputs.to(args.device)
            
            # 获取输出和符号
            outputs, _, _, c1_val, c2_val = model(inputs, stage=stage)
            
            # 获取发射和接收符号
            x_symbols, y_symbols, _, _ = model(inputs, stage=stage, return_symbols=True)
            
            # 计算BER
            ber = calculate_ber(x_symbols, y_symbols, M=16)
            ber_list.append(ber)
            
            # 计算PSNR
            loss = MSEnoAvg(outputs, inputs)
            MSE_each_image = (torch.sum(loss.view(b,-1),dim=1))/(c*h*w)
            PSNR_each_image = 10 * torch.log10(1 / MSE_each_image)
            one_batch_PSNR = PSNR_each_image.data.cpu().numpy()
            psnr_all_list.extend(one_batch_PSNR)
            
        test_PSNR=np.mean(psnr_all_list)
        test_PSNR=np.around(test_PSNR,5)
        test_BER=np.mean(ber_list)
        
        # 计算平均MSE (从PSNR转换)
        # PSNR = 10 * log10(1/MSE), 所以 MSE = 10^(-PSNR/10)
        avg_MSE = np.mean([10**(-psnr/10) for psnr in psnr_all_list])
        
        stage_str = "Stage1" if stage == 1 else "Stage2"
        print(f"{stage_str} - Epoch {epoch} Test Results:")
        print(f"  Test PSNR: {format_number(test_PSNR, 5)}")
        print(f"  Test BER: {format_number(test_BER, 6)}")
        print(f"  Mean MSE: {format_number(avg_MSE, 6)}")
        print(f"  c1: {format_number(c1_val.item(), 4)}")
        print(f"  c2: {format_number(c2_val.item(), 4)}")
        print(f"  SNR: {format_number(args.snr, 1)}dB")
        print(f"  Best PSNR so far: {format_number(best_PSNR, 5)}")
        
        # 记录到TensorBoard
        if writer is not None:
            writer.add_scalar(f'{stage_str}/Test/PSNR', test_PSNR, epoch)
            writer.add_scalar(f'{stage_str}/Test/BER', test_BER, epoch)
            writer.add_scalar(f'{stage_str}/Test/MSE', avg_MSE, epoch)
            writer.add_scalar(f'{stage_str}/Test/c1', c1_val.item(), epoch)
            writer.add_scalar(f'{stage_str}/Test/c2', c2_val.item(), epoch)
            writer.add_scalar(f'{stage_str}/Test/SNR', args.snr, epoch)
            writer.add_scalar(f'{stage_str}/Test/Best_PSNR', best_PSNR, epoch)
            
            # 记录测试图片 (每个epoch记录一次)
            # 获取第一个batch的图片进行记录
            testloader_iter = iter(testloader)
            test_inputs, _ = next(testloader_iter)
            test_inputs = test_inputs.to(args.device)
            test_outputs, _, _, c1_test, c2_test = model(test_inputs, stage=stage)
            
            # 记录测试输入图片
            writer.add_images(f'{stage_str}/Test/Input', test_inputs[:4], epoch)
            # 记录测试输出图片
            writer.add_images(f'{stage_str}/Test/Output', test_outputs[:4], epoch)
            # 记录测试重建误差
            test_error_images = torch.abs(test_inputs[:4] - test_outputs[:4])
            writer.add_images(f'{stage_str}/Test/Error', test_error_images, epoch)

    if saveflag == 1:
        # Save checkpoint.
        if test_PSNR > best_PSNR:
            print(f'New best PSNR achieved! Saving checkpoint...')
            print(f'  Previous best: {format_number(best_PSNR, 5)}')
            print(f'  New best: {format_number(test_PSNR, 5)}')
            print(f'  Improvement: {format_number(test_PSNR - best_PSNR, 5)}')
            
            # 记录到TensorBoard
            if writer is not None:
                writer.add_scalar(f'{stage_str}/Checkpoint/New_Best_PSNR', test_PSNR, epoch)
                writer.add_scalar(f'{stage_str}/Checkpoint/Improvement', test_PSNR - best_PSNR, epoch)
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'test_PSNR': test_PSNR,
                'test_BER': test_BER,
            }

            filename = f"{stage_str}_snr{int(args.snr)}_precoding{args.precoding}_mapping{args.mapping}_lamb{args.lamb}_clip{args.clip}_fading{args.fading}_hstd{args.hstd}"
            # save checkpoint
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/'+ filename + '.pth')
            best_PSNR = test_PSNR

            # save matlab file
            mdic = {"PSNR": best_PSNR, "BER": test_BER}
            savemat("./checkpoint/" + filename + ".mat", mdic)

        return best_PSNR

def format_number(value, decimal_places=4):
    """格式化数值，处理NaN和Inf"""
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return "N/A"
        return f"{value:.{decimal_places}f}"
    elif hasattr(value, 'item'):
        val = value.item()
        if np.isnan(val) or np.isinf(val):
            return "N/A"
        return f"{val:.{decimal_places}f}"
    else:
        return str(value)

def main(model, args):
    # ======================================================= load datasets
    trainloader, testloader = load_CIFAR10(args.BatchSize)

    # ======================================================= start (train or test)
    if args.load == 0:
        # 初始化TensorBoard
        log_dir = f"runs/AFDM_snr{args.snr}_stage1_{args.numepoch1}_stage2_{args.numepoch2}"
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")
        
        # 第一阶段训练：固定信道参数和AFDM参数，训练语义通信部分
        best_PSNR_stage1 = 0
        print(f"\n{'='*60}")
        print(f"Starting Stage 1 Training for {args.numepoch1} epochs")
        print(f"Initial best PSNR: {format_number(best_PSNR_stage1, 5)}")
        print(f"{'='*60}")
        
        for epoch in np.arange(args.numepoch1):
            print(f"\n{'='*40} Stage 1 - Epoch {epoch+1}/{args.numepoch1} {'='*40}")
            train_stage1(epoch, args, model, trainloader, best_PSNR_stage1, writer)
            best_PSNR_stage1 = test(epoch, args, model, testloader, best_PSNR_stage1, 1, writer, stage=1)
            
            # 打印当前学习率
            current_lr = args.optimizer_stage1.param_groups[0]['lr']
            print(f"Current Learning Rate: {current_lr:.6f}")
            
            # 记录学习率到TensorBoard
            writer.add_scalar('Stage1/Training/Learning_Rate', current_lr, epoch)
            
            args.scheduler_stage1.step()
            
            print(f"Stage 1 - Epoch {epoch+1} completed. Best PSNR so far: {format_number(best_PSNR_stage1, 5)}")
            print(f"{'='*80}")
        
        print(f"\nStage 1 Training completed! Final best PSNR: {format_number(best_PSNR_stage1, 5)}")
        
        # 第二阶段训练：固定语义编码器解码器，训练AFDM参数生成网络
        best_PSNR_stage2 = best_PSNR_stage1
        print(f"\n{'='*60}")
        print(f"Starting Stage 2 Training for {args.numepoch2} epochs")
        print(f"Initial best PSNR: {format_number(best_PSNR_stage2, 5)}")
        print(f"{'='*60}")
        
        for epoch in np.arange(args.numepoch2):
            print(f"\n{'='*40} Stage 2 - Epoch {epoch+1}/{args.numepoch2} {'='*40}")
            train_stage2(epoch, args, model, trainloader, best_PSNR_stage2, writer)
            best_PSNR_stage2 = test(epoch, args, model, testloader, best_PSNR_stage2, 1, writer, stage=2)
            
            # 打印当前学习率
            current_lr = args.optimizer_stage2.param_groups[0]['lr']
            print(f"Current Learning Rate: {current_lr:.6f}")
            
            # 记录学习率到TensorBoard
            writer.add_scalar('Stage2/Training/Learning_Rate', current_lr, epoch)
            
            args.scheduler_stage2.step()
            
            print(f"Stage 2 - Epoch {epoch+1} completed. Best PSNR so far: {format_number(best_PSNR_stage2, 5)}")
            print(f"{'='*80}")
        
        print(f"\nStage 2 Training completed! Final best PSNR: {format_number(best_PSNR_stage2, 5)}")
        
        # 关闭TensorBoard writer
        writer.close()
        print(f"TensorBoard logs saved to: {log_dir}")
        print(f"To view logs, run: tensorboard --logdir={log_dir}")
    else:
        # load a trained model and test
        filename = "snr" + str(int(args.snr)) + "_precoding" + str(args.precoding) + "_mapping" + str(args.mapping)+ "_lamb" + str(args.lamb) + "_clip" + str(0.0) + "_fading" + str(args.fading) + "_hstd" + str(args.hstd)
        checkpoint = torch.load("./checkpoint/" + filename + ".pth")
        model.load_state_dict(checkpoint['model'])
        print("=======>>>>>>>> Successfully load the pretrained data!")
        test(0, args, model, testloader, 0, saveflag = 1)
        # pdb.set_trace()

if __name__ == '__main__':
    # ======================================================= parse args
    args = args_parser()
    args.device = 'cuda' 
    args.loss = nn.MSELoss()
    print(args.device)

    # 添加两阶段训练的epoch数
    args.numepoch1 = 100  # 第一阶段epoch数
    args.numepoch2 = 100  # 第二阶段epoch数
    
    # ======================================================= Initialize the model
    model = SemanticComm(args).to(args.device)

    # ======================================================= 第一阶段优化器 (语义通信部分)
    # 只优化编码器和解码器
    stage1_params = []
    for name, param in model.named_parameters():
        if 'afdm_param_net' not in name:
            stage1_params.append(param)
    
    if args.adamW == 1:
        args.optimizer_stage1 = torch.optim.AdamW(stage1_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd, amsgrad=False)
    else:
        args.optimizer_stage1 = torch.optim.Adam(stage1_params, lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    # 第一阶段学习率调度
    lambdafn_stage1 = lambda epoch: (1-epoch/args.numepoch1)
    args.scheduler_stage1 = torch.optim.lr_scheduler.LambdaLR(args.optimizer_stage1, lr_lambda=lambdafn_stage1)

    # ======================================================= 第二阶段优化器 (AFDM参数生成网络)
    # 只优化AFDM参数生成网络
    stage2_params = []
    for name, param in model.named_parameters():
        if 'afdm_param_net' in name:
            stage2_params.append(param)
    
    # 第二阶段使用较小的学习率
    lr_stage2 = args.lr * 0.5  # 第二阶段学习率为第一阶段的0.5倍
    
    if args.adamW == 1:
        args.optimizer_stage2 = torch.optim.AdamW(stage2_params, lr=lr_stage2, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd, amsgrad=False)
    else:
        args.optimizer_stage2 = torch.optim.Adam(stage2_params, lr=lr_stage2, betas=(0.9, 0.98), eps=1e-9)

    # 第二阶段学习率调度
    lambdafn_stage2 = lambda epoch: (1-epoch/args.numepoch2)
    args.scheduler_stage2 = torch.optim.lr_scheduler.LambdaLR(args.optimizer_stage2, lr_lambda=lambdafn_stage2)

    main(model, args)
