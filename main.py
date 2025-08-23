import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch, time, pdb, os, random, math
from utils import load_CIFAR10, get_RRC, args_parser, progress_bar
from resnet import ResNetTx, ResNetRx
from scipy.io import savemat

# 设置随机种子
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

class AFDM_Modulator(nn.Module):
    """
    AFDM调制器模块
    """
    def __init__(self, N, device):
        super(AFDM_Modulator, self).__init__()
        self.N = N
        self.device = device
        
        # 可训练的AFDM参数
        self.c1 = nn.Parameter(torch.tensor(0.0))
        self.c2 = nn.Parameter(torch.tensor(0.0))
        
        # 预计算索引矩阵
        n = torch.arange(0, N, device=device).float()
        self.register_buffer('n_vec', n)
        self.register_buffer('n_sq_mat', torch.outer(n, n))
        self.register_buffer('n_mat', n.unsqueeze(0).repeat(N, 1))
        self.register_buffer('m_mat', n.unsqueeze(1).repeat(1, N))
    
    def forward(self, x_freq):
        """
        AFDM调制
        x_freq: [batch, num_symbols, N] 频域信号
        返回: [batch, num_symbols, N] 时域信号
        """
        batch_size, num_symbols, N = x_freq.shape
        
        # 计算调制矩阵
        mod_matrix = torch.exp(1j * 2 * torch.pi * (
            self.c1 * self.n_sq_mat + 
            self.n_mat * self.m_mat / self.N + 
            self.c2 * self.m_mat**2
        ))
        
        # 应用调制矩阵
        x_time = torch.matmul(x_freq, mod_matrix) / torch.sqrt(torch.tensor(N, dtype=torch.float32))
        
        return x_time
    
    def inverse(self, y_time):
        """
        AFDM解调
        y_time: [batch, num_symbols, N] 时域信号
        返回: [batch, num_symbols, N] 频域信号
        """
        batch_size, num_symbols, N = y_time.shape
        
        # 计算解调矩阵
        demod_matrix = torch.exp(-1j * 2 * torch.pi * (
            self.c1 * self.n_sq_mat + 
            self.n_mat * self.m_mat / self.N + 
            self.c2 * self.m_mat**2
        ))
        
        # 应用解调矩阵
        y_freq = torch.matmul(y_time, demod_matrix) / torch.sqrt(torch.tensor(N, dtype=torch.float32))
        
        return y_freq

class SemanticComm(nn.Module):
    def __init__(self, args):
        super(SemanticComm, self).__init__()
        self.args = args
        self.M = args.M
        self.N = args.N
        self.Enc = ResNetTx()
        self.Dec = ResNetRx()
        
        # 添加AFDM调制器
        self.afdm_modulator = AFDM_Modulator(self.N, args.device)
        
        # 将编码器和解码器移动到指定设备
        self.Enc = self.Enc.to(args.device)
        self.Dec = self.Dec.to(args.device)
        self.afdm_modulator = self.afdm_modulator.to(args.device)
        
        # load RRC pulse
        self.RRC, self.lenRRC = get_RRC()
        self.RRC = self.RRC.to(args.device)  # 将RRC脉冲移动到GPU
        
        # generate carrier
        fc = 25e6  # carrier frequency
        fb = 10e6  # baseband frequency
        fs = fb * self.args.sps  # sampling rate
        numSamples = int(256/self.N) * (self.M+self.args.lenCP) * self.args.sps + self.lenRRC * 2  # number of samples 
        tt = torch.arange(0, numSamples/fs, 1/fs, device=args.device)  # 直接在GPU上创建
        self.carrier_cos = torch.sqrt(torch.tensor(2.0, device=args.device)) * torch.cos(2 * torch.pi * fc * tt)
        self.carrier_sin = torch.sqrt(torch.tensor(2.0, device=args.device)) * torch.sin(2 * torch.pi * fc * tt)

    def power_norm(self, feature):
        in_shape = feature.shape
        sig_in = feature.reshape(in_shape[0], -1)
        # each pkt in a batch, compute the mean and var
        sig_mean = torch.mean(sig_in, dim = 1)
        sig_std = torch.std(sig_in, dim = 1)
        # normalize
        sig_out = (sig_in-sig_mean.unsqueeze(dim=1))/(sig_std.unsqueeze(dim=1)+1e-8)
        return sig_out.reshape(in_shape)

    def pulse_filter_complex_sig(self, x):
        # RRC已经在初始化时移动到GPU，无需重复移动
        rrc_filter = self.RRC.flip(dims=[2])
        xreal = F.conv1d(x[:,:,0].unsqueeze(1), rrc_filter, stride=1, padding=self.lenRRC-1)
        ximag = F.conv1d(x[:,:,1].unsqueeze(1), rrc_filter, stride=1, padding=self.lenRRC-1)
        # ---------------------------------------------------------- convert back to complex
        x = torch.cat([xreal.squeeze(1).unsqueeze(2),ximag.squeeze(1).unsqueeze(2)], dim=2)
        return torch.view_as_complex(x)

    def compute_PAPR(self, x_t, len_data_t):
        # truncation, only compute the PAPR of the signal part (keep the signal length to len_data_t)
        truncloc = torch.arange(int((self.lenRRC+1)/2),int(((self.lenRRC+1)/2+len_data_t*self.args.sps)), 
                              device=self.args.device)
        x_t = torch.index_select(x_t, 1, truncloc)
        # ---------------------------------------------------------- compute PAPR
        data_t_power = torch.square(torch.abs(x_t))
        meanPower = torch.mean(data_t_power, dim = 1)
        maxPower = torch.max(data_t_power, dim = 1).values
        PAPRdB = 10 * torch.log10(maxPower/meanPower)
        PAPRloss = F.relu(PAPRdB-self.args.thres).mean()
        return PAPRdB, PAPRloss

    def channel(self, data_x):
        inputBS, len_data_x = data_x.size(0), data_x.size(1)
        noise_std = 10 ** (-self.args.snr * 1.0 / 10 / 2)
        # real channel - 直接在GPU上生成噪声
        AWGN = torch.normal(0, std=noise_std, size=(inputBS, len_data_x), 
                           device=self.args.device, requires_grad=False)

        if self.args.fading == 1:
            # 在GPU上生成瑞利衰落
            self.hh = torch.sqrt(torch.randn(1, device=self.args.device)**2 + 
                               torch.randn(1, device=self.args.device)**2) * self.args.hstd
        else:
            self.hh = torch.tensor([1.0], device=self.args.device)

        data_r = self.hh * data_x + AWGN
        return data_r

    def clip(self, x):
        x_power_mean_amp = torch.sqrt(torch.mean(torch.square(x), dim = 1))
        thres = self.args.clip * x_power_mean_amp
        non_neg_diff = F.relu(torch.abs(x) - thres.unsqueeze(1))
        x = (1 - non_neg_diff/(torch.abs(x)+1e-8)) * x # scale the symbol with amplitude larger than thres
        return x

    def forward(self, x):
        inputBS = x.shape[0]
        # =================================================================================== Encoding
        # the dim of enc output is fixed to [BS, 512] (rate = 1/12)
        x = self.Enc(x)
        # power norm
        x = self.power_norm(x)
        # reshape to a vector BS*256*2 (for construting complex symbols)
        x = x.view(inputBS, 256, 2)
        # (512 real symbols) to (256 complex symbols); power of real = 1; power of complex = 2
        x = torch.view_as_complex(x)
        # =================================================================================== AFDM modulation
        # total subcarriers = M = 128, # allocated subcarriers = N = 64
        numOFDM = int(256/self.N)
        # ---------------------------------------------------------- modulation
        # oneOFDM = x[:, (idx*self.N):((idx+1)*self.N)] # take out each OFDM symbols separately
        x = x.view(inputBS, numOFDM, self.N)
        
        # 使用AFDM调制代替原有的IFFT
        data_t = self.afdm_modulator(x)
        
        # ---------------------------------------------------------- add CP
        len_data_t = numOFDM * (self.M+self.args.lenCP)
        if self.args.lenCP != 0:
            data_t = torch.cat([data_t[:,:,-self.args.lenCP:], data_t], dim=-1)
        # reshape back to a packet
        data_t = data_t.view(inputBS, len_data_t)

        # ---------------------------------------------------------- oversampling
        data_t_over = torch.zeros(inputBS, len_data_t*self.args.sps, dtype = torch.complex64, device=self.args.device)
        data_t_over[:, torch.arange(0, len_data_t*self.args.sps, self.args.sps, device=self.args.device)] = data_t
        # ---------------------------------------------------------- pulse shaping (real in, complex out)
        data_t_over = torch.view_as_real(data_t_over)
        data_x = self.pulse_filter_complex_sig(data_t_over)
        # ---------------------------------------------------------- RF signal (real)
        data_x = data_x.real * self.carrier_cos[:data_x.size(1)] - data_x.imag * self.carrier_sin[:data_x.size(1)]
        # ---------------------------------------------------------- clipping
        if self.args.clip != 0.0:
            data_x = self.clip(data_x)
        # ---------------------------------------------------------- compute PAPR
        PAPRdB, PAPRloss = self.compute_PAPR(data_x, len_data_t)

        # =================================================================================== Channel
        data_r = self.channel(data_x)

        # =================================================================================== AFDM demodulation
        # ---------------------------------------------------------- baseband signal
        data_r_real = data_r * self.carrier_cos[:data_r.size(1)]
        data_r_imag = data_r * -self.carrier_sin[:data_r.size(1)]
        data_r_cpx = torch.cat([data_r_real.unsqueeze(2),data_r_imag.unsqueeze(2)], dim=2)

        # ---------------------------------------------------------- matched filtering (real in, complex out)
        data_r_filtered = self.pulse_filter_complex_sig(data_r_cpx)

        # synchronization and samling
        samplingloc = torch.arange(self.lenRRC-1, data_r_filtered.size(1)-self.lenRRC, self.args.sps, device=self.args.device)
        y = torch.index_select(data_r_filtered, 1, samplingloc)
  
        # remove CP
        y = y.view(inputBS, numOFDM, self.M+self.args.lenCP)
        y = y[:,:,self.args.lenCP:]
        
        # 使用AFDM解调代替原有的FFT
        y = self.afdm_modulator.inverse(y)
        
        # reshape to a packet, BS*4*64 -> BS*256
        y = y.view(inputBS, numOFDM * self.N)
        # complex to real, BS*256*2
        y = torch.view_as_real(y)
        # reshape to a compressed image
        y = y.view(y.size(0), 8, 8, 8)
        y = self.Dec(y)

        # 约束c1和c2在[0,1)范围内
        with torch.no_grad():
            self.afdm_modulator.c1.data = self.afdm_modulator.c1.data % 1
            self.afdm_modulator.c2.data = self.afdm_modulator.c2.data % 1

        return y, PAPRdB, PAPRloss

def train(epoch, args, model, trainloader, best_PSNR):
    # ============================================= training
    print('\nEpoch: %d' % epoch)
    model.train()
    
    # 记录c1和c2的值
    c1_values = []
    c2_values = []
    
    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(args.device)
        args.optimizer.zero_grad()
        outputs, PAPRdB, PAPRloss = model(inputs)
        if args.lamb == 0.0:
            loss = args.loss(outputs, inputs)
        else:
            loss = args.loss(outputs, inputs) + args.lamb * PAPRloss
        loss.backward()
        args.optimizer.step()
        
        # 记录c1和c2的值
        c1_values.append(model.afdm_modulator.c1.item())
        c2_values.append(model.afdm_modulator.c2.item())
        
        # GPU内存监控
        if args.device == 'cuda' and batch_idx % 100 == 0:
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            progress_bar(batch_idx, len(trainloader), 
                        'bestPSNR: %.2f, MSE: %.4f, PAPRdB: %.4f, c1: %.4f, c2: %.4f, GPU: %.2fGB'%
                        (best_PSNR, loss, PAPRdB.mean(), c1_values[-1], c2_values[-1], gpu_memory))
        else:
            progress_bar(batch_idx, len(trainloader), 
                        'bestPSNR: %.2f, MSE: %.4f, PAPRdB: %.4f, c1: %.4f, c2: %.4f'%
                        (best_PSNR, loss, PAPRdB.mean(), c1_values[-1], c2_values[-1]))
    
    # 打印c1和c2的平均值
    avg_c1 = sum(c1_values) / len(c1_values)
    avg_c2 = sum(c2_values) / len(c2_values)
    print(f'Epoch {epoch}: c1 = {avg_c1:.4f}, c2 = {avg_c2:.4f}')

def test(epoch, args, model, testloader, best_PSNR, saveflag = 1):
    model.eval()
    psnr_all_list = []
    MSEnoAvg = nn.MSELoss(reduction = 'none')
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(testloader):
            b,c,h,w=inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]
            inputs = inputs.to(args.device)
            outputs, PAPRdB, _ = model(inputs)
            loss = MSEnoAvg(outputs, inputs)
            MSE_each_image = (torch.sum(loss.view(b,-1),dim=1))/(c*h*w)
            PSNR_each_image = 10 * torch.log10(1 / MSE_each_image)
            one_batch_PSNR = PSNR_each_image.data.cpu().numpy()
            psnr_all_list.extend(one_batch_PSNR)
            if batch_idx == 0:
                PAPRdBarray = PAPRdB
            else:
                PAPRdBarray = torch.cat([PAPRdBarray, PAPRdB],dim=0)
        test_PSNR=np.mean(psnr_all_list)
        test_PSNR=np.around(test_PSNR,5)
        
        # 打印最终的c1和c2值
        final_c1 = model.afdm_modulator.c1.item()
        final_c2 = model.afdm_modulator.c2.item()
        print("test_PSNR = {}, meanPAPR = {}, final_c1 = {:.4f}, final_c2 = {:.4f}".format(
            test_PSNR, PAPRdBarray.mean().cpu().numpy(), final_c1, final_c2))

    if saveflag == 1:
        # Save checkpoint.
        if test_PSNR > best_PSNR:
            print('Saving..')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'test_PSNR': test_PSNR,
                'PAPR': PAPRdBarray,
                'c1': model.afdm_modulator.c1.item(),
                'c2': model.afdm_modulator.c2.item(),
            }

            filename = "afdm_snr" + str(int(args.snr)) + "_c1_" + str(round(model.afdm_modulator.c1.item(), 4)) + "_c2_" + str(round(model.afdm_modulator.c2.item(), 4)) + "_precoding" + str(args.precoding) + "_mapping" + str(args.mapping)+ "_lamb" + str(args.lamb) + "_clip" + str(args.clip) + "_fading" + str(args.fading) + "_hstd" + str(args.hstd)
            # save checkpoint
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/'+ filename + '.pth')
            best_PSNR = test_PSNR

            # save matlab file
            mdic = {
                "PSNR": best_PSNR, 
                "PAPRarray": PAPRdBarray.cpu().numpy(),
                "c1": model.afdm_modulator.c1.item(),
                "c2": model.afdm_modulator.c2.item()
            }
            savemat("./checkpoint/" + filename + ".mat", mdic)

        return best_PSNR


def main(model, args):
    # ======================================================= load datasets
    trainloader, testloader = load_CIFAR10(args.BatchSize)

    # ======================================================= start (train or test)
    if args.load == 0:
        # start training
        best_PSNR = 0
        for epoch in np.arange(args.numepoch):
            train(epoch, args, model, trainloader, best_PSNR)
            best_PSNR = test(epoch, args, model, testloader, best_PSNR)

            args.scheduler.step()
    else:
        # load a trained model and test
        filename = "afdm_snr" + str(int(args.snr)) + "_precoding" + str(args.precoding) + "_mapping" + str(args.mapping)+ "_lamb" + str(args.lamb) + "_clip" + str(0.0) + "_fading" + str(args.fading) + "_hstd" + str(args.hstd)
        checkpoint = torch.load("./checkpoint/" + filename + ".pth")
        model.load_state_dict(checkpoint['model'])
        # 加载保存的c1和c2值
        model.afdm_modulator.c1.data = torch.tensor(checkpoint['c1'], device=args.device)
        model.afdm_modulator.c2.data = torch.tensor(checkpoint['c2'], device=args.device)
        print("=======>>>>>>>> Successfully load the pretrained data!")
        test(0, args, model, testloader, 0, saveflag = 1)

if __name__ == '__main__':
    # ======================================================= parse args
    args = args_parser()
    
    # GPU设置和优化
    if torch.cuda.is_available():
        args.device = 'cuda'
        # 设置CUDA设备
        torch.cuda.set_device(0)  # 使用第一个GPU
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 启用CUDA优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # 清空GPU缓存
        torch.cuda.empty_cache()
    else:
        args.device = 'cpu'
        print("CUDA不可用，使用CPU")
    
    args.loss = nn.MSELoss()
    # ======================================================= Initialize the model
    model = SemanticComm(args).to(args.device)
    
    # GPU优化设置
    if args.device == 'cuda':
        # 使用DataParallel进行多GPU训练（如果有多个GPU）
        if torch.cuda.device_count() > 1:
            print(f"检测到 {torch.cuda.device_count()} 个GPU，使用DataParallel")
            model = torch.nn.DataParallel(model)
        else:
            print("使用单GPU训练")
        
        # 将模型移动到GPU
        model = model.to(args.device)
        
        # 打印模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")

    # ======================================================= Optimizer
    # 为AFDM参数设置不同的学习率
    afdm_params = [model.afdm_modulator.c1, model.afdm_modulator.c2]
    other_params = [p for n, p in model.named_parameters() if not any(nd in n for nd in ['afdm_modulator.c1', 'afdm_modulator.c2'])]
    
    if args.adamW == 1:
        args.optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': args.lr},
            {'params': afdm_params, 'lr': args.lr * 0.1}  # AFDM参数使用较低的学习率
        ], betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd, amsgrad=False)
    else:
        args.optimizer = torch.optim.Adam([
            {'params': other_params, 'lr': args.lr},
            {'params': afdm_params, 'lr': args.lr * 0.1}  # AFDM参数使用较低的学习率
        ], betas=(0.9, 0.98), eps=1e-9)

    # ======================================================= lr scheduling
    lambdafn = lambda epoch: (1-epoch/args.numepoch)
    args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda=lambdafn)

    main(model, args)