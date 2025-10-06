import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PSNR(nn.Module):
    def __init__(self, max_val=1.0):
        super(PSNR, self).__init__()
        self.max_val = max_val

    def forward(self, input, target):
        # 如果是 3D 输入（单张图像），添加批次维度
        if input.dim() == 3:
            input = input.unsqueeze(0)
            target = target.unsqueeze(0)
        
        # 计算每个图像的 MSE
        mse = F.mse_loss(input, target, reduction='none').mean(dim=[1, 2, 3])
        psnr = 10 * torch.log10(self.max_val ** 2 / (mse + 1e-10))
        
        # 如果原输入是 3D，返回单个值；否则返回批次 PSNR
        return psnr[0] if psnr.dim() == 1 and len(psnr) == 1 else psnr

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

class SSIM(nn.Module):
    def __init__(self, window_size=11, channel=3, data_range=1.0):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.data_range = data_range
        self.window = create_window(window_size, channel)

    def forward(self, img1, img2):
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_values = ssim_map.mean(dim=[1, 2, 3])  # 每张图像的 SSIM
        
        return ssim_values[0] if ssim_values.dim() == 1 and len(ssim_values) == 1 else ssim_values