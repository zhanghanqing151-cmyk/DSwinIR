import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

class TensorboardVisualizer:
    def __init__(self, log_dir='visual/logs', fig_save_dir=None):
        """初始化TensorBoard可视化工具"""
        self.log_dir = log_dir
        self.fig_save_dir = fig_save_dir or log_dir  # 图表保存目录，默认与日志目录相同
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.fig_save_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        # 存储验证集前十张图片的固定索引
        self.val_image_indices = list(range(10))
        self.val_images = {}  # 缓存验证集图片

    def cache_val_images(self, val_loader, device):
        """缓存验证集前十张图片用于后续对比"""
        self.val_images.clear()
        for idx, (input_img, target_img) in enumerate(val_loader):
            if idx in self.val_image_indices:
                self.val_images[idx] = {
                    'input': input_img.to(device),
                    'target': target_img.to(device)
                }
            if len(self.val_images) >= 10:
                break

    def add_scalar(self, tag, value, step):
        """添加标量数据（loss、PSNR、SSIM等）"""
        self.writer.add_scalar(tag, value, step)

    def add_losses(self, train_loss, val_loss, epoch):
        """添加训练集和验证集损失"""
        self.add_scalar('Loss/Train', train_loss, epoch)
        self.add_scalar('Loss/Validation', val_loss, epoch)

    def add_metrics(self, train_psnr, train_ssim, val_psnr, val_ssim, epoch):
        """添加PSNR和SSIM指标"""
        self.add_scalar('PSNR/Train', train_psnr, epoch)
        self.add_scalar('PSNR/Validation', val_psnr, epoch)
        self.add_scalar('SSIM/Train', train_ssim, epoch)
        self.add_scalar('SSIM/Validation', val_ssim, epoch)

    def visualize_val_images(self, model, epoch, device):
        """可视化验证集前十张图片的恢复过程"""
        model.eval()
        with torch.no_grad():
            for idx in self.val_image_indices:
                if idx not in self.val_images:
                    continue
                
                # 获取缓存的图片
                data = self.val_images[idx]
                input_img = data['input']
                target_img = data['target']
                
                # 获取模型输出
                outputs = model(input_img)
                # 取所有迭代步骤的输出（用于展示恢复过程）
                step_outputs = outputs
                
                # 准备可视化的图片列表
                visualize_list = [input_img[0], target_img[0]]  # 输入和目标图
                visualize_list.extend([out[0] for out in step_outputs])  # 各步骤输出
                
                # 归一化处理
                visualize_list = [self.normalize_img(img) for img in visualize_list]
                
                # 制作网格图
                grid = make_grid(visualize_list, nrow=len(visualize_list), padding=2)
                
                # 添加到TensorBoard
                self.writer.add_image(
                    f'Validation_Recovery/Image_{idx}',
                    grid,
                    global_step=epoch,
                    dataformats='CHW'
                )
        model.train()

    @staticmethod
    def normalize_img(img):
        """将图片归一化到[0,1]范围"""
        img = img.cpu().detach()
        min_val = torch.min(img)
        max_val = torch.max(img)
        return (img - min_val) / (max_val - min_val + 1e-8)

    def save_training_metrics_plots(self, loss_list, train_psnr_history, val_psnr_history,
                                   train_ssim_history, val_ssim_history, val_loss_history, epochs):
        """保存训练过程中的指标变化图表"""
        # 创建一个包含4个子图的图表
        plt.figure(figsize=(16, 12))
        
        # 1. 训练损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(range(1, epochs+1), loss_list, label='Training Loss')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # 2. PSNR曲线 (训练和验证)
        plt.subplot(2, 2, 2)
        plt.plot(range(1, epochs+1), train_psnr_history, label='Training PSNR')
        # 验证PSNR是每5个epoch记录一次，调整x轴坐标
        val_epochs = [i*5 + 1 for i in range(len(val_psnr_history))]
        plt.plot(val_epochs, val_psnr_history, label='Validation PSNR', color='orange')
        plt.title('PSNR Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.grid(True)
        plt.legend()
        
        # 3. SSIM曲线 (训练和验证)
        plt.subplot(2, 2, 3)
        plt.plot(range(1, epochs+1), train_ssim_history, label='Training SSIM')
        plt.plot(val_epochs, val_ssim_history, label='Validation SSIM', color='orange')
        plt.title('SSIM Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.grid(True)
        plt.legend()
        
        # 4. 验证损失曲线
        plt.subplot(2, 2, 4)
        plt.plot(val_epochs, val_loss_history, label='Validation Loss', color='green')
        plt.title('Validation Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        training_metrics_path = os.path.join(self.fig_save_dir, 'training_metrics.png')
        plt.savefig(training_metrics_path, dpi=150)
        plt.close()

    def save_per_image_metrics_plots(self, psnr_values, ssim_values):
        """保存每张图片的PSNR和SSIM指标图表"""
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(psnr_values, marker='o')
        plt.title('Per-image PSNR')
        plt.xlabel('Image index')
        plt.ylabel('PSNR (dB)')

        plt.subplot(1, 2, 2)
        plt.plot(ssim_values, marker='o', color='orange')
        plt.title('Per-image SSIM')
        plt.xlabel('Image index')
        plt.ylabel('SSIM')
        plt.tight_layout()
        
        # 保存指标图
        metrics_plot_path = os.path.join(self.fig_save_dir, 'per_image_metrics.png')
        plt.savefig(metrics_plot_path, dpi=150)
        plt.close()

    def close(self):
        """关闭TensorBoard写入器"""
        self.writer.close()

# tensorboard --logdir=run/result1/logs