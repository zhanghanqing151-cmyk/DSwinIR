import sys
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm, trange  # 进度条
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import utils
from utils import SSIM
from basicsr.archs.MSDeformableNAIR import MSDeformableNAIR  # 导入新的模型
from MyDataset import *
from collections import OrderedDict  # 导入有序字典类
import torch.nn.functional as F 


if __name__ == '__main__':

    inputPathTrain = f'/home/zgf/桌面/dataset/allweather5000/input/'
    targetPathTrain = f'/home/zgf/桌面/dataset/allweather5000/gt/' 
    inputPathTest = f'/home/zgf/桌面/dataset/allweather5000/inputTest/' 
    resultPathTest = f'/home/zgf/桌面/dataset/allweather5000/resultTest/'  
    targetPathTest = f'/home/zgf/桌面/dataset/allweather5000/gtTest/' 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    myNet = MSDeformableNAIR(
        inp_channels=3,      
        out_channels=3,      
        dim=48,              
        num_blocks=[4, 6, 6, 8],
        kernel_size=[3, 3, 3, 3],
        dilation=[1, 1, 1, 1],
        heads=[3, 4, 8, 8],
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type='WithBias',
        rel_pos_bias=True,
        global_residual=True
    )
    myNet.to(device)

    psnr = utils.PSNR()  # 实例化峰值信噪比计算类
    psnr = psnr.cuda()
    ssim = utils.SSIM()  # 实例化结构相似性计算类
    ssim = ssim.cuda()


    # 测试数据
    datasetTest = MyTestDataSet(inputPathTest)  # 实例化测试数据集类
    # 可迭代数据加载器加载测试数据
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=False, num_workers=6,
                            pin_memory=True)
    # 计算数据
    datasetCom = MyComDataSet(resultPathTest, targetPathTest)  # 实例化计算数据集类
    # 可迭代数据加载器加载计算数据
    comLoader = DataLoader(dataset=datasetCom, batch_size=1, shuffle=False, drop_last=False, num_workers=6,
                           pin_memory=True)

    # 测试.py中加载模型的部分
    print('--------------------------------------------------------------')
    # 加载模型参数并处理多GPU前缀
    state_dict = torch.load('./run/result1/model_best.pth')
    new_state_dict = OrderedDict()  # 需要导入from collections import OrderedDict
    for k, v in state_dict.items():
        # 移除键名中的'module.'前缀
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    # 加载处理后的参数
    myNet.load_state_dict(new_state_dict)
    myNet.eval()  # 指定网络模型测试状态


    with torch.no_grad():  # 测试阶段不需要梯度
        timeStart = time.time()  # 测试开始时间
        for index, x in enumerate(tqdm(testLoader, desc='Testing !!! ', file=sys.stdout), 0):
            torch.cuda.empty_cache()  # 释放显存

            # 获取原始维度（假设 x 为 (B, C, H, W)，B=1）
            orig_H, orig_W = x.shape[2], x.shape[3]

            # 计算填充，使 H 和 W 能被 8 整除（对 3 次下采样安全）
            pad_H = (8 - orig_H % 8) % 8
            pad_W = (8 - orig_W % 8) % 8

            # 使用反射模式填充输入（图像恢复常用模式，以避免伪影）
            x_padded = F.pad(x, (0, pad_W, 0, pad_H), mode='reflect')

            # 移动到 GPU 并运行推理
            input_test = x_padded.cuda()  # 放入 GPU
            output_test = myNet(input_test)  # 输入网络，得到输出

            # 裁剪回原始大小（从底部和右侧移除填充）
            output_test = output_test[..., :orig_H, :orig_W]

            # 保存结果（根据需要调整索引）
            save_image(output_test[-1], resultPathTest + str(index + 1).zfill(3) + '.png')  # 保存网络输出结果
        timeEnd = time.time()  # 测试结束时间
        print('---------------------------------------------------------')
        print("Testing Process Finished !!! Time: {:.4f} s".format(timeEnd - timeStart))


    # 计算 PSNR 和 SSIM
    print('------------------------------------------')
    print("Computing PSNR and SSIM !")
    print('------------------------------------------')
    sum1, sum2 = 0, 0
    count = 0  # 独立计数器，记录有效样本数量

    # 遍历数据集计算指标
    for batch_idx, (x, y) in enumerate(tqdm(comLoader, file=sys.stdout), 0):
        result, target = x.cuda(), y.cuda()
        sum1 += psnr(target, result).item()
        sum2 += ssim(target, result).item()
        count += 1  # 每处理一个样本，计数器+1

    # 避免除零错误，判断是否有有效样本
    if count == 0:
        print("警告：未处理任何测试样本，请检查comLoader是否有数据")
        ps = 0.0
        ss = 0.0
    else:
        ps = sum1 / count
        ss = sum2 / count

    print(f"PSNR: {ps:.4f}, SSIM: {ss:.4f}")