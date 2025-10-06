import os
import sys
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # 进度条
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR  # 使用余弦退火调度器
import utils
import torch.nn.utils as F
from basicsr.archs.MSDeformableNAIR import MSDeformableNAIR  # 导入新的模型
from MyDataset import *
from collections import OrderedDict
from visual.tensorboard_visualizer import TensorboardVisualizer

if __name__ == '__main__':  # 只有在 main 中才能开多线程
    # 新增：权重加载控制标志符 (True: 加载权重, False: 不加载权重直接训练)
    LOAD_WEIGHTS = False  # 可根据需要修改为True
    # 新增：权重文件路径（当LOAD_WEIGHTS为True时需确保路径正确）
    WEIGHTS_PATH = "run/result3/model_best.pth"  # 请根据实际目录修改
    
    random.seed(1234)  # 随机种子
    torch.manual_seed(1234)
    EPOCH = 250  # 训练次数
    BATCH_SIZE = 8  # 每批的训练数量 (更新为8)
    LEARNING_RATE = 2e-4  # 学习率（更新为2e-4）
    loss_list = []  # 损失存储数组
    train_psnr_history = []  # 新增：训练PSNR历史记录
    train_ssim_history = []  # 新增：训练SSIM历史记录
    val_psnr_history = []    # 新增：验证PSNR历史记录
    val_ssim_history = []    # 新增：验证SSIM历史记录
    val_loss_history = []    # 新增：验证损失历史记录
    
    # 创建run目录及本次运行的子目录（用result+递增数字命名）
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run')
    os.makedirs(run_dir, exist_ok=True)  # 确保run目录存在

    # 遍历现有result前缀目录，计算最大序号
    max_num = 0
    for item in os.listdir(run_dir):
        item_path = os.path.join(run_dir, item)
        if os.path.isdir(item_path) and item.startswith('result'):
            try:
                num = int(item[6:])  # 'result'长度为6
                if num > max_num:
                    max_num = num
            except ValueError:
                continue

    new_num = max_num + 1
    current_run_dir = os.path.join(run_dir, f'result{new_num}')
    os.makedirs(current_run_dir, exist_ok=True)  # 创建本次运行的目录
    print(f"当前运行目录: {current_run_dir}")  # 打印目录路径，方便查看

    inputPathTrain = f'/home/zgf/桌面/dataset/allweather/input/'
    targetPathTrain = f'/home/zgf/桌面/dataset/allweather/gt/' 
    inputPathTest = f'/home/zgf/桌面/dataset/allweather/inputTest/' 
    resultPathTest = f'/home/zgf/桌面/dataset/allweather/resultTest/'  
    targetPathTest = f'/home/zgf/桌面/dataset/allweather/gtTest/' 
    
    best_psnr = 0
    best_epoch = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
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

    if LOAD_WEIGHTS:
        if os.path.exists(WEIGHTS_PATH):
            state_dict = torch.load(WEIGHTS_PATH, map_location=device)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            myNet.load_state_dict(new_state_dict)
            print(f"成功从 {WEIGHTS_PATH} 加载权重")
        else:
            print(f"警告：权重文件 {WEIGHTS_PATH} 不存在，将从头开始训练")
    
    total_params = sum(p.numel() for p in myNet.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params / 1e6:.2f} M')
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        myNet = nn.DataParallel(myNet)
    
    criterion_l1 = nn.L1Loss(reduction='mean').to(device)  
    psnr = utils.PSNR().to(device)
    ssim = utils.SSIM().to(device)
    
    optimizer = optim.AdamW(myNet.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCH)
    
    datasetTrain = MyTrainDataSet(inputPathTrain, targetPathTrain)
    trainLoader = DataLoader(dataset=datasetTrain, batch_size=BATCH_SIZE, shuffle=True, 
                             drop_last=False, num_workers=0, pin_memory=True)
    
    datasetValue = MyValueDataSet(inputPathTest, targetPathTest)
    valueLoader = DataLoader(dataset=datasetValue, batch_size=BATCH_SIZE, shuffle=True, 
                             drop_last=False, num_workers=0, pin_memory=True)
    
    datasetTest = MyTestDataSet(inputPathTest)
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, 
                            drop_last=False, num_workers=0, pin_memory=True)
    
    datasetCom = MyComDataSet(resultPathTest, targetPathTest)
    if len(datasetCom.resultImages) != len(datasetCom.targetImages):
        print(f"警告: 结果图片数量({len(datasetCom.resultImages)})与目标图片数量({len(datasetCom.targetImages)})不匹配!")
        min_len = min(len(datasetCom.resultImages), len(datasetCom.targetImages))
        datasetCom.resultImages = datasetCom.resultImages[:min_len]
        datasetCom.targetImages = datasetCom.targetImages[:min_len]

    comLoader = DataLoader(dataset=datasetCom, batch_size=1, shuffle=False, 
                        drop_last=False, num_workers=0, pin_memory=True)
    
    visualizer = TensorboardVisualizer(
        log_dir=os.path.join(current_run_dir, 'logs'),
        fig_save_dir=current_run_dir
    )
    print("缓存验证集图片用于可视化...")
    visualizer.cache_val_images(valueLoader, device)
    
    for epoch in range(EPOCH):
        myNet.train()
        iters = tqdm(trainLoader, file=sys.stdout)
        epochLoss = 0
        train_psnr_list = []
        train_ssim_list = []
        iter_count = 0
        
        timeStart = time.time()
        
        for index, (x, y) in enumerate(iters, 0):
            myNet.zero_grad()
            optimizer.zero_grad()
            
            input_train, target = Variable(x).to(device), Variable(y).to(device)
            outputs = myNet(input_train)

            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            
            loss = 0.0
            wsum = 0.0
            n_out = len(outputs)
            for i, output_train in enumerate(outputs):
                weight = 1.0 if n_out == 1 else (0.5 + 0.5 * (i / (n_out - 1)))
                loss += weight * criterion_l1(output_train, target)
                wsum += weight
            loss = loss / (wsum + 1e-12)
            
            loss.backward()
            F.clip_grad_norm_(myNet.parameters(), max_norm=1.0)
            optimizer.step()
            
            epochLoss += loss.item()
            iter_count += 1
            
            with torch.no_grad():
                final_output = outputs[-1]
                train_psnr = psnr(final_output, target).mean().item()
                train_ssim = ssim(final_output, target).mean().item()
                train_psnr_list.append(train_psnr)
                train_ssim_list.append(train_ssim)
            
            iters.set_description(f"Training Epoch {epoch+1}/{EPOCH}, Loss: {loss.item():.6f}")
        
        avg_train_psnr = sum(train_psnr_list) / len(train_psnr_list)
        avg_train_ssim = sum(train_ssim_list) / len(train_ssim_list)
        avg_train_loss = epochLoss / iter_count if iter_count > 0 else 0
        
        loss_list.append(avg_train_loss)
        train_psnr_history.append(avg_train_psnr)
        train_ssim_history.append(avg_train_ssim)
        
        visualizer.add_scalar('Loss/Train', avg_train_loss, epoch)
        visualizer.add_scalar('PSNR/Train', avg_train_psnr, epoch)
        visualizer.add_scalar('SSIM/Train', avg_train_ssim, epoch)

        if (epoch + 1) % 5 == 0:
            myNet.eval()
            psnr_val_rgb = []
            ssim_val_rgb = []
            val_loss_total = 0
            val_iter_count = 0

            with torch.no_grad():
                for index, (x, y) in enumerate(valueLoader, 0):
                    input_, target_value = x.to(device), y.to(device)
                    outputs_value = myNet(input_)
                    if not isinstance(outputs_value, (list, tuple)):
                        outputs_value = [outputs_value]
                    output_value = outputs_value[-1]
                    
                    val_loss = criterion_l1(output_value, target_value)
                    val_loss_total += val_loss.item()
                    val_iter_count += 1

                    for out_img, target_img in zip(output_value, target_value):
                        psnr_val_rgb.append(psnr(out_img, target_img))
                
                psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
                for index, (x, y) in enumerate(valueLoader, 0):
                    input_, target_value = x.to(device), y.to(device)
                    outputs_value = myNet(input_)
                    if not isinstance(outputs_value, (list, tuple)):
                        outputs_value = [outputs_value]
                    output_value = outputs_value[-1]
                    for out_img, target_img in zip(output_value, target_value):
                        ssim_val_rgb.append(ssim(out_img.unsqueeze(0), target_img.unsqueeze(0)))
                
                ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()
                avg_val_loss = val_loss_total / val_iter_count if val_iter_count > 0 else 0

            val_psnr_history.append(psnr_val_rgb)
            val_ssim_history.append(ssim_val_rgb)
            val_loss_history.append(avg_val_loss)

            visualizer.add_scalar('Loss/Validation', avg_val_loss, epoch)
            visualizer.add_scalar('PSNR/Validation', psnr_val_rgb, epoch)
            visualizer.add_scalar('SSIM/Validation', ssim_val_rgb, epoch)
            visualizer.visualize_val_images(myNet, epoch, device)

            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                best_model_path = os.path.join(current_run_dir, 'model_best.pth')
                torch.save(myNet.state_dict(), best_model_path)
                print(f"Saved new best model with PSNR: {best_psnr:.4f} at epoch {best_epoch} to {best_model_path}")
        
        scheduler.step(epoch)

        current_model_path = os.path.join(current_run_dir, 'model.pth')
        torch.save(myNet.state_dict(), current_model_path)
        timeEnd = time.time()
        
        print(f"Epoch {epoch+1} finished in {timeEnd-timeStart:.2f} seconds.")
        print(f"Avg loss: {avg_train_loss:.6f}, PSNR: {avg_train_psnr:.4f}, SSIM: {avg_train_ssim:.4f}")
    
    print("Training Finished! Best PSNR: {:.4f} at Epoch: {}".format(best_psnr, best_epoch))
