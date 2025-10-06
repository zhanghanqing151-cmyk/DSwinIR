import os
import random
import torch
import torchvision.transforms.functional as ttf
from torch.utils.data import Dataset
from PIL import Image


class MyTrainDataSet(Dataset):  # 训练数据集
    def __init__(self, inputPathTrain, targetPathTrain, patch_size=128):
        super(MyTrainDataSet, self).__init__()

        self.inputPath = inputPathTrain
        self.inputImages = os.listdir(inputPathTrain)  # 输入图片路径下的所有文件名列表

        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)  # 目标图片路径下的所有文件名列表

        self.ps = patch_size

    def __len__(self):
        return len(self.targetImages)

    def __getitem__(self, index):
        ps = self.ps
        index = index % len(self.targetImages)

        # 读取图像
        inputImage = Image.open(os.path.join(self.inputPath, self.inputImages[index])).convert('RGB')
        targetImage = Image.open(os.path.join(self.targetPath, self.targetImages[index])).convert('RGB')

        # 统一调整为正方形（长边缩放）
        w, h = inputImage.size
        new_size = max(w, h)  # 直接计算new_size为宽高中的最大值
        if w != h:
            inputImage = inputImage.resize((new_size, new_size), Image.BILINEAR)
            targetImage = targetImage.resize((new_size, new_size), Image.BILINEAR)

        # 如果尺寸仍小于patch_size，按比例放大
        if new_size < ps:
            inputImage = inputImage.resize((ps, ps), Image.BILINEAR)
            targetImage = targetImage.resize((ps, ps), Image.BILINEAR)

        # 转换为张量后随机裁剪
        input_tensor = ttf.to_tensor(inputImage)
        target_tensor = ttf.to_tensor(targetImage)
        
        # 随机裁剪
        hh, ww = target_tensor.shape[1], target_tensor.shape[2]
        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        
        return input_tensor[:, rr:rr+ps, cc:cc+ps], target_tensor[:, rr:rr+ps, cc:cc+ps]


class MyValueDataSet(Dataset):  # 评估数据集
    def __init__(self, inputPathTrain, targetPathTrain, patch_size=128):
        super(MyValueDataSet, self).__init__()

        self.inputPath = inputPathTrain
        self.inputImages = os.listdir(inputPathTrain)  # 输入图片路径下的所有文件名列表

        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)  # 目标图片路径下的所有文件名列表

        self.ps = patch_size

    def __len__(self):
        return len(self.targetImages)

    def __getitem__(self, index):

        ps = self.ps
        index = index % len(self.targetImages)

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])  # 图片完整路径
        inputImage = Image.open(inputImagePath).convert('RGB')  # 读取图片,灰度图

        targetImagePath = os.path.join(self.targetPath, self.targetImages[index])
        targetImage = Image.open(targetImagePath).convert('RGB')

        # 中心裁剪
        inputImage = ttf.center_crop(inputImage, (ps, ps))
        targetImage = ttf.center_crop(targetImage, (ps, ps))

        input_ = ttf.to_tensor(inputImage)  # 将图片转为张量
        target = ttf.to_tensor(targetImage)

        return input_, target


class MyTestDataSet(Dataset):  # 测试数据集
    def __init__(self, inputPathTest):
        super(MyTestDataSet, self).__init__()

        self.inputPath = inputPathTest
        self.inputImages = sorted(os.listdir(inputPathTest))  # ✅ 添加排序

    def __len__(self):
        return len(self.inputImages)

    def __getitem__(self, index):
        index = index % len(self.inputImages)
        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])
        inputImage = Image.open(inputImagePath).convert('RGB')
        input_ = ttf.to_tensor(inputImage)
        return input_


class MyComDataSet(Dataset):  # 计算指标数据集
    def __init__(self, resultPath, targetPath):
        self.resultPath = resultPath
        self.targetPath = targetPath
        # 获取所有图片文件名
        self.resultImages = [f for f in os.listdir(resultPath) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.targetImages = [f for f in os.listdir(targetPath) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # 排序确保对应关系
        self.resultImages.sort()
        self.targetImages.sort()
        
        # 添加调试信息
        print(f"加载到的结果图片数量: {len(self.resultImages)}")
        print(f"加载到的目标图片数量: {len(self.targetImages)}")
        if len(self.resultImages) == 0:
            print(f"警告：在{resultPath}中未找到图片")
        if len(self.targetImages) == 0:
            print(f"警告：在{targetPath}中未找到图片")

    def __len__(self):
        return len(self.targetImages)

    def __getitem__(self, index):

        resultImagePath = os.path.join(self.resultPath, self.resultImages[index])  # 图片完整路径
        resultImage = Image.open(resultImagePath).convert('L')  # 读取图片

        targetImagePath = os.path.join(self.targetPath, self.targetImages[index])
        targetImage = Image.open(targetImagePath).convert('L')

        # resultImage = resultImage.resize(targetImage.resize, Image.BILINEAR)

        result = ttf.to_tensor(resultImage)  # 将图片转为张量
        target = ttf.to_tensor(targetImage)

        return result, target
