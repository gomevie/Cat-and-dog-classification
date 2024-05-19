# coding=utf-8

from train import image_size, crop_size
from model import Net

from PIL import Image
import torch
import numpy as np
from torchvision import datasets, models, transforms

net = Net(2) ## 定义模型
net.eval() ## 设置推理模式，使得dropout和batchnorm等网络层在train和val模式间切换
torch.no_grad() ## 停止autograd模块的工作，以起到加速和节省显存

## 载入模型权重
modelpath = 'models/model.pt' #sys.argv[1]
net.load_state_dict(torch.load(modelpath))

## 定义预处理函数
data_transforms =  transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(crop_size),

        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

## 读取3通道图片，并扩充为4通道tensor
imagepath = "../data/val/1/cat.4003.jpg" #sys.argv[2]
image = Image.open(imagepath)
imgblob = data_transforms(image).unsqueeze(0)

## 获得预测结果predict，得到预测的标签值label
predict = net(imgblob)
index = np.argmax(predict.detach().numpy())
## print(predict)
## print(index)

if index == 0:
    print('the predict of photo is '+str('dog'))
else:
    print('the predict of photo is '+str('cat'))