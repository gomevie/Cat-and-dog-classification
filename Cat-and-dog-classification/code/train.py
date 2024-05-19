# coding=utf-8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com
#
# or create issues
# =============================================================================
from __future__ import print_function, division

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import os
from model import ResNet, simpleconv3, Net
from tqdm import tqdm
from plot import draw_fig

image_size = 256  ##图像统一缩放大小
crop_size = 244  ##图像裁剪大小，即训练输入大小
nclass = 2  ##分类类别数
batch_size = 16
num_epochs = 50


## 训练主函数
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    print( )
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    best_val_acc = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  ## 设置为训练模式
            else:
                model.train(False)  ## 设置为验证模式

            running_loss = 0.0  ##损失变量
            running_accs = 0.0  ##精度变量
            number_batch = 0  ##
            ## 从dataloaders中获得数据
            time.sleep(0.05)
            for data in tqdm(dataloaders[phase]):
                inputs, labels = data
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()  ##清空梯度
                outputs = model(inputs)  ##前向运行
                # print(outputs)
                _, preds = torch.max(outputs.data, 1)  ##使用max()函数对输出值进行操作，得到预测值索引
                loss = criterion(outputs, labels)  ##计算损失
                # print(preds, labels)
                if phase == 'train':

                    loss.backward()  ##误差反向传播
                    optimizer.step()  ##参数更新

                running_loss += loss.data.item()
                running_accs += torch.sum(preds == labels).item()
                number_batch += 1

            ## 得到每一个epoch的平均损失与精度
            epoch_loss = running_loss / number_batch
            epoch_acc = running_accs / dataset_sizes[phase]
            print(running_accs, dataset_sizes[phase])

            ## 收集精度和损失用于可视化
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)

            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    torch.save(model.state_dict(), 'models/best.pt')

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            scheduler.step()

    draw_fig('loss', train_loss, val_loss, num_epochs)
    draw_fig('acc', train_acc, val_acc, num_epochs)

    return model


if __name__ == '__main__':



    # model = ResNet(crop_size, num_class=nclass, layers=[1,1,2,1])  ##创建模型
    # model = simpleconv3(num_class=nclass)
    model = Net(num_class=nclass)

    ## 加载模型
    try:
        model.load_state_dict(torch.load('models/model.pt'))
        model.eval()
    except:
        pass

    data_dir = '../data'  ##数据目录

    ## 模型缓存接口
    if not os.path.exists('models'):
        os.mkdir('models')

    ## 检查GPU是否可用，如果是使用GPU，否使用CPU
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("使用gpu")
        model = model.cuda()
    print(model)

    ## 创建数据预处理函数，训练预处理包括随机裁剪缩放、归一化，验证预处理包括归一化
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.Resize(image_size),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(crop_size),

            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    ## 使用torchvision的dataset ImageFolder接口读取数据
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'val']}

    ## 创建数据指针，shuffle，多进程数量
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=4) for x in ['train', 'val']}
    ## 获得数据集大小
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model.parameters(), lr=0.01)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.5)

    model = train_model(model=model,
                        criterion=criterion,
                        optimizer=optimizer_ft,
                        scheduler=exp_lr_scheduler,
                        num_epochs=num_epochs)

    torch.save(model.state_dict(), 'models/model.pt')
