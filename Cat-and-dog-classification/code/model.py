# coding=utf-8
import torch
from torch import nn
import torch.nn.functional as F
class Bottleneck(nn.Module):
    #每个stage维度中扩展的倍数
    extention=4
    def __init__(self,inplanes,planes,stride,downsample=None):
        '''

        :param inplanes: 输入block的之前的通道数
        :param planes: 在block中间处理的时候的通道数
                planes*self.extention:输出的维度
        :param stride:
        :param downsample:
        '''
        super(Bottleneck, self).__init__()

        self.conv1=nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride,bias=False)
        self.bn1=nn.BatchNorm2d(planes)

        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)

        self.conv3=nn.Conv2d(planes,planes*self.extention,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm2d(planes*self.extention)

        self.relu=nn.ReLU(inplace=True)

        #判断残差有没有卷积
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        #参差数据
        residual=x

        #卷积操作
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)
        out=self.relu(out)

        #是否直连（如果Indentity blobk就是直连；如果Conv2 Block就需要对残差边就行卷积，改变通道数和size
        if self.downsample is not None:
            residual=self.downsample(x)

        #将残差部分和卷积部分相加
        out = torch.add(out, residual)
        out=self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,input_size, num_class, layers=[3,4,6,3], block=Bottleneck):
        #inplane=当前的fm的通道数
        self.inplane=48
        super(ResNet, self).__init__()

        #参数
        self.block=block
        self.layers=layers

        #stem的网络层
        self.conv1=nn.Conv2d(3,self.inplane,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(self.inplane)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        #64,128,256,512指的是扩大4倍之前的维度，即Identity Block中间的维度
        self.stage1=self.make_layer(self.block,64,layers[0],stride=1)
        self.stage2=self.make_layer(self.block,128,layers[1],stride=2)
        self.stage3=self.make_layer(self.block,256,layers[2],stride=2)
        self.stage4=self.make_layer(self.block,512,layers[3],stride=2)

        #后续的网络
        self.avgpool=nn.AvgPool2d(7)
        self.fc1=nn.Linear(2048*block.extention,2048)
        self.fc2=nn.Linear(2048,num_class)
        self.softmax=nn.Softmax(dim=1)

    def forward(self,x):
        #stem部分：conv+bn+maxpool
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.maxpool(out)

        #block部分
        out=self.stage1(out)
        out=self.stage2(out)
        out=self.stage3(out)
        out=self.stage4(out)

        #分类
        out=self.avgpool(out)
        out=torch.flatten(out,1)
        out=self.fc1(out)
        out=torch.sigmoid(out)
        out=self.fc2(out)

        return out

    def make_layer(self,block,plane,block_num,stride=1):
        '''
        :param block: block模板
        :param plane: 每个模块中间运算的维度，一般等于输出维度/4
        :param block_num: 重复次数
        :param stride: 步长
        :return:
        '''
        block_list=[]
        #先计算要不要加downsample
        downsample=None
        if(stride!=1 or self.inplane!=plane*block.extention):
            downsample=nn.Sequential(
                nn.Conv2d(self.inplane,plane*block.extention,stride=stride,kernel_size=1,bias=False),
                nn.BatchNorm2d(plane*block.extention)
            )

        # Conv Block输入和输出的维度（通道数和size）是不一样的，所以不能连续串联，他的作用是改变网络的维度
        # Identity Block 输入维度和输出（通道数和size）相同，可以直接串联，用于加深网络
        #Conv_block
        conv_block=block(self.inplane,plane,stride=stride,downsample=downsample)
        block_list.append(conv_block)
        self.inplane=plane*block.extention

        #Identity Block
        for i in range(1,block_num):
            block_list.append(block(self.inplane,plane,stride=1))

        return nn.Sequential(*block_list)


class simpleconv3(nn.Module):
    ## 初始化函数
    def __init__(self, num_class):
        super(simpleconv3,self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3, 2)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, 3, 2)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 48, 3, 2)

        self.bn3 = nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(48, 24, 3, 2)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(24, 12, 3, 2)
        self.bn5 = nn.BatchNorm2d(12)
        self.conv6 = nn.Conv2d(12, 4, 3, 1)
        self.bn6 = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(64 , 12) #输入向量长为1200，输出向量长为128
        self.fc2 = nn.Linear(12 , num_class) #输入向量长为128，输出向量长为nclass，等于类别数

    ## 前向函数
    def forward(self, x):
        ## relu函数，不需要进行实例化，直接进行调用
        ## conv，fc层需要调用nn.Module进行实例化
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Conv(nn.Module):
    def __init__(self, input, output, k=3, stride=2, pad=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(input, output, k, stride, pad)
        self.bn = nn.BatchNorm2d(output)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Net(nn.Module):
    def __init__(self, num_class):
        super(Net, self).__init__()
        self.conv1 = Conv(3, 12)
        self.conv2 = Conv(12, 24)
        self.conv3 = nn.Sequential(
            Conv(24, 24, 3, 1, 1),
            Conv(24, 24, 3, 1, 1),
            Conv(24, 24, 3, 1, 1),
            Conv(24, 24, 3, 1, 1)
        )
        self.conv4 = Conv(24, 48)
        self.conv5 = nn.Sequential(
            Conv(48, 48, 3, 1, 1),
            Conv(48, 48, 3, 1, 1),
            Conv(48, 48, 3, 1, 1),
            Conv(48, 48, 3, 1, 1)
        )
        self.conv6 = Conv(48, 24)
        self.conv7 = self.conv3 = nn.Sequential(
            Conv(24, 24, 3, 1, 1),
            Conv(24, 24, 3, 1, 1),
            Conv(24, 24, 3, 1, 1),
            Conv(24, 24, 3, 1, 1)
        )
        self.conv8 = Conv(24, 12)
        self.conv9 = nn.Sequential(
            Conv(12, 12, 3, 1, 1),
            Conv(12, 12, 3, 1, 1),
            Conv(12, 12, 3, 1, 1)
        )
        self.conv10 = Conv(12, 6)
        self.fn1 = nn.Linear(96, 12)
        self.fn2 = nn.Linear(12, num_class)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.conv2(self.conv1(x))
        y2 = self.conv4(self.conv3(y1) + y1)
        y3 = self.conv6(self.conv5(y2) + y2)
        y4 = self.conv8(self.conv7(y3) + y3)
        y4 = self.conv10(self.conv9(y4) + y4)

        y4 = torch.flatten(y4, 1)

        output = self.fn2(self.sigmoid(self.fn1(y4)))

        return output


if __name__ == "__main__":
    # resnet=ResNet(448, 2, layers=[1,1,2,1])
    # x=torch.randn(1,3,448,448)
    # X=resnet(x)
    # print(X.shape)
    # print(resnet)
    # print(X)

    # x = torch.randn(1, 3, 244, 244)
    # model = simpleconv3(2)
    # y = model(x)
    # print(model)
    # print(y.shape)

    x = torch.randn(255, 3, 244, 244)
    model = Net(2)
    y = model(x)
    print(model)
    print(y.shape)