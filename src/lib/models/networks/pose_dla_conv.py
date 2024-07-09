from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from dcn_v2 import DCN

BN_MOMENTUM = 0.1  # 定义批量归一化的动量
logger = logging.getLogger(__name__)  # 获取日志记录器

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):  # 定义获取模型URL的函数
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))  # 返回模型的URL

def conv3x3(in_planes, out_planes, stride=1):  # 定义3x3卷积函数
    "3x3 convolution with padding"  # 3x3卷积，带有填充
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,  # 返回卷积层
                     padding=1, bias=False)  # 填充为1，无偏置项

class BasicBlock(nn.Module):  # 定义基础块类
    def __init__(self, inplanes, planes, stride=1, dilation=1):  # 初始化函数
        super(BasicBlock, self).__init__()  # 调用父类初始化函数
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,  # 定义第一个卷积层
                               stride=stride, padding=dilation,  # 设置步长和填充
                               bias=False, dilation=dilation)  # 无偏置项，设置膨胀
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)  # 定义第一个批量归一化层
        self.relu = nn.ReLU(inplace=True)  # 定义ReLU激活函数
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,  # 定义第二个卷积层
                               stride=1, padding=dilation,  # 步长为1，设置填充
                               bias=False, dilation=dilation)  # 无偏置项，设置膨胀
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)  # 定义第二个批量归一化层
        self.stride = stride  # 保存步长

    def forward(self, x, residual=None):  # 定义前向传播函数
        if residual is None:  # 如果残差为空
            residual = x  # 设置残差为输入

        out = self.conv1(x)  # 通过第一个卷积层
        out = self.bn1(out)  # 通过第一个批量归一化层
        out = self.relu(out)  # 通过ReLU激活函数

        out = self.conv2(out)  # 通过第二个卷积层
        out = self.bn2(out)  # 通过第二个批量归一化层

        out += residual  # 加上残差
        out = self.relu(out)  # 再次通过ReLU激活函数

        return out  # 返回输出


class Bottleneck(nn.Module):  # 定义瓶颈块类
    expansion = 2  # 扩展系数

    def __init__(self, inplanes, planes, stride=1, dilation=1):  # 初始化函数
        super(Bottleneck, self).__init__()  # 调用父类的初始化方法
        expansion = Bottleneck.expansion  # 获取扩展系数
        bottle_planes = planes // expansion  # 计算瓶颈层的平面数
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,  # 定义第一个卷积层
                               kernel_size=1, bias=False)  # 核大小为1，无偏置
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)  # 定义第一个批量归一化层
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,  # 定义第二个卷积层
                               stride=stride, padding=dilation,  # 设置步长和填充
                               bias=False, dilation=dilation)  # 无偏置，设置膨胀
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)  # 定义第二个批量归一化层
        self.conv3 = nn.Conv2d(bottle_planes, planes,  # 定义第三个卷积层
                               kernel_size=1, bias=False)  # 核大小为1，无偏置
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)  # 定义第三个批量归一化层
        self.relu = nn.ReLU(inplace=True)  # 定义ReLU激活函数
        self.stride = stride  # 保存步长

    def forward(self, x, residual=None):  # 定义前向传播函数
        if residual is None:  # 如果残差为空
            residual = x  # 设置残差为输入

        out = self.conv1(x)  # 通过第一个卷积层
        out = self.bn1(out)  # 通过第一个批量归一化层
        out = self.relu(out)  # 通过ReLU激活函数

        out = self.conv2(out)  # 通过第二个卷积层
        out = self.bn2(out)  # 通过第二个批量归一化层
        out = self.relu(out)  # 再次通过ReLU激活函数

        out = self.conv3(out)  # 通过第三个卷积层
        out = self.bn3(out)  # 通过第三个批量归一化层

        out += residual  # 加上残差
        out = self.relu(out)  # 再次通过ReLU激活函数

        return out  # 返回输出


class BottleneckX(nn.Module):  # 定义BottleneckX类，继承自nn.Module
    expansion = 2  # 扩展系数
    cardinality = 32  # 设置基数

    def __init__(self, inplanes, planes, stride=1, dilation=1):  # 初始化函数
        super(BottleneckX, self).__init__()  # 调用父类的初始化方法
        cardinality = BottleneckX.cardinality  # 获取基数
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))  # 计算维度，此行被注释
        # bottle_planes = dim * cardinality  # 计算瓶颈层的平面数，此行被注释
        bottle_planes = planes * cardinality // 32  # 通过基数和平面数计算瓶颈层的平面数
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,  # 定义第一个卷积层
                               kernel_size=1, bias=False)  # 核大小为1，无偏置
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)  # 定义第一个批量归一化层
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,  # 定义第二个卷积层
                               stride=stride, padding=dilation, bias=False,  # 设置步长和填充
                               dilation=dilation, groups=cardinality)  # 设置膨胀和分组
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)  # 定义第二个批量归一化层
        self.conv3 = nn.Conv2d(bottle_planes, planes,  # 定义第三个卷积层
                               kernel_size=1, bias=False)  # 核大小为1，无偏置
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)  # 定义第三个批量归一化层
        self.relu = nn.ReLU(inplace=True)  # 定义ReLU激活函数
        self.stride = stride  # 保存步长

    def forward(self, x, residual=None):  # 定义前向传播函数
        if residual is None:  # 如果残差为空
            residual = x  # 设置残差为输入

        out = self.conv1(x)  # 通过第一个卷积层
        out = self.bn1(out)  # 通过第一个批量归一化层
        out = self.relu(out)  # 通过ReLU激活函数

        out = self.conv2(out)  # 通过第二个卷积层
        out = self.bn2(out)  # 通过第二个批量归一化层
        out = self.relu(out)  # 再次通过ReLU激活函数

        out = self.conv3(out)  # 通过第三个卷积层
        out = self.bn3(out)  # 通过第三个批量归一化层

        out += residual  # 加上残差
        out = self.relu(out)  # 再次通过ReLU激活函数

        return out  # 返回输出


cclass Root(nn.Module):  # 定义Root类，继承自nn.Module
    def __init__(self, in_channels, out_channels, kernel_size, residual):  # 初始化函数
        super(Root, self).__init__()  # 调用父类的初始化方法
        self.conv = nn.Conv2d(  # 定义卷积层
            in_channels, out_channels, 1,  # 输入通道数，输出通道数，卷积核大小
            stride=1, bias=False, padding=(kernel_size - 1) // 2)  # 步长，无偏置，计算填充
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)  # 定义批量归一化层
        self.relu = nn.ReLU(inplace=True)  # 定义ReLU激活函数
        self.residual = residual  # 保存残差标志

    def forward(self, *x):  # 定义前向传播函数
        children = x  # 保存输入
        x = self.conv(torch.cat(x, 1))  # 对输入进行拼接后进行卷积操作
        x = self.bn(x)  # 通过批量归一化层
        if self.residual:  # 如果使用残差
            x += children[0]  # 加上残差
        x = self.relu(x)  # 通过ReLU激活函数

        return x  # 返回输出

class Tree(nn.Module):  # 定义Tree类，继承自nn.Module
    def __init__(self, levels, block, in_channels, out_channels, stride=1,  # 初始化函数
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()  # 调用父类的初始化方法
        if root_dim == 0:  # 如果root_dim为0
            root_dim = 2 * out_channels  # 计算root_dim
        if level_root:  # 如果是level_root
            root_dim += in_channels  # root_dim增加输入通道数
        if levels == 1:  # 如果层级为1
            self.tree1 = block(in_channels, out_channels, stride,  # 定义tree1
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,  # 定义tree2
                               dilation=dilation)
        else:  # 如果层级不为1
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,  # 定义tree1
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,  # 定义tree2
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:  # 如果层级为1
            self.root = Root(root_dim, out_channels, root_kernel_size,  # 定义root
                             root_residual)
        self.level_root = level_root  # 保存level_root标志
        self.root_dim = root_dim  # 保存root_dim
        self.downsample = None  # 初始化下采样为None
        self.project = None  # 初始化投影为None
        self.levels = levels  # 保存层级
        if stride > 1:  # 如果步长大于1
            self.downsample = nn.MaxPool2d(stride, stride=stride)  # 定义下采样层
        if in_channels != out_channels:  # 如果输入通道数不等于输出通道数
            self.project = nn.Sequential(  # 定义投影
                nn.Conv2d(in_channels, out_channels,  # 定义卷积层
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)  # 定义批量归一化层
            )

    def forward(self, x, residual=None, children=None):  # 定义前向传播函数
        children = [] if children is None else children  # 如果children为None，则初始化为空列表，否则使用传入的children
        bottom = self.downsample(x) if self.downsample else x  # 如果存在下采样，则对x进行下采样，否则直接使用x
        residual = self.project(bottom) if self.project else bottom  # 如果存在投影，则对bottom进行投影，否则直接使用bottom
        if self.level_root:  # 如果是根层级
            children.append(bottom)  # 将bottom添加到children列表中
        x1 = self.tree1(x, residual)  # 使用tree1处理x和residual，得到x1
        if self.levels == 1:  # 如果层级为1
            x2 = self.tree2(x1)  # 使用tree2处理x1，得到x2
            x = self.root(x2, x1, *children)  # 使用root处理x2，x1和children，得到最终的x
        else:  # 如果层级不为1
            children.append(x1)  # 将x1添加到children列表中
            x = self.tree2(x1, children=children)  # 使用tree2处理x1和children，得到最终的x
        return x  # 返回x


class DLA(nn.Module):  # 定义DLA类，继承自nn.Module
    def __init__(self, levels, channels, num_classes=1000,  # 初始化函数
                 block=BasicBlock, residual_root=False, linear_root=False):  # 接收层级、通道数、类别数等参数
        super(DLA, self).__init__()  # 调用父类的初始化方法
        self.channels = channels  # 初始化通道数
        self.num_classes = num_classes  # 初始化类别数
        self.base_layer = nn.Sequential(  # 定义基础层
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,  # 2D卷积层
                      padding=3, bias=False),  # 填充为3，无偏置项
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),  # 批量归一化层
            nn.ReLU(inplace=True))  # ReLU激活函数
        self.level0 = self._make_conv_level(  # 定义level0
            channels[0], channels[0], levels[0])  # 调用_make_conv_level方法
        self.level1 = self._make_conv_level(  # 定义level1
            channels[0], channels[1], levels[1], stride=2)  # 步长为2
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,  # 定义level2，使用Tree结构
                           level_root=False,  # level_root为False
                           root_residual=residual_root)  # 根据residual_root参数决定是否使用残差连接
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,  # 定义level3，使用Tree结构
                           level_root=True, root_residual=residual_root)  # level_root为True
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,  # 定义level4，使用Tree结构
                           level_root=True, root_residual=residual_root)  # level_root为True
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,  # 定义level5，使用Tree结构
                           level_root=True, root_residual=residual_root)  # level_root为True

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):  # 定义构建网络层级的函数
        downsample = None  # 初始化下采样为None
        if stride != 1 or inplanes != planes:  # 如果步长不为1或输入平面数不等于输出平面数
            downsample = nn.Sequential(  # 定义下采样模块
                nn.MaxPool2d(stride, stride=stride),  # 最大池化层
                nn.Conv2d(inplanes, planes,  # 卷积层
                          kernel_size=1, stride=1, bias=False),  # 核大小为1，步长为1，无偏置
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),  # 批量归一化
            )

        layers = []  # 初始化层列表
        layers.append(block(inplanes, planes, stride, downsample=downsample))  # 添加第一个块，可能包含下采样
        for i in range(1, blocks):  # 循环添加剩余的块
            layers.append(block(inplanes, planes))  # 添加块，不包含下采样

        return nn.Sequential(*layers)  # 返回由层组成的序列模型

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):  # 定义构建卷积层级的函数
        modules = []  # 初始化模块列表
        for i in range(convs):  # 循环添加卷积模块
            modules.extend([  # 向模块列表中添加元素
                nn.Conv2d(inplanes, planes, kernel_size=3,  # 卷积层
                          stride=stride if i == 0 else 1,  # 第一个卷积层的步长根据参数决定，之后为1
                          padding=dilation, bias=False, dilation=dilation),  # 填充和膨胀根据参数决定
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),  # 批量归一化
                nn.ReLU(inplace=True)])  # ReLU激活函数
            inplanes = planes  # 更新输入平面数为输出平面数
        return nn.Sequential(*modules)  # 返回由模块组成的序列模型

    def forward(self, x):  # 定义前向传播函数
        y = []  # 初始化输出列表
        x = self.base_layer(x)  # 通过基础层处理输入
        for i in range(6):  # 循环通过每个层级
            x = getattr(self, 'level{}'.format(i))(x)  # 获取并调用对应的层级
            y.append(x)  # 将结果添加到输出列表
        return y  # 返回输出列表

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):  # 定义加载预训练模型的函数
        # fc = self.fc  # 注释掉的代码，可能是旧代码或备忘
        if name.endswith('.pth'):  # 如果模型名称以.pth结尾
            model_weights = torch.load(data + name)  # 直接加载模型权重
        else:  # 如果不是以.pth结尾
            model_url = get_model_url(data, name, hash)  # 获取模型的URL
            model_weights = model_zoo.load_url(model_url)  # 从URL加载模型权重
        num_classes = len(model_weights[list(model_weights.keys())[-1]])  # 获取类别数
        self.fc = nn.Conv2d(  # 定义全连接层为卷积层
            self.channels[-1], num_classes,  # 输入通道数和类别数
            kernel_size=1, stride=1, padding=0, bias=True)  # 核大小为1，步长为1，无填充，有偏置
        self.load_state_dict(model_weights)  # 加载模型权重
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # 定义dla34函数，用于创建DLA-34模型
    model = DLA([1, 1, 1, 2, 2, 1],  # 模型层次结构
                [16, 32, 64, 128, 256, 512],  # 各层通道数
                block=BasicBlock, **kwargs)  # 使用BasicBlock作为基础块
    if pretrained:  # 如果需要加载预训练模型
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')  # 加载预训练模型
    return model  # 返回模型

class Identity(nn.Module):  # 定义Identity类，继承自nn.Module

    def __init__(self):  # 初始化函数
        super(Identity, self).__init__()  # 调用父类初始化函数

    def forward(self, x):  # 前向传播函数
        return x  # 直接返回输入

def fill_fc_weights(layers):  # 定义填充全连接层权重的函数
    for m in layers.modules():  # 遍历所有模块
        if isinstance(m, nn.Conv2d):  # 如果是卷积层
            if m.bias is not None:  # 如果有偏置项
                nn.init.constant_(m.bias, 0)  # 初始化偏置为0

def fill_up_weights(up):  # 定义填充上采样权重的函数
    w = up.weight.data  # 获取权重数据
    f = math.ceil(w.size(2) / 2)  # 计算f值
    c = (2 * f - 1 - f % 2) / (2. * f)  # 计算c值
    for i in range(w.size(2)):  # 遍历高度
        for j in range(w.size(3)):  # 遍历宽度
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))  # 计算权重
    for c in range(1, w.size(0)):  # 遍历通道，除了第一个
        w[c, 0, :, :] = w[0, 0, :, :]  # 复制权重

class DeformConv(nn.Module):  # 定义DeformConv类，继承自nn.Module
    def __init__(self, chi, cho):  # 初始化函数
        super(DeformConv, self).__init__()  # 调用父类初始化函数
        self.actf = nn.Sequential(  # 定义激活函数序列
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),  # 批量归一化
            nn.ReLU(inplace=True)  # ReLU激活函数
        )
        #self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)
        self.conv = nn.Conv2d(chi, cho, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)  # 定义卷积层

    def forward(self, x):  # 前向传播函数
        x = self.conv(x)  # 通过卷积层
        x = self.actf(x)  # 通过激活函数序列
        return x  # 返回结果


class IDAUp(nn.Module):  # 定义IDAUp类，继承自nn.Module

    def __init__(self, o, channels, up_f):  # 初始化函数
        super(IDAUp, self).__init__()  # 调用父类的初始化方法
        for i in range(1, len(channels)):  # 遍历channels列表，从第二个元素开始
            c = channels[i]  # 获取当前通道数
            f = int(up_f[i])  # 获取上采样因子，并转换为整数
            proj = DeformConv(c, o)  # 创建DeformConv实例，用于投影
            node = DeformConv(o, o)  # 创建DeformConv实例，用于节点处理
     
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,  # 创建转置卷积层，用于上采样
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)  # 调用fill_up_weights函数，初始化上采样权重

            setattr(self, 'proj_' + str(i), proj)  # 将proj实例设置为类的属性
            setattr(self, 'up_' + str(i), up)  # 将up实例设置为类的属性
            setattr(self, 'node_' + str(i), node)  # 将node实例设置为类的属性
                 
        
    def forward(self, layers, startp, endp):  # 前向传播函数
        for i in range(startp + 1, endp):  # 遍历层，从startp+1到endp
            upsample = getattr(self, 'up_' + str(i - startp))  # 获取上采样实例
            project = getattr(self, 'proj_' + str(i - startp))  # 获取投影实例
            layers[i] = upsample(project(layers[i]))  # 对当前层进行投影和上采样
            node = getattr(self, 'node_' + str(i - startp))  # 获取节点处理实例
            layers[i] = node(layers[i] + layers[i - 1])  # 将当前层与前一层相加后进行节点处理


class DLAUp(nn.Module):  # 定义DLAUp类，继承自nn.Module
    def __init__(self, startp, channels, scales, in_channels=None):  # 初始化函数
        super(DLAUp, self).__init__()  # 调用父类的初始化方法
        self.startp = startp  # 初始化startp属性
        if in_channels is None:  # 如果未提供in_channels参数
            in_channels = channels  # 使用channels作为in_channels
        self.channels = channels  # 初始化channels属性
        channels = list(channels)  # 将channels转换为列表
        scales = np.array(scales, dtype=int)  # 将scales转换为整数类型的numpy数组
        for i in range(len(channels) - 1):  # 遍历channels列表，除了最后一个元素
            j = -i - 2  # 计算索引j
            setattr(self, 'ida_{}'.format(i),  # 创建IDAUp实例，并设置为类的属性
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]  # 更新scales数组
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]  # 更新in_channels列表

    def forward(self, layers):  # 前向传播函数
        out = [layers[-1]]  # 以layers列表的最后一个元素作为输出列表的初始元素
        for i in range(len(layers) - self.startp - 1):  # 遍历layers列表，从startp+1到最后一个元素
            ida = getattr(self, 'ida_{}'.format(i))  # 获取IDAUp实例
            ida(layers, len(layers) -i - 2, len(layers))  # 调用IDAUp实例的forward方法
            out.insert(0, layers[-1])  # 将layers列表的最后一个元素插入到输出列表的开头
        return out  # 返回输出列表


class Interpolate(nn.Module):  # 定义Interpolate类，继承自nn.Module
    def __init__(self, scale, mode):  # 初始化函数，接收缩放比例和插值模式作为参数
        super(Interpolate, self).__init__()  # 调用父类的初始化方法
        self.scale = scale  # 保存缩放比例
        self.mode = mode  # 保存插值模式
        
    def forward(self, x):  # 前向传播函数
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)  # 对输入x进行插值操作
        return x  # 返回插值后的结果


class DLASeg(nn.Module):  # 定义DLASeg类，继承自nn.Module
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,  # 初始化函数
                 last_level, head_conv, out_channel=0):
        super(DLASeg, self).__init__()  # 调用父类的初始化方法
        assert down_ratio in [2, 4, 8, 16]  # 断言down_ratio在指定的列表中
        self.first_level = int(np.log2(down_ratio))  # 计算first_level
        self.last_level = last_level  # 设置last_level
        self.base = globals()[base_name](pretrained=pretrained)  # 通过base_name获取模型并设置pretrained参数
        channels = self.base.channels  # 获取模型的通道数
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]  # 计算scales
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)  # 创建DLAUp对象

        if out_channel == 0:  # 如果未指定out_channel
            out_channel = channels[self.first_level]  # 使用first_level的通道数作为out_channel

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],  # 创建IDAUp对象
                            [2 ** i for i in range(self.last_level - self.first_level)])
        
        self.heads = heads  # 设置heads
        for head in self.heads:  # 遍历heads
            classes = self.heads[head]  # 获取每个head的类别数
            if head_conv > 0:  # 如果指定了head_conv
              fc = nn.Sequential(  # 创建一个Sequential模型
                  nn.Conv2d(channels[self.first_level], head_conv,  # 添加卷积层
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),  # 添加ReLU激活层
                  nn.Conv2d(head_conv, classes,  # 添加卷积层
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
              if 'hm' in head:  # 如果head包含'hm'
                fc[-1].bias.data.fill_(-2.19)  # 设置偏置
              else:
                fill_fc_weights(fc)  # 调用fill_fc_weights函数
            else:
              fc = nn.Conv2d(channels[self.first_level], classes,  # 创建卷积层
                  kernel_size=final_kernel, stride=1, 
                  padding=final_kernel // 2, bias=True)
              if 'hm' in head:  # 如果head包含'hm'
                fc.bias.data.fill_(-2.19)  # 设置偏置
              else:
                fill_fc_weights(fc)  # 调用fill_fc_weights函数
            self.__setattr__(head, fc)  # 将fc设置为类的属性

    def forward(self, x):  # 定义前向传播函数
        x = self.base(x)  # 通过基础网络处理输入x
        x = self.dla_up(x)  # 通过DLA上采样模块处理x

        y = []  # 初始化列表y，用于存储中间特征图
        for i in range(self.last_level - self.first_level):  # 遍历从first_level到last_level的范围
            y.append(x[i].clone())  # 将x中的特征图克隆到y列表中
        self.ida_up(y, 0, len(y))  # 通过IDA上采样模块处理y列表中的特征图

        z = {}  # 初始化字典z，用于存储最终的预测结果
        for head in self.heads:  # 遍历所有的头网络
            z[head] = self.__getattr__(head)(y[-1])  # 对最后一个特征图应用头网络，并将结果存储到z字典中
        return [z]  # 返回包含预测结果的列表
    

def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4):  # 定义获取姿态网络的函数
  model = DLASeg('dla{}'.format(num_layers), heads,  # 创建DLASeg模型实例
                 pretrained=True,  # 指定预训练模型为True
                 down_ratio=down_ratio,  # 设置下采样比率
                 final_kernel=1,  # 设置最终卷积核大小为1
                 last_level=5,  # 设置最后一层级别为5
                 head_conv=head_conv)  # 设置头部卷积层的通道数
  return model  # 返回模型实例
