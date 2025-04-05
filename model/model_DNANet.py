import torch
import torch.nn as nn


class VGG_CBAM_Block(nn.Module):
    """VGG风格的卷积块，集成了CBAM注意力机制
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 第一个卷积层+批归一化+激活
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层+批归一化+激活
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 通道注意力和空间注意力模块
        self.ca = ChannelAttention(out_channels)  # 通道注意力
        self.sa = SpatialAttention()  # 空间注意力

    def forward(self, x):
        # 第一个卷积块处理
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 第二个卷积块处理
        out = self.conv2(out)
        out = self.bn2(out)
        # 应用通道注意力
        out = self.ca(out) * out
        # 应用空间注意力
        out = self.sa(out) * out
        # 最终激活
        out = self.relu(out)
        return out


class ChannelAttention(nn.Module):
    """通道注意力模块 - 捕获通道间的依赖关系
    
    参数:
        in_planes: 输入特征通道数
        ratio: 降维比率，用于减少参数量
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
        # MLP层 - 使用1x1卷积实现全连接效果
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)  # 降维
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)  # 升维
        self.sigmoid = nn.Sigmoid()  # 归一化为0-1之间的权重

    def forward(self, x):
        # 通过平均池化分支
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 通过最大池化分支
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 合并两个分支
        out = avg_out + max_out
        # 通过Sigmoid归一化
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块 - 捕获空间位置的重要性
    
    参数:
        kernel_size: 卷积核大小，用于聚合空间信息
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 确保kernel_size为3或7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # 使用卷积层聚合空间特征
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度计算平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通道平均
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通道最大
        # 拼接平均值和最大值特征
        x = torch.cat([avg_out, max_out], dim=1)
        # 通过卷积层学习空间注意力权重
        x = self.conv1(x)
        # 归一化为0-1之间的注意力图
        return self.sigmoid(x)


class Res_CBAM_block(nn.Module):
    """结合残差连接和CBAM注意力机制的卷积块
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        stride: 卷积步长，控制特征图尺寸
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_CBAM_block, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 残差连接 - 如果尺寸不匹配，使用1x1卷积调整
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        # 注意力模块
        self.ca = ChannelAttention(out_channels)  # 通道注意力
        self.sa = SpatialAttention()  # 空间注意力

    def forward(self, x):
        # 保存输入用于残差连接
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
            
        # 前向传播通过主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 应用注意力机制
        out = self.ca(out) * out  # 通道注意力
        out = self.sa(out) * out  # 空间注意力
        
        # 添加残差连接并激活
        out += residual
        out = self.relu(out)
        return out


class DNANet(nn.Module):
    """密集嵌套注意力网络 (Dense Nested Attention Network)
    
    专为小目标检测设计的多尺度特征提取和融合网络，
    通过密集连接、嵌套结构和注意力机制增强特征表示
    
    参数:
        num_classes: 输出类别数，二分类时为1
        input_channels: 输入图像通道数，通常为3(RGB)
        block: 使用的基本块类型，如Res_CBAM_block
        num_blocks: 每个尺度层次中的块数量
        nb_filter: 每个尺度层次的滤波器数量
        deep_supervision: 是否使用深度监督机制
    """
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter, deep_supervision=False):
        super(DNANet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.deep_supervision = deep_supervision
        
        # 下采样和上采样操作
        self.pool = nn.MaxPool2d(2, 2)  # 2倍下采样
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 2倍上采样
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)  # 0.5倍下采样
        
        # 多尺度上采样模块
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 4倍上采样
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)  # 8倍上采样
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)  # 16倍上采样

        # 编码器部分 - 下采样路径
        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2], nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3], nb_filter[4], num_blocks[3])

        # 第一级嵌套解码器连接
        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2], nb_filter[3], num_blocks[2])

        # 第二级嵌套解码器连接
        self.conv0_2 = self._make_layer(block, nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = self._make_layer(block, nb_filter[1]*2 + nb_filter[2] + nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_2 = self._make_layer(block, nb_filter[2]*2 + nb_filter[3] + nb_filter[1], nb_filter[2], num_blocks[1])

        # 第三级嵌套解码器连接
        self.conv0_3 = self._make_layer(block, nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1]*3 + nb_filter[2] + nb_filter[0], nb_filter[1], num_blocks[0])

        # 第四级嵌套解码器连接
        self.conv0_4 = self._make_layer(block, nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        # 多尺度特征融合
        self.conv0_4_final = self._make_layer(block, nb_filter[0]*5, nb_filter[0])

        # 多尺度特征转换
        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        # 输出层 - 带或不带深度监督
        if self.deep_supervision:
            # 多尺度深度监督输出
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            # 单一输出
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        """创建包含多个块的层
        
        参数:
            block: 基本块类型
            input_channels: 输入通道数
            output_channels: 输出通道数
            num_blocks: 块的数量
        """
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        """前向传播函数
        
        实现了密集嵌套的特征提取和多路径特征融合
        """
        # 编码器路径
        x0_0 = self.conv0_0(input)  # 第一级特征
        x1_0 = self.conv1_0(self.pool(x0_0))  # 第二级特征
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))  # 第一级细化特征

        x2_0 = self.conv2_0(self.pool(x1_0))  # 第三级特征
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0), self.down(x0_1)], 1))  # 第二级细化特征
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))  # 第一级再细化特征

        x3_0 = self.conv3_0(self.pool(x2_0))  # 第四级特征
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0), self.down(x1_1)], 1))  # 第三级细化特征
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1), self.down(x0_2)], 1))  # 第二级再细化特征
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))  # 第一级三次细化特征

        x4_0 = self.conv4_0(self.pool(x3_0))  # 第五级特征(最深层特征)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0), self.down(x2_1)], 1))  # 第四级细化特征
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1), self.down(x1_2)], 1))  # 第三级再细化特征
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2), self.down(x0_3)], 1))  # 第二级三次细化特征
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))  # 第一级四次细化特征

        # 多尺度特征融合 - 结合不同层次的特征
        Final_x0_4 = self.conv0_4_final(
            torch.cat([
                self.up_16(self.conv0_4_1x1(x4_0)),  # 最深层特征(16倍上采样)
                self.up_8(self.conv0_3_1x1(x3_1)),   # 第四级特征(8倍上采样)
                self.up_4(self.conv0_2_1x1(x2_2)),   # 第三级特征(4倍上采样)
                self.up(self.conv0_1_1x1(x1_3)),     # 第二级特征(2倍上采样)
                x0_4                                  # 第一级特征(原始尺寸)
            ], 1)
        )

        # 输出处理
        if self.deep_supervision:
            # 深度监督 - 返回多个尺度的预测结果
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(Final_x0_4)
            return [output1, output2, output3, output4]
        else:
            # 标准输出 - 只返回最终预测
            output = self.final(Final_x0_4)
            return output


