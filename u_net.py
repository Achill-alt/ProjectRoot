import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from models.mha import TextEncoderWithMHA
import argparse
import requests
import time
import json
import hashlib

class DoubleConv(nn.Module):
    """
    双卷积模块
    """
    def __init__(self, in_channels, out_channels):
        """
        初始化双卷积模块

        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        """
        super().__init__()
        self.conv = nn.Sequential(
            # 替换为深度可分离卷积
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 保持第二层卷积不变
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        前向传播函数

        :param x: 输入张量
        :return: 输出张量
        """
        return self.conv(x)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialSemanticAttention(nn.Module):
    """
    空间语义注意力模块
    """
    def __init__(self, in_channels, semantic_dim, conflict_init=0.5):
        """
        初始化空间语义注意力模块

        :param in_channels: 输入通道数
        :param semantic_dim: 语义嵌入维度
        """
        super().__init__()
        self.semantic_dim = semantic_dim
        self.W_q = nn.Linear(semantic_dim, in_channels)
        self.W_k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.W_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.W_o = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conflict_adapter = nn.Sequential(
            nn.Linear(1, in_channels),
            nn.Sigmoid()
        )

        # 添加SE模块
        self.se = SEBlock(in_channels)
        
        # 修改原卷积为1x1卷积+SE
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            self.se
        )

    def forward(self, x, semantic_embedding, weights):
        """
        前向传播函数

        :param x: 输入张量
        :param semantic_embedding: 语义嵌入张量
        :param weights: 动态权重
        :return: 输出张量
        """
        b, c, h, w = x.shape
        # 确保 weights 的形状正确
        weights = torch.tensor(weights, device=semantic_embedding.device).unsqueeze(0).unsqueeze(-1)  # 增加批量维度
        # 应用动态权重
        weighted_embedding = semantic_embedding * weights
        # 计算查询
        Q = self.W_q(weighted_embedding).unsqueeze(-1).unsqueeze(-1)
        # 确保 expand 的尺寸和 Q 的维度匹配
        Q = Q.expand(b, -1, -1, h, w)

        # 计算键和值
        K = self.W_k(x)
        V = self.W_v(x)

        # 计算注意力分数
        scores = torch.sum(Q * K.unsqueeze(1), dim=2, keepdim=True) / (self.semantic_dim ** 0.5)

        attention_weights = torch.softmax(scores, dim=-1)

        # 计算注意力输出
        attn_output = attention_weights * V.unsqueeze(1)
        attn_output = attn_output.sum(dim=1)
        attn_output = self.W_o(attn_output)

        # 融合原始特征图和注意力输出
        output = x + attn_output

        # 使用带SE的卷积代替原卷积操作
        attn_output = self.conv(attn_output)

        return output

class SkipGate(nn.Module):
    """跳跃连接门控机制"""
    def __init__(self, channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//16, 1),
            nn.ReLU(),
            nn.Conv2d(channels//16, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels*2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        # 添加可学习的对齐卷积层
        self.align_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x_skip, x_up):
        # 使用可学习的对齐卷积
        x_up = self.align_conv(x_up)
        
        # 通道注意力（使用对齐后的特征）
        ca = self.channel_att(x_skip + x_up)
        # 空间注意力（输入通道数修正为2倍）
        sa = self.spatial_att(torch.cat([x_skip, x_up], dim=1))
        return x_skip * ca + x_up * sa

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512], semantic_dim=768, clip_conflict_detector=None):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 修改后的下采样部分
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            self.downs.append(SpatialSemanticAttention(feature, semantic_dim))  # 新增空间注意力
            in_channels = feature

        # 修正跳跃连接门控初始化
        self.skip_gates = nn.ModuleList()
        for feature in reversed(features):  # 改为逆序初始化以匹配上采样顺序
            self.skip_gates.append(SkipGate(feature))
        
        # 保持原有的跳跃连接注意力层
        self.skip_attentions = nn.ModuleList()
        for feature in features:
            self.skip_attentions.append(
                SpatialSemanticAttention(feature, semantic_dim)
            )

        # 瓶颈层
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 上采样部分
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))
            self.ups.append(SpatialSemanticAttention(feature, semantic_dim))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, semantic_embedding, weights):
        skip_connections = []
        
        # 下采样部分保持不变
        for i in range(0, len(self.downs), 2):
            x = self.downs[i](x)
            x = self.downs[i+1](x, semantic_embedding, weights)
            
            # 在存入跳跃连接前应用空间注意力
            skip_idx = i // 2
            attn_skip = self.skip_attentions[skip_idx](x, semantic_embedding, weights)
            skip_connections.append(attn_skip)  # 使用注意力处理后的特征
            
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # 上采样部分保持原有插值方式
        for idx in range(0, len(self.ups), 3):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 3]
            
            # 保持原有形状检查和插值逻辑
            if x.shape != skip_connection.shape:
                # 保留原有的三线性/双线性插值方式
                x = nn.functional.interpolate(
                    x,
                    size=skip_connection.shape[2:],
                    mode='trilinear' if len(skip_connection.shape) == 5 else 'bilinear',
                    align_corners=True
                )

            # 使用改进的门控机制融合特征
            gate = self.skip_gates[idx//3]
            x = gate(skip_connection, x)  # 使用门控机制代替简单拼接

            # 后续处理保持不变
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
            x = self.ups[idx + 2](x, semantic_embedding, weights)

        return self.final_conv(x)

    def _init_weights_with_conflict(self, init_value):
        # 使用冲突值初始化关键参数
        for layer in [self.W_q, self.W_k, self.W_v]:
            nn.init.normal_(layer.weight, mean=init_value, std=0.02)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UNet with Attention')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=3, help='Number of output channels')
    args = parser.parse_args()

    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 新增CLIP模型初始化
    from models.clip_alignment_v3 import MemoryOptimizedCLIP
    clip_model = MemoryOptimizedCLIP(device=device).to(device)

    # 初始化文本编码器
    text_encoder = TextEncoderWithMHA()
    text = ["dog","grass"]  # 输入文本
    result = text_encoder.encode_text(text)
    semantic_embedding = result["embeddings"].to(device)

    # 构造请求数据
    request_data = {
        "base_prompt": "A dog on the grass",
        "attributes": [
            {
                "name": "object",
                "type": "text",
                "value": text[0],
                "initial_weight": 0.8,
                "constraints": {
                    "min_weight": 0.4,
                    "max_weight": 0.9,
                }
            },
            {
                "name": "background",
                "type": "text",
                "value": text[1],
                "initial_weight": 0.2,
                "constraints": {
                    "min_weight": 0.1,
                    "max_weight": 0.5,
                }
            }
        ],
        "temperature": 1.8,
        "fallback_strategy": "creative",
    }

    # 获取时间戳
    timestamp = str(int(time.time()))
    # 检查时间戳是否在有效范围内（假设有效期为 60 秒）
    if int(time.time()) - int(timestamp) > 60:
        print("时间戳已过期，请重新获取")
        timestamp = str(int(time.time()))

    data_str = json.dumps(request_data, sort_keys=True)

    # 计算签名
    message = f"{data_str}{timestamp}"
    signature = hashlib.sha256(message.encode()).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": "api_key",
        "X-Timestamp": timestamp,
        "X-Signature": signature,
    }

    # 调用 dynamic_weights 接口
    url = 'http://127.0.0.1:5000/api/v5/weight/calculate'
    response = requests.post(url, headers=headers, json=request_data)
    if response.status_code == 200:
        response_data = response.json()
        if response_data["code"] == 200:
            final_weights = response_data["data"]["final_weights"]
            weights = [final_weights[key] for key in text]
        else:
            print(f"Failed to get dynamic weights: {response_data}")
            weights = [1.0] * len(text)
    else:
        print(f"Failed to get dynamic weights: {response.text}")
        weights = [1.0] * len(text)

    # 修改UNet初始化
    model = UNet(
        in_channels=args.in_channels, 
        out_channels=args.out_channels,
        clip_conflict_detector=clip_model.get_conflict_detector()  # 添加冲突检测器
    ).to(device)
    # 加载图片
    image_path = 'data/image3.jpg'  # 输入图像地址
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    input_image = transform(image).unsqueeze(0).to(device)

    # 模型推理
    output = model(input_image, semantic_embedding, weights)
    output = output.cpu().squeeze(0).detach().numpy()

    # 归一化到 [0, 1] 范围
    output = (output - output.min()) / (output.max() - output.min() + 1e-8)

    # 调整维度以符合 imshow 要求 (H, W, C)
    if args.out_channels == 1:
        output = output.squeeze()
    else:
        output = output.transpose(1, 2, 0)

    # 显示和保存结果
    plt.imshow(output)
    plt.axis('off')
    plt.savefig('output_image.jpg', bbox_inches='tight', pad_inches=0)
    plt.show()

    # 清空缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
