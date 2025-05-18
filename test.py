import os
import torch
import torch.nn as nn
from torch.nn import DataParallel
from collections import defaultdict

# 环境配置
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
torch.backends.cudnn.benchmark = True

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel//reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction, channel, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)

class SpatialSemanticAttention(nn.Module):
    def __init__(self, in_channels, semantic_dim):
        super().__init__()
        self.W_q = nn.Linear(semantic_dim, in_channels)
        self.W_k = nn.Conv2d(in_channels, in_channels, 1)
        self.W_v = nn.Conv2d(in_channels, in_channels, 1)
        self.W_o = nn.Conv2d(in_channels, in_channels, 1)
        self.se = SEBlock(in_channels)

    def forward(self, x, semantic_embedding, weights):
        b, c, h, w = x.shape
        device = x.device
        
        # 权重维度校正（严格对齐第二个文件）
        weights = torch.tensor(weights, device=device).view(1, -1)  # [1, num_attr]
        semantic_embedding = semantic_embedding.view(b, -1)  # [b, semantic_dim]
        
        # 广播机制优化 
        expanded_weights = weights.expand(b, -1)  # [b, num_attr]
        weighted_embedding = semantic_embedding * expanded_weights  # [b, semantic_dim]
        
        # 查询向量计算
        Q = self.W_q(weighted_embedding)  # [b, in_channels]
        Q = Q.view(b, c, 1, 1).expand(-1, -1, h, w)  # [b, c, h, w]
        
        # 键和值向量
        K = self.W_k(x)  # [b, c, h, w]
        V = self.W_v(x)  # [b, c, h, w]
        
        # 注意力计算优化 
        attn = torch.softmax((Q * K).sum(dim=1, keepdim=True), dim=-1)  # [b, 1, h, w]
        output = self.W_o(attn * V) * self.se(x)
        return x + output

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512], semantic_dim=768):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        # 下采样模块构建
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            self.downs.append(SpatialSemanticAttention(feature, semantic_dim))
            in_channels = feature
            
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # 上采样模块构建
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, 2, 2))
            self.ups.append(DoubleConv(feature*2, feature))
            self.ups.append(SpatialSemanticAttention(feature, semantic_dim))
            
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x, semantic_embedding, weights):
        skip_connections = []
        
        # 编码器
        for idx in range(0, len(self.downs), 2):
            x = self.downs[idx](x)
            x = self.downs[idx+1](x, semantic_embedding, weights)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        
        # 解码器
        for idx in range(0, len(self.ups), 3):
            x = self.ups[idx](x)
            skip = skip_connections.pop()
            
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx+1](x)
            x = self.ups[idx+2](x, semantic_embedding, weights)
            
        return self.final_conv(x)

class MemoryMonitor:
    def __init__(self):
        self.device_mem = defaultdict(list)
        self.hooks = []

    def __enter__(self):
        def hook(module, inp, out):
            dev_id = torch.cuda.current_device()
            mem = out.element_size() * out.nelement() / 1024**3
            self.device_mem[dev_id].append(mem)
        
        for name, layer in self.model.module.named_modules():
            if isinstance(layer, SpatialSemanticAttention):
                self.hooks.append(layer.register_forward_hook(hook))
        return self

    def __exit__(self, *args):
        [h.remove() for h in self.hooks]
        for dev_id in range(torch.cuda.device_count()):
            total = sum(self.device_mem.get(dev_id, []))
            peak = torch.cuda.max_memory_allocated(dev_id) / 1024**3
            print(f"GPU{dev_id}: 累计显存 {total:.2f}GB | 峰值显存 {peak:.2f}GB")
        torch.cuda.empty_cache()

def validate_tensor_memory(model, input_shape=(8,3,256,256)):
    monitor = MemoryMonitor()
    model = DataParallel(model.cuda(), device_ids=range(torch.cuda.device_count()))
    monitor.model = model
    
    with torch.no_grad(), monitor:
        dummy = torch.randn(input_shape).cuda()
        semantic = torch.randn(input_shape[0], 768).cuda()
        _ = model(dummy, semantic, [0.5]*768)  # 使用完整权重维度

def main():
    model = UNet().cuda()
    
    # 多GPU配置
    if torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 块GPU，启用DataParallel")
        model = DataParallel(model)
    
    # 内存验证
    validate_tensor_memory(model)
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(3):
        inputs = torch.randn(8,3,256,256).cuda()
        semantic = torch.randn(8,768).cuda()
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs, semantic, [0.5]*768)
            loss = nn.MSELoss()(outputs, torch.randn_like(outputs))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()


