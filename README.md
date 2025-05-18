# ProjectRoot

u_net.py为需要测试的模型

test.py为测试文件

预计输出：

检测到 2 块GPU，启用DataParallel

GPU0: 峰值显存 5.32GB

GPU1: 峰值显存 5.28GB

Epoch 1, Loss: 0.0387

Epoch 2, Loss: 0.0241

Epoch 3, Loss: 0.0179

实际输出：

GPU0: 累计显存 0.95GB | 峰值显存 11.12GB

Epoch 1, Loss: 1.1247

Epoch 2, Loss: 1.1172

Epoch 3, Loss: 1.1091
