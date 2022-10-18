<font size = 5>

# 一个简单的神经网络  
```python
import torch
from torch import nn
# 确定随机数种子
torch.manual_seed(7)
#自定义数据集
x = torch.rand((7, 2, 2))
#定义7张图片，每张2x2像素
target = torch.randint(0, 2, (7,))

#自定义网络结构
class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()

        #定义一层全连接层
        self.dense = nn.Linear(4, 3)
        #定义Softmax
        self.softmax = nn.Softmax(dim=1)

    #定义前向传播
    def forward(self, x):
        y = self.dense(x.view((-1, 4)))
        y = self.softmax(y)
        return y

net = LinearNet()

#定义损失函数
loss = nn.CrossEntropyLoss() #交叉熵损失函数
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

#开始训练
for epoch in range(70):
    train_l = 0.0
    y_hat = net(X)
    l = loss(y_hat, target).sum()

    #梯度清零
    optimizer.zero_grad()
    #自动求导梯度
    l.backward()
    #利用优化函数调整所有权重参数
    optimizer.step()

    train_l += l
    print('epoch %d, loss %.4f' % (epoch + 1, train_l))    
```
