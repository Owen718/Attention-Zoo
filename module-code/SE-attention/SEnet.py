from torch import nn
import torch

'''
SEnet attention module
注意力模块

'''

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #自适应平均池化,给定输出的feature map的size x，自动计算kernel和stride
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) #平均池化后的x，resize到 (b,c)大小
        y = self.fc(y).view(b, c, 1, 1) #两个全连接层后，resize到 (b,c,1,1)大小
        return x * y.expand_as(x) #将tensor y expand 到x的size



if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    feats = torch.randn((1, 64, 120, 120)).cuda()
    mem = torch.cuda.max_memory_allocated()
    base_infer = SELayer(
        channel=64,reduction=16
    )
    base_infer.eval()
    base_infer.cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    avg_time = 0
    avg_mem  = 0
    import time
    with torch.no_grad():
        for i in range(110):
            start_time = time.time()
            outputs = base_infer(feats)
            torch.cuda.synchronize()
            if i >= 10:
                avg_time += (time.time() - start_time)
                avg_mem += (torch.cuda.max_memory_allocated() - mem)

    print("Average Parameters : {}".format(count_parameters(base_infer)))
    print("Average Running Time: {}".format(avg_time/100))
    print("Average GPU Memory: {}".format(avg_mem/100))
