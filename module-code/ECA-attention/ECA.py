import torch
from torch import nn
from torch.nn.parameter import Parameter

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)



if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    feats = torch.randn((1, 64, 120, 120)).cuda()
    mem = torch.cuda.max_memory_allocated()
    baseoc_infer = eca_layer(channel=64)
    baseoc_infer.eval()
    baseoc_infer.cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    avg_time = 0
    avg_mem  = 0
    import time
    with torch.no_grad():
        for i in range(110):
            start_time = time.time()
            outputs = baseoc_infer(feats)
            torch.cuda.synchronize()
            if i >= 10:
                avg_time += (time.time() - start_time)
                avg_mem += (torch.cuda.max_memory_allocated() - mem)

    print("Average Parameters : {}".format(count_parameters(baseoc_infer)))
    print("Average Running Time: {}".format(avg_time/100))
    print("Average GPU Memory: {}".format(avg_mem/100))
