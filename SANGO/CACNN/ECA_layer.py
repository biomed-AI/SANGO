from torch import nn
import math

class eca_layer(nn.Module):
    """Constructs a ECA module. from https://arxiv.org/abs/1910.03151
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, gamma=2, b=1):
        """ A method for adaptively adjusting the size of convolutional kernels
            according to gamma and b
        
        """
        super(eca_layer, self).__init__()

        t = int(abs((math.log(channel, 2) + b) / gamma))
        k_size = t if t % 2 else t+1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        y = self.sigmoid(y)

        return x * y.expand_as(x)
