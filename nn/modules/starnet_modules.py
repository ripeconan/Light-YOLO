# starnet_modules.py

import torch
import torch.nn as nn
from .conv import Conv

class SEBlock(nn.Module):
    # Squeeze-and-Excitation Block to enhance channel-wise features.
    def __init__(self, c1, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c1, c1 // r, bias=False), nn.ReLU(inplace=True),
            nn.Linear(c1 // r, c1, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SE_StarBlock(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels are based on the OUTPUT channels `c2`
        
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = Conv(2 * c_, c2, 1, 1)
        
        self.main_conv = Conv(c_, c_, 3, 1, g=g)
        self.cheap_conv = Conv(c_, c_, 3, 1, g=c_) # Depth-wise conv

        self.se = SEBlock(c2) # SE block operates on the output channels c2
        
        # The residual connection is only possible if shortcut is True AND dimensions match.
        self.add = shortcut and c1 == c2

    def forward(self, x):
        x_hidden = self.cv1(x)
        x1, x2 = x_hidden.chunk(2, 1)
        
        fused_features = torch.cat((self.main_conv(x1), self.cheap_conv(x2)), 1)
        out = self.cv2(fused_features)
        
        se_out = self.se(out)
        
        return x + se_out if self.add else se_out
        
