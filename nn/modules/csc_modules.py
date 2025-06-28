import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv

class CSC(nn.Module):
    def __init__(self, c: int, shortcut=False, g: int = 1, squeeze_radio: int = 2):
        super().__init__()
        if c % 4 != 0:
            raise ValueError(f"Input channel 'c' for CSC module must be a multiple of 4 for symmetric splits (current: {c}).")
        if c < 4:
            raise ValueError(f"Input channel 'c' must be at least 4 for CSC module (current: {c}).")

        self.up_raw_channels = c // 2
        self.low_raw_channels = c // 2

        self.up_squeezed_channels = max(1, self.up_raw_channels // squeeze_radio)
        self.low_squeezed_channels = max(1, self.low_raw_channels // squeeze_radio)

        self.squeeze1 = Conv(self.up_raw_channels, self.up_squeezed_channels, k=1, s=1, act=True)
        self.squeeze2 = Conv(self.low_raw_channels, self.low_squeezed_channels, k=1, s=1, act=True)

        self.Y1_out_channels = c 

        self.group_conv_X1 = Conv(self.up_squeezed_channels, self.Y1_out_channels, k=3, s=1, p=1, g=g, act=True)
        self.point_conv_X1 = Conv(self.up_squeezed_channels, self.Y1_out_channels, k=1, s=1, act=True)

        self.Y2_out_channels = c
        
        self.PWC2_conv_out_channels = self.Y2_out_channels - self.low_squeezed_channels
        if self.PWC2_conv_out_channels <= 0:
            raise ValueError(f"PWC2_conv_out_channels ({self.PWC2_conv_out_channels}) is not positive. Adjust squeeze_radio or input 'c'.")
        
        self.point_conv_X2 = Conv(self.low_squeezed_channels, self.PWC2_conv_out_channels, k=1, s=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1_raw, x2_raw = torch.split(x, [self.up_raw_channels, self.low_raw_channels], dim=1)
        
        X1 = self.squeeze1(x1_raw)
        X2 = self.squeeze2(x2_raw)

        Y1_gwc_res = self.group_conv_X1(X1)
        Y1_pwc_res = self.point_conv_X1(X1)
        Y1 = Y1_gwc_res + Y1_pwc_res

        Y2_pwc_res = self.point_conv_X2(X2)
        Y2 = torch.cat([Y2_pwc_res, X2], dim=1)

        if Y1.size(1) != self.Y1_out_channels or Y2.size(1) != self.Y2_out_channels:
            raise RuntimeError(f"CSC internal channel mismatch: Y1_channels={Y1.size(1)}, Y2_channels={Y2.size(1)}. Expected {self.Y1_out_channels} for both.")
        
        if Y1.size(1) % 2 != 0 or Y2.size(1) % 2 != 0:
            raise ValueError(f"Y1 ({Y1.size(1)}) and Y2 ({Y2.size(1)}) channels must be even for internal splitting in reconstruction.")

        Y11, Y12 = torch.split(Y1, Y1.size(1) // 2, dim=1)
        Y21, Y22 = torch.split(Y2, Y2.size(1) // 2, dim=1)

        a = Y11 + Y22
        b = Y12 + Y21

        output = torch.cat([a, b], dim=1)
        
        return output


class C2f_CSC(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut=False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        
        if self.c < 4 or self.c % 4 != 0:
            raise ValueError(f"Intermediate channel 'self.c' for C2f_CSC ({self.c}) must be a multiple of 4 and >= 4 for internal CSC splitting.")

        self.cv1 = Conv(c1, 2 * self.c, k=1, s=1, act=True) 

        self.cv2 = Conv((2 + n) * self.c, c2, k=1, s=1, act=True) 

        self.m = nn.ModuleList(CSC(c=self.c, g=g) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_branch = self.cv1(x)

        y = list(x_branch.split((self.c, self.c), 1))
        
        y.extend(m(y[-1]) for m in self.m)
        
        concatenated_features = torch.cat(y, 1)

        output = self.cv2(concatenated_features)

        return output