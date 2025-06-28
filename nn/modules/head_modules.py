# head_modules.py
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv, autopad
from ultralytics.nn.modules.block import DFL
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import dist2bbox, make_anchors, dist2rbox
import math
from ultralytics.nn.modules.csc_modules import CSC
from typing import List, Optional, Tuple, Union

class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

class Detect_LSCD_V3(Detect): # Inherit from original Detect class
    """
    Lightweight Shared Convolutional Detection Head (V3)
    Integrates shared CSC blocks directly, inheriting from YOLOv8's Detect.
    """

    def __init__(self, nc=80, hidc=256, ch=()): # Keep your custom hidc parameter
        # Call parent's __init__. This initializes nc, nl, reg_max, no, stride,
        # and importantly, it creates self.cv2 (box convs) and self.cv3 (cls convs) as ModuleLists.
        # It also sets up self.dfl as a ModuleList (if reg_max > 1).
        super().__init__(nc=nc, ch=ch) 

        self.conv_align = nn.ModuleList(
            nn.Sequential(
                Conv(x, hidc, 1, act=False), # Standard Conv to align channels
                nn.GroupNorm(16, hidc), # GroupNorm
                nn.SiLU() # SiLU Activation
            ) for x in ch
        )
        
        self.shared_block = nn.Sequential(
            CSC(hidc, shortcut=False, g=1, squeeze_radio=2), # First CSC block
            nn.GroupNorm(16, hidc), # GN after CSC
            nn.SiLU(), # SiLU after GN
            CSC(hidc, shortcut=False, g=1, squeeze_radio=2), # Second CSC block
            nn.GroupNorm(16, hidc), # GN after CSC
            nn.SiLU() # SiLU after GN
        )

        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1) # Shared box regression conv
        self.cv3 = nn.Conv2d(hidc, self.nc, 1) # Shared class prediction conv

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity() # Shared DFL

        self.scale = nn.ModuleList(Scale(1.0) for _ in ch) # Per-level learnable scale parameter

    def forward(self, x: List[torch.Tensor]) -> Union[List[torch.Tensor], Tuple]:
        if self.stride.sum() == 0: 
            if self.nl == 3: # Assuming standard P3, P4, P5 outputs from Neck
                self.stride = torch.tensor([8., 16., 32.], dtype=torch.float32, device=x[0].device)
            else:
                raise ValueError(
                    f"Detect_LSCD_V3: Cannot auto-set strides for {self.nl} layers. "
                    "Expected 3 (for P3, P4, P5). Please manually set self.stride or ensure framework sets it."
                )
        
        # Apply shared head logic for each feature map level
        processed_x = []
        for i in range(self.nl):
            xi = self.conv_align[i](x[i]) # Per-level 1x1 conv for initial alignment (Conv + GN + SiLU)
            xi = self.shared_block(xi) # Apply shared CSC feature extraction block
            
            # Apply shared final convs and per-level scale
            xi = torch.cat((self.scale[i](self.cv2(xi)), self.cv3(xi)), 1)
            processed_x.append(xi)

        if self.training:  # Training path: return raw processed outputs for loss calculation
            return processed_x

        # Inference path: leverage parent's _inference method for common logic
        y = self._inference(processed_x)
        return y if self.export else (y, processed_x)

   def bias_init(self):
        """Initialize Detect_LSCD_V3 biases."""
        if self.stride.sum() == 0: 
             if self.nl == 3:
                self.stride = torch.tensor([8., 16., 32.], dtype=torch.float32, device=next(self.parameters()).device)
             else:
                 # Fallback for unexpected nl, or raise error.
                 # Using the largest stride if not 3 layers, as a general fallback.
                 LOGGER.warning("Detect_LSCD_V3: Strides not set before bias_init. Using default stride for calculations.")
                 self.stride = torch.tensor([max(self.stride.tolist()) if self.nl > 0 else 16.0] * self.nl, dtype=torch.float32, device=next(self.parameters()).device) 

        self.cv2.bias.data[:] = 1.0  # box bias
        self.cv3.bias.data[: self.nc] = math.log(5 / self.nc / (640 / (self.stride[1] if self.nl > 1 else 16)) ** 2)
