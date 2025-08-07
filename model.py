import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# 2D Haar DWT
# --------------------------
class HaarDWT(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        kernel = torch.tensor([
            [[0.5, 0.5], [0.5, 0.5]],     # LL
            [[0.5, -0.5], [0.5, -0.5]],   # LH
            [[0.5, 0.5], [-0.5, -0.5]],   # HL
            [[0.5, -0.5], [-0.5, 0.5]]    # HH
        ], dtype=torch.float32).unsqueeze(1)
        self.register_buffer('filters', kernel.repeat(in_channels, 1, 1, 1))

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        return F.conv2d(x, self.filters, stride=2, groups=self.in_channels)


# --------------------------
# Inverse Haar DWT
# --------------------------
class HaarIDWT(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        kernel = torch.tensor([
            [[0.5, 0.5], [0.5, 0.5]],     # LL
            [[0.5, -0.5], [0.5, -0.5]],   # LH
            [[0.5, 0.5], [-0.5, -0.5]],   # HL
            [[0.5, -0.5], [-0.5, 0.5]]    # HH
        ], dtype=torch.float32).unsqueeze(1)
        self.register_buffer('inv_filters', kernel.repeat(out_channels, 1, 1, 1))

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        return F.conv_transpose2d(x, self.inv_filters, stride=2, groups=self.out_channels)


# --------------------------
# Flatness Mask (LL2-based)
# --------------------------
def compute_flatness_mask(ll2):
    mean = F.avg_pool2d(ll2, kernel_size=3, stride=1, padding=1)
    variance = (ll2 - mean).pow(2)
    flatness = 1 - torch.tanh(variance.mean(dim=1, keepdim=True) * 10)
    return flatness


# --------------------------
# Edge Attention Module
# --------------------------
class EdgeAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels//2, 1, 3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.sigmoid(self.conv2(x))


# --------------------------
# DoubleConv Block
# --------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# --------------------------
# Gated UNet for HF denoising
# --------------------------
class GatedUNet(nn.Module):
    def __init__(self, in_ch=18, base_ch=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_ch * 2, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, flatness_mask):
        # Apply per-band flatness gating
        x = x * flatness_mask
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


# --------------------------
# Improved DWT2 Denoising Model
# --------------------------
class DWT2DenoisingModel(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels

        # Wavelet transforms
        self.dwt1 = HaarDWT(in_channels)
        self.dwt2 = HaarDWT(in_channels)
        self.idwt2 = HaarIDWT(in_channels)
        self.idwt1 = HaarIDWT(in_channels)

        # Consistent downsampling
        self.down_LH1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
        )
        self.down_HL1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
        )
        self.down_HH1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
        )
        
        # Edge attention
        self.edge_attention = EdgeAttention(in_channels)
        
        self.unet = GatedUNet(in_ch=6 * in_channels, base_ch=32)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Pad input to be divisible by 4 for clean DWT
        pad_h = (4 - H % 4) % 4
        pad_w = (4 - W % 4) % 4
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        # Compute edge attention from input
        edge_attn = self.edge_attention(x_pad)
        
        # DWT Level-1
        dwt1_out = self.dwt1(x_pad)
        LL1, LH1, HL1, HH1 = torch.chunk(dwt1_out, 4, dim=1)

        # DWT Level-2
        dwt2_out = self.dwt2(LL1)
        LL2, LH2, HL2, HH2 = torch.chunk(dwt2_out, 4, dim=1)

        # Combined flatness and edge mask
        flatness_mask = compute_flatness_mask(LL2)
        combined_mask = flatness_mask * edge_attn

        # Downsample Level-1 HF bands
        LH1_ds = self.down_LH1(LH1)
        HL1_ds = self.down_HL1(HL1)
        HH1_ds = self.down_HH1(HH1)

        # UNet input with masking
        unet_input = torch.cat([
            LH2 * combined_mask,
            HL2 * combined_mask,
            HH2 * combined_mask,
            LH1_ds * combined_mask,
            HL1_ds * combined_mask,
            HH1_ds * combined_mask
        ], dim=1)

        denoised = self.unet(unet_input, combined_mask)
        den_LH2, den_HL2, den_HH2, den_LH1, den_HL1, den_HH1 = torch.chunk(denoised, 6, dim=1)

        # Inverse DWT Level-2
        idwt2_input = torch.cat([LL2, den_LH2, den_HL2, den_HH2], dim=1)
        LL1_recon = self.idwt2(idwt2_input)

        # Upsample with bilinear interpolation
        den_LH1_up = F.interpolate(den_LH1, size=LL1_recon.shape[-2:], mode='bilinear', align_corners=False)
        den_HL1_up = F.interpolate(den_HL1, size=LL1_recon.shape[-2:], mode='bilinear', align_corners=False)
        den_HH1_up = F.interpolate(den_HH1, size=LL1_recon.shape[-2:], mode='bilinear', align_corners=False)

        # Inverse DWT Level-1
        idwt1_input = torch.cat([LL1_recon, den_LH1_up, den_HL1_up, den_HH1_up], dim=1)
        recon = self.idwt1(idwt1_input)

        # Remove padding and return to original size
        return recon[:, :, :H, :W]


# --------------------------
# Edge-Aware Loss Function
# --------------------------
class EdgeAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
    def forward(self, output, target):
        # Standard MSE
        mse_loss = F.mse_loss(output, target)
        
        # Edge-aware loss
        target_edges_x = F.conv2d(target, self.sobel_x.to(target.device), padding=1)
        target_edges_y = F.conv2d(target, self.sobel_y.to(target.device), padding=1)
        target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2 + 1e-6)
        
        output_edges_x = F.conv2d(output, self.sobel_x.to(output.device), padding=1)
        output_edges_y = F.conv2d(output, self.sobel_y.to(output.device), padding=1)
        output_edges = torch.sqrt(output_edges_x**2 + output_edges_y**2 + 1e-6)
        
        edge_loss = F.l1_loss(output_edges, target_edges)
        
        return mse_loss + 0.5 * edge_loss


# --------------------------
# Test Run
# --------------------------
if __name__ == '__main__':
    # Create model and test data
    model = DWT2DenoisingModel(in_channels=3)
    criterion = EdgeAwareLoss()
    
    # Test forward pass
    image = torch.randn(1, 3, 256, 256)
    target = torch.randn(1, 3, 256, 256)
    
    output = model(image)
    loss = criterion(output, target)
    
    print("Output shape:", output.shape)  # Should be [1, 3, 256, 256]
    print("Loss value:", loss.item())
