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
            [[1, 1], [1, 1]],     # LL
            [[1, -1], [1, -1]],   # LH
            [[1, 1], [-1, -1]],   # HL
            [[1, -1], [-1, 1]]    # HH
        ], dtype=torch.float32).unsqueeze(1) / 2.0
        self.register_buffer('filters', kernel.repeat(in_channels, 1, 1, 1))

    def forward(self, x):
        return F.conv2d(x, self.filters, stride=2, groups=self.in_channels)


# --------------------------
# Inverse Haar DWT
# --------------------------
class HaarIDWT(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        inv_kernel = torch.tensor([
            [[1, 1], [1, 1]],
            [[1, -1], [1, -1]],
            [[1, 1], [-1, -1]],
            [[1, -1], [-1, 1]]
        ], dtype=torch.float32).unsqueeze(1) / 2.0
        self.register_buffer('inv_filters', inv_kernel.repeat(out_channels, 1, 1, 1))

    def forward(self, x):
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
        x = x * flatness_mask
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


# --------------------------
# Full DWT2 Denoising Model
# --------------------------
class DWT2DenoisingModel(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.dwt1 = HaarDWT(in_channels=in_channels)        # Output: 4*in_channels channels
        self.dwt2 = HaarDWT(in_channels=in_channels)        # Input: LL1 → in_channels channels
        self.idwt2 = HaarIDWT(out_channels=in_channels)     # Output: LL1_recon → in_channels channels
        self.idwt1 = HaarIDWT(out_channels=in_channels)     # Final image → in_channels channels
        self.unet = GatedUNet(in_ch=6*in_channels, base_ch=32)

    def forward(self, x):
        # Level-1 DWT
        dwt1_out = self.dwt1(x)  # (B, 4*in_channels, H/2, W/2)
        LL1, LH1, HL1, HH1 = torch.chunk(dwt1_out, 4, dim=1)  # (B, in_channels, H/2, W/2)

        # Level-2 DWT
        dwt2_out = self.dwt2(LL1)  # (B, 4*in_channels, H/4, W/4)
        LL2, LH2, HL2, HH2 = torch.chunk(dwt2_out, 4, dim=1)  # (B, in_channels, H/4, W/4)

        # Flatness-aware gating
        flatness_mask = compute_flatness_mask(LL2)  # (B, 1, H/4, W/4)

        # Downsample level-1 HF bands
        LH1_ds = F.interpolate(LH1, scale_factor=0.5, mode='bilinear', align_corners=False)
        HL1_ds = F.interpolate(HL1, scale_factor=0.5, mode='bilinear', align_corners=False)
        HH1_ds = F.interpolate(HH1, scale_factor=0.5, mode='bilinear', align_corners=False)

        # UNet input
        unet_input = torch.cat([LH2, HL2, HH2, LH1_ds, HL1_ds, HH1_ds], dim=1)  # (B, 6*in_channels, H/4, W/4)

        # Denoise HF
        denoised = self.unet(unet_input, flatness_mask)
        den_LH2, den_HL2, den_HH2, den_LH1, den_HL1, den_HH1 = torch.chunk(denoised, 6, dim=1)

        # Inverse DWT2
        idwt2_input = torch.cat([LL2, den_LH2, den_HL2, den_HH2], dim=1)  # (B, 4*in_channels, H/4, W/4)
        LL1_recon = self.idwt2(idwt2_input)  # (B, in_channels, H/2, W/2)

        # Upsample denoised LH1, HL1, HH1 back to H/2, W/2
        den_LH1_up = F.interpolate(den_LH1, size=LL1_recon.shape[-2:], mode='bilinear', align_corners=False)
        den_HL1_up = F.interpolate(den_HL1, size=LL1_recon.shape[-2:], mode='bilinear', align_corners=False)
        den_HH1_up = F.interpolate(den_HH1, size=LL1_recon.shape[-2:], mode='bilinear', align_corners=False)

        # Inverse DWT1
        idwt1_input = torch.cat([LL1_recon, den_LH1_up, den_HL1_up, den_HH1_up], dim=1)
        recon = self.idwt1(idwt1_input)  # (B, in_channels, H, W)

        return recon


# --------------------------
# Test Run
# --------------------------
if __name__ == '__main__':
    image = torch.randn(1, 3, 256, 256)  # Input image
    model = DWT2DenoisingModel()
    output = model(image)
    print("Output shape:", output.shape)  # Expected: [1, 3, 256, 256]
