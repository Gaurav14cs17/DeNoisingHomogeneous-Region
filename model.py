import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_

class HaarDWT(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        # Orthonormal Haar wavelet kernels
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) * 0.5
        
        kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('filters', kernel.repeat(in_channels, 1, 1, 1))
        
    def forward(self, x):
        # Symmetric padding for perfect reconstruction
        x = F.pad(x, (1, 0, 1, 0), mode='reflect')
        return F.conv2d(x, self.filters, stride=2, groups=self.in_channels)

class HaarIDWT(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        
        # Adjoint operators for perfect reconstruction
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) * 0.5
        
        kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('inv_filters', kernel.repeat(out_channels, 1, 1, 1))
        
    def forward(self, x):
        # Matching padding for perfect reconstruction
        x = F.pad(x, (0, 1, 0, 1), mode='reflect')
        return F.conv_transpose2d(x, self.inv_filters, stride=2, groups=self.out_channels)

class PerfectReconstructionTest(nn.Module):
    """Module to verify perfect reconstruction property"""
    def __init__(self, channels):
        super().__init__()
        self.dwt = HaarDWT(channels)
        self.idwt = HaarIDWT(channels)
        
    def forward(self, x):
        coeffs = self.dwt(x)
        return self.idwt(coeffs)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
    def forward(self, x):
        return x + self.conv(x)

class FrequencyAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//4, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels//4, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return x * self.fc(x)

class EdgeAwareDenoiser(nn.Module):
    def __init__(self, in_channels, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels*4),
            FrequencyAttention(base_channels*4),
            ResidualBlock(base_channels*4)
        )
        
        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels, 2, stride=2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dec1 = nn.Conv2d(base_channels*2, in_channels, 3, padding=1)
        
        # Edge enhancement
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x, mask):
        # Apply frequency mask
        x = x * mask
        
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Bottleneck
        b = self.bottleneck(e3)
        
        # Decoder path with skip connections
        d3 = self.dec3(b)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        # Edge-aware refinement
        edges = self.edge_conv(x)
        return d1 + edges

class DWT2DenoisingModel(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        
        # Wavelet transforms
        self.dwt1 = HaarDWT(in_channels)
        self.dwt2 = HaarDWT(in_channels)
        self.idwt2 = HaarIDWT(in_channels)
        self.idwt1 = HaarIDWT(in_channels)
        
        # Low-frequency processor
        self.ll_processor = nn.Sequential(
            ResidualBlock(in_channels),
            FrequencyAttention(in_channels),
            ResidualBlock(in_channels)
        )
        
        # Edge and flatness detection
        self.edge_detector = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels//2, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Main denoiser
        self.denoiser = EdgeAwareDenoiser(in_channels=6*in_channels)
        
        # Initialize
        self._initialize_weights()
        self.recon_test = PerfectReconstructionTest(in_channels)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def compute_frequency_mask(self, ll_band):
        # Flatness mask
        mean = F.avg_pool2d(ll_band, 3, stride=1, padding=1)
        variance = (ll_band - mean).pow(2)
        flatness = 1 - torch.tanh(variance.mean(dim=1, keepdim=True) * 5
        
        # Edge mask
        edges = self.edge_detector(ll_band)
        
        # Combined mask
        return torch.sigmoid(flatness + edges)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Pad to multiples of 4
        pad_h = (4 - H % 4) % 4
        pad_w = (4 - W % 4) % 4
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        # Level 1 decomposition
        dwt1_out = self.dwt1(x_pad)
        LL1, LH1, HL1, HH1 = torch.chunk(dwt1_out, 4, dim=1)
        
        # Level 2 decomposition
        dwt2_out = self.dwt2(LL1)
        LL2, LH2, HL2, HH2 = torch.chunk(dwt2_out, 4, dim=1)
        
        # Process LL band
        LL2_processed = self.ll_processor(LL2)
        
        # Compute frequency-aware mask
        freq_mask = self.compute_frequency_mask(LL2_processed)
        
        # Prepare denoiser input
        denoiser_input = torch.cat([
            LH2 * freq_mask,
            HL2 * freq_mask,
            HH2 * freq_mask,
            F.interpolate(LH1, scale_factor=0.5, mode='bilinear') * freq_mask,
            F.interpolate(HL1, scale_factor=0.5, mode='bilinear') * freq_mask,
            F.interpolate(HH1, scale_factor=0.5, mode='bilinear') * freq_mask
        ], dim=1)
        
        # Denoise high frequencies
        denoised = self.denoiser(denoiser_input, freq_mask)
        den_LH2, den_HL2, den_HH2, den_LH1, den_HL1, den_HH1 = torch.chunk(denoised, 6, dim=1)
        
        # Reconstruct level 2
        idwt2_in = torch.cat([LL2_processed, den_LH2, den_HL2, den_HH2], dim=1)
        LL1_recon = self.idwt2(idwt2_in)
        
        # Reconstruct level 1
        idwt1_in = torch.cat([
            LL1_recon,
            F.interpolate(den_LH1, size=LL1_recon.shape[-2:], mode='bilinear'),
            F.interpolate(den_HL1, size=LL1_recon.shape[-2:], mode='bilinear'),
            F.interpolate(den_HH1, size=LL1_recon.shape[-2:], mode='bilinear')
        ], dim=1)
        
        recon = self.idwt1(idwt1_in)
        
        # Crop to original size
        return recon[:, :, :H, :W]

class MultiScaleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
    def forward(self, output, target):
        # Pixel-level loss
        mse_loss = self.mse(output, target)
        
        # Edge loss
        def sobel(t):
            gx = F.conv2d(t, self.sobel_x.repeat(t.size(1), 1, 1, 1), padding=1, groups=t.size(1))
            gy = F.conv2d(t, self.sobel_y.repeat(t.size(1), 1, 1, 1), padding=1, groups=t.size(1))
            return torch.sqrt(gx**2 + gy**2 + 1e-6)
            
        edge_loss = F.l1_loss(sobel(output), sobel(target))
        
        # Frequency loss
        dwt = HaarDWT(3)
        out_coeffs = dwt(output)
        tgt_coeffs = dwt(target)
        freq_loss = sum(F.mse_loss(o, t) for o, t in zip(out_coeffs, tgt_coeffs))
        
        return mse_loss + 0.3 * edge_loss + 0.2 * freq_loss

if __name__ == '__main__':
    # Test perfect reconstruction
    print("Testing perfect reconstruction...")
    test_input = torch.randn(1, 3, 256, 256)
    model = DWT2DenoisingModel(in_channels=3)
    recon = model.recon_test(test_input)
    mse = F.mse_loss(test_input, recon[:, :, :256, :256]).item()
    print(f"Reconstruction MSE: {mse:.2e} (should be < 1e-6)")
    
    # Test full model
    print("\nTesting full model...")
    output = model(test_input)
    print(f"Input shape: {test_input.shape}, Output shape: {output.shape}")
    
    # Test loss function
    criterion = MultiScaleLoss()
    loss = criterion(output, test_input)
    print(f"Loss value: {loss.item():.4f}")
