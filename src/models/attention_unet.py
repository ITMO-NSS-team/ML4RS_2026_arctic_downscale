import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """Attention gate for filtering encoder skip features"""

    def __init__(self, encoder_channels, decoder_channels, intermediate_channels):
        super(AttentionGate, self).__init__()

        self.encoder_projection = nn.Sequential(
            nn.Conv2d(
                encoder_channels, intermediate_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(intermediate_channels),
        )

        self.decoder_projection = nn.Sequential(
            nn.Conv2d(
                decoder_channels, intermediate_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(intermediate_channels),
        )

        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, encoder_features, decoder_features):
        attention = self.encoder_projection(
            encoder_features
        ) + self.decoder_projection(decoder_features)
        attention = self.attention(attention)

        return encoder_features * attention


class AttentionUNet(nn.Module):
    """Attention U-Net"""

    def __init__(self, in_channels=1, out_channels=1):
        super(AttentionUNet, self).__init__()

        self.enc1 = self.conv_block(in_channels, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)

        self.bottleneck = self.conv_block(64, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.attention1 = AttentionGate(64, 64, 32)
        self.dec1 = self.conv_block(128, 64)

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.attention2 = AttentionGate(32, 32, 16)
        self.dec2 = self.conv_block(64, 32)

        self.upconv3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.attention3 = AttentionGate(16, 16, 8)
        self.dec3 = self.conv_block(32, 16)

        self.upscale1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        self.upscale2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        self.final_upscale = nn.Sequential(
            nn.Upsample(size=(2100, 2550), mode="bilinear", align_corners=False),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, out_channels, kernel_size=1),
        )

    def conv_block(self, in_channels, out_channels):
        """Create a two-layer convolution block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))

        bottleneck = self.bottleneck(F.max_pool2d(enc3, 2))

        up1 = self.upconv1(bottleneck)
        gated_enc3 = self.attention1(enc3, up1)
        dec1 = self.dec1(torch.cat([up1, gated_enc3], dim=1))

        up2 = self.upconv2(dec1)
        gated_enc2 = self.attention2(enc2, up2)
        dec2 = self.dec2(torch.cat([up2, gated_enc2], dim=1))

        up3 = self.upconv3(dec2)
        gated_enc1 = self.attention3(enc1, up3)
        dec3 = self.dec3(torch.cat([up3, gated_enc1], dim=1))

        del enc1, enc2, enc3, bottleneck, up1, up2, up3, dec1, dec2
        del gated_enc1, gated_enc2, gated_enc3

        out = self.upscale1(dec3)
        del dec3

        out = self.upscale2(out)

        out = self.final_upscale(out)

        return out
