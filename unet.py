import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Attention gate for Attention U-Net.

    """

    def __init__(self, in_channels, gate_channels):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv2d(in_channels, gate_channels, kernel_size=1, stride=1, padding=0)
        self.W_x = nn.Conv2d(in_channels, gate_channels, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(gate_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = F.relu(g1 + x1, inplace=True)
        psi = self.psi(psi)
        psi = F.sigmoid(psi)
        return x * psi


class DoubleConv(nn.Module):
    """
    Double convolution block used in the U-Net down and up paths.
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, should_pool=True):
        super(EncoderBlock, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.should_pool = should_pool

    def forward(self, x):
        if self.should_pool:
            x = self.conv(x)
            pool = self.pool(x)
            return x, pool
        else:
            return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = DoubleConv(in_channels * 2, out_channels)  # Concatenation of skip and upsampled signal
        self.attention = AttentionGate(in_channels, in_channels)

    def forward(self, x, s):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        s = self.attention(x, s)
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        return x


class AttUNet(nn.Module):
    """
    Attention U-Net model for image segmentation.
    """

    def __init__(self, in_channels, num_classes, filters=None, final_activation=None, include_reconstruction=True):
        super(AttUNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        if filters is None:
            filters = [32, 64, 128, 256, 512]

        # Encoder path
        for i in range(len(filters)):
            if i == 0:
                self.encoder.append(EncoderBlock(in_channels, filters[i]))
            else:
                self.encoder.append(EncoderBlock(filters[i - 1], filters[i]))

        # Bottleneck
        self.bottleneck = DoubleConv(filters[-1], filters[-1])

        # Decoder path
        for i in reversed(range(len(filters))):
            if i == 0:
                self.decoder.append(DecoderBlock(filters[i], filters[i]))
            else:
                self.decoder.append(DecoderBlock(filters[i], filters[i - 1]))

        self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        if final_activation == 'softmax':
            self.final_activation = nn.Softmax(dim=1)
        elif final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = None

        if include_reconstruction:
            self.recon_final = nn.Conv2d(filters[0], 1, kernel_size=1)
        else:
            self.recon_final = None

    def forward(self, x):

        # Down path
        skips = []
        for encoder in self.encoder:
            signal, pooled = encoder(x)
            skips.append(signal)
            x = pooled

        # Bottleneck
        x = self.bottleneck(x)

        # Up path
        for decoder in self.decoder:
            x = decoder(x, skips.pop())

        # Final prediction
        seg = self.final(x)
        if self.final_activation:
            seg = self.final_activation(seg)
        if self.recon_final:
            recon = self.recon_final(x)

            return seg, recon
        else:
            return seg
