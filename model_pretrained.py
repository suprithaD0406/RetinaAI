import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PretrainedUNet(nn.Module):
    def __init__(self, pretrained=True, out_channels=1):
        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        features = efficientnet_b0(weights=weights).features

        self.encoder0 = features[0]
        self.encoder1 = features[1]
        self.encoder2 = features[2]
        self.encoder3 = features[3]
        self.encoder4 = features[4]
        self.encoder5 = features[5]
        self.encoder6 = features[6]
        self.encoder7 = features[7]
        self.encoder8 = features[8]

        self.up1 = nn.ConvTranspose2d(1280, 256, 2, stride=2)
        self.dec1 = ConvBlock(256 + 112 + 80, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(128 + 40, 128)

        self.up3 = nn.ConvTranspose2d(128, 96, 2, stride=2)
        self.dec3 = ConvBlock(96 + 24, 96)

        self.up4 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.dec4 = ConvBlock(64 + 16, 64)

        self.up5 = nn.ConvTranspose2d(64, 48, 2, stride=2)
        self.dec5 = ConvBlock(48 + 32, 48)

        self.up6 = nn.ConvTranspose2d(48, 32, 2, stride=2)
        self.final = nn.Sequential(
            ConvBlock(32, 24),
            nn.Conv2d(24, out_channels, kernel_size=1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        x0 = self.encoder0(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)
        x6 = self.encoder6(x5)
        x7 = self.encoder7(x6)
        x8 = self.encoder8(x7)

        d1 = self.up1(x8)
        d1 = nn.functional.interpolate(
            d1, size=x5.shape[-2:],
            mode="bilinear", align_corners=False
        )
        d1 = self.dec1(torch.cat([d1, x5, x4], dim=1))

        d2 = self.up2(d1)
        d2 = nn.functional.interpolate(
            d2, size=x3.shape[-2:],
            mode="bilinear", align_corners=False
        )
        d2 = self.dec2(torch.cat([d2, x3], dim=1))

        d3 = self.up3(d2)
        d3 = nn.functional.interpolate(
            d3, size=x2.shape[-2:],
            mode="bilinear", align_corners=False
        )
        d3 = self.dec3(torch.cat([d3, x2], dim=1))

        d4 = self.up4(d3)
        d4 = nn.functional.interpolate(
            d4, size=x1.shape[-2:],
            mode="bilinear", align_corners=False
        )
        d4 = self.dec4(torch.cat([d4, x1], dim=1))

        d5 = self.up5(d4)
        d5 = nn.functional.interpolate(
            d5, size=x0.shape[-2:],
            mode="bilinear", align_corners=False
        )
        d5 = self.dec5(torch.cat([d5, x0], dim=1))

        d6 = self.up6(d5)
        output = self.final(d6)

        if output.shape[-2:] != input_size:
            output = nn.functional.interpolate(
                output, size=input_size,
                mode="bilinear", align_corners=False
            )

        return output


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PretrainedUNet(pretrained=True).to(device)
    model.eval()

    x = torch.randn(1, 3, 512, 512, device=device)

    with torch.inference_mode():
        output = model(x)

    print("Device:", device)
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(output.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Model shape test passed.")