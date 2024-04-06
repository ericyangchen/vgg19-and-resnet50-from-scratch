from torch import nn


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Bottleneck(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, stride=1, downsample=None
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.__name__ = "ResNet50"

        # layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            Bottleneck(
                in_channels=64,
                hidden_channels=64,
                out_channels=256,
                stride=1,
                downsample=Downsample(in_channels=64, out_channels=256, stride=1),
            ),
            Bottleneck(in_channels=256, hidden_channels=64, out_channels=256, stride=1),
            Bottleneck(in_channels=256, hidden_channels=64, out_channels=256, stride=1),
        )
        self.layer2 = nn.Sequential(
            Bottleneck(
                in_channels=256,
                hidden_channels=128,
                out_channels=512,
                stride=2,
                downsample=Downsample(in_channels=256, out_channels=512, stride=2),
            ),
            Bottleneck(
                in_channels=512, hidden_channels=128, out_channels=512, stride=1
            ),
            Bottleneck(
                in_channels=512, hidden_channels=128, out_channels=512, stride=1
            ),
            Bottleneck(
                in_channels=512, hidden_channels=128, out_channels=512, stride=1
            ),
        )
        self.layer3 = nn.Sequential(
            Bottleneck(
                in_channels=512,
                hidden_channels=256,
                out_channels=1024,
                stride=2,
                downsample=Downsample(in_channels=512, out_channels=1024, stride=2),
            ),
            Bottleneck(
                in_channels=1024, hidden_channels=256, out_channels=1024, stride=1
            ),
            Bottleneck(
                in_channels=1024, hidden_channels=256, out_channels=1024, stride=1
            ),
            Bottleneck(
                in_channels=1024, hidden_channels=256, out_channels=1024, stride=1
            ),
            Bottleneck(
                in_channels=1024, hidden_channels=256, out_channels=1024, stride=1
            ),
            Bottleneck(
                in_channels=1024, hidden_channels=256, out_channels=1024, stride=1
            ),
        )
        self.layer4 = nn.Sequential(
            Bottleneck(
                in_channels=1024,
                hidden_channels=512,
                out_channels=2048,
                stride=2,
                downsample=Downsample(in_channels=1024, out_channels=2048, stride=2),
            ),
            Bottleneck(
                in_channels=2048, hidden_channels=512, out_channels=2048, stride=1
            ),
            Bottleneck(
                in_channels=2048, hidden_channels=512, out_channels=2048, stride=1
            ),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    import torch

    model = ResNet50()
    print(model)
