from torch import nn


class VGGNet19(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.__name__ = "VGGNet19"

        self.conv_configs = [
            # layer: 1,2
            {
                "in_channels": 3,
                "out_channels": 64,
                "num_convs": 2,
            },
            # layer: 3,4
            {
                "in_channels": 64,
                "out_channels": 128,
                "num_convs": 2,
            },
            # layer: 5,6,7,8
            {
                "in_channels": 128,
                "out_channels": 256,
                "num_convs": 4,
            },
            # layer: 9,10,11,12
            {
                "in_channels": 256,
                "out_channels": 512,
                "num_convs": 4,
            },
            # layer: 13,14,15,16
            {
                "in_channels": 512,
                "out_channels": 512,
                "num_convs": 4,
            },
        ]

        self.fc_configs = [
            # layer: 17
            {
                "in_features": 512 * 7 * 7,
                "out_features": 4096,
            },
            # layer: 18
            {
                "in_features": 4096,
                "out_features": 4096,
            },
            # layer: 19
            {
                "in_features": 4096,
                "out_features": 1000,
            },
        ]

        self.conv_blocks = self._build_conv_blocks()
        self.fc_blocks = self._build_fc_blocks()

    def _build_conv_blocks(self):
        blocks = []
        for conv_config in self.conv_configs:
            in_channels = conv_config["in_channels"]
            out_channels = conv_config["out_channels"]
            num_convs = conv_config["num_convs"]

            layers = []
            for _ in range(num_convs):
                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    )
                )
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels

            layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

            blocks.extend(layers)

        return nn.Sequential(*blocks)

    def _build_fc_blocks(self):
        blocks = []

        for i, fc_config in enumerate(self.fc_configs):
            in_features = fc_config["in_features"]
            out_features = fc_config["out_features"]

            if i < len(self.fc_configs) - 1:
                layers = [
                    nn.Linear(in_features=in_features, out_features=out_features),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                ]
            else:
                layers = [nn.Linear(in_features=in_features, out_features=out_features)]

            blocks.extend(layers)

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.flatten(start_dim=1)

        x = self.fc_blocks(x)

        return x


if __name__ == "__main__":
    import torch

    model = VGGNet19()
    print(model)
