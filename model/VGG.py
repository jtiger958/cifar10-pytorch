import torch.nn as nn
import torchvision.models.vgg

class vgg16(nn.Module):
    def __init__(self, num_classes):
        super(vgg16, self).__init__()

        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        self.feature = self.make_feature_layer(cfg=cfg)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        print(x.shape)
        x = self.feature(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.classifier(x)
        return x

    def make_feature_layer(self, cfg, batch_noram=False):
        layers = []
        in_channels = 3

        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                if batch_noram == True:
                    layers.append(
                        nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, stride=1, padding=1))
                    layers.append(nn.BatchNorm2d(v))
                    layers.append(nn.ReLU(inplace=True))
                else:
                    layers.append(
                        nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, stride=1, padding=1))
                    layers.append(nn.ReLU(inplace=True))
                in_channels = v

        return nn.Sequential(*layers)
