import torch.nn as nn


class vgg16(nn.Module):
    def __init__(self, num_classes, init_weights=True):
        super(vgg16, self).__init__()

        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256]

        self.feature = self.make_feature_layer(cfg=cfg)

        self.avg_pool = nn.AvgPool2d(kernel_size=7)

        self.classifier = nn.Sequential(
            nn.Linear(8 * 8 * 256, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
            nn.Softmax()
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
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
                    layers += [nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, stride=1, padding=1)]
                    layers += [nn.BatchNorm2d(v)]
                    layers += [nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, stride=1, padding=1)]
                    layers += [nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)