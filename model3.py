import torch
import torch.nn as nn
from config import Config

config = Config()

class ConvModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvModule, self).__init__()

        layers = [
            nn.Conv2d(input_dim, output_dim, 1),
            nn.Conv2d(output_dim, output_dim, 3, padding=1, groups=output_dim),
            nn.InstanceNorm2d(output_dim),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Dropout(config.drop),
            nn.Conv2d(output_dim, output_dim, 1),
            nn.Conv2d(output_dim, output_dim, 3, padding=1, groups=output_dim),
            nn.InstanceNorm2d(output_dim),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Dropout(config.drop),
        ]

        if not config.useInstanceNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.InstanceNorm2d)]
        if not config.useBatchNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.BatchNorm2d)]
        if not config.useDropout:
            layers = [layer for layer in layers if not isinstance(layer, nn.Dropout)]

        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)

class BaseNet(nn.Module):
    def __init__(self, input_channels=3,
    encoder=config.encoderLayerSizes, decoder=config.decoderLayerSizes, output_channels=config.k):
        super(BaseNet, self).__init__()

        layers = [
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(config.drop),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(config.drop),
        ]

        if not config.useInstanceNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.InstanceNorm2d)]
        if not config.useBatchNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.BatchNorm2d)]
        if not config.useDropout:
            layers = [layer for layer in layers if not isinstance(layer, nn.Dropout)]

        self.first_module = nn.Sequential(*layers)


        self.pool = nn.MaxPool2d(2, 2)
        self.enc_modules = nn.ModuleList(
            [ConvModule(channels, 2*channels) for channels in encoder])


        decoder_out_sizes = [int(x/2) for x in decoder]
        self.dec_transpose_layers = nn.ModuleList(
            [nn.ConvTranspose2d(channels, channels, 2, stride=2) for channels in decoder]) # Stride of 2 makes it right size
        self.dec_modules = nn.ModuleList(
            [ConvModule(3*channels_out, channels_out) for channels_out in decoder_out_sizes])
        self.last_dec_transpose_layer = nn.ConvTranspose2d(128, 128, 2, stride=2)

        layers = [
            nn.Conv2d(128+64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(config.drop),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(config.drop),

            nn.Conv2d(64, output_channels, 1), # No padding on pointwise
            nn.ReLU(),
        ]

        if not config.useInstanceNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.InstanceNorm2d)]
        if not config.useBatchNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.BatchNorm2d)]
        if not config.useDropout:
            layers = [layer for layer in layers if not isinstance(layer, nn.Dropout)]

        self.last_module = nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.first_module(x)
        activations = [x1]
        for module in self.enc_modules:
            activations.append(module(self.pool(activations[-1])))

        x_ = activations.pop(-1)

        for conv, upconv in zip(self.dec_modules, self.dec_transpose_layers):
            skip_connection = activations.pop(-1)
            x_ = conv(
                torch.cat((skip_connection, upconv(x_)), 1)
            )

        segmentations = self.last_module(
            torch.cat((activations[-1], self.last_dec_transpose_layer(x_)), 1)
        )
        return segmentations, x_


class WNet(nn.Module):
    def __init__(self):
        super(WNet, self).__init__()

        self.U_encoder = BaseNet(input_channels=3, encoder=config.encoderLayerSizes,
                                    decoder=config.decoderLayerSizes, output_channels=config.k)
        self.softmax = nn.Softmax2d()
        self.U_decoder = BaseNet(input_channels=config.k+512, encoder=config.encoderLayerSizes,
                                    decoder=config.decoderLayerSizes, output_channels=3)

    def forward_encoder(self, x):
        x9, encoder_last = self.U_encoder(x)
        segmentations = self.softmax(x9)
        return segmentations, encoder_last

    def forward_decoder(self, segmentations, encoder_last):
        x18, _ = self.U_decoder(torch.cat((segmentations, encoder_last), 1))
        reconstructions = x18
        return reconstructions

    def forward(self, x): 
        segmentations, encoder_last = self.forward_encoder(x)
        x_prime = self.forward_decoder(segmentations, encoder_last)
        return segmentations, x_prime
