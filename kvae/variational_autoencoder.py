import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from misc import compute_conv2d_output_size


class Encoder(nn.Module):
    def __init__(self, image_size, image_channels, a_dim):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=image_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1
        )

        conv_output_size = image_size
        for _ in range(3):
            conv_output_size = compute_conv2d_output_size(
                conv_output_size, kernel_size=3, stride=2, padding=1
            )

        self.fc_mean = nn.Linear(
            in_features=32 * conv_output_size[0] * conv_output_size[1],
            out_features=a_dim,
        )
        self.fc_std = nn.Linear(
            in_features=32 * conv_output_size[0] * conv_output_size[1],
            out_features=a_dim,
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print("conv1:", x.shape)
        x = F.relu(self.conv2(x))
        # print("conv2:", x.shape)
        x = F.relu(self.conv3(x))
        # print("conv3:", x.shape)

        x_mean = self.fc_mean(x.view(x.shape[0], -1))
        x_std = F.softplus(self.fc_std(x.view(x.shape[0], -1)))

        return D.Normal(loc=x_mean, scale=x_std)


class BernoulliDecoder(nn.Module):
    def __init__(self, a_dim, image_size, image_channels):
        super(BernoulliDecoder, self).__init__()

        conv_output_size = image_size
        for _ in range(3):
            conv_output_size = compute_conv2d_output_size(
                conv_output_size, kernel_size=3, stride=2, padding=1
            )
        self.conv_output_size = conv_output_size

        self.fc = nn.Linear(
            in_features=a_dim,
            out_features=32 * conv_output_size[0] * conv_output_size[1],
        )
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=image_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(-1, 32, *self.conv_output_size)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)

        return D.Bernoulli(logits=x)


class GaussianDecoder(nn.Module):
    def __init__(self, a_dim, image_size, image_channels):
        super(GaussianDecoder, self).__init__()

        conv_output_size = image_size
        for _ in range(3):
            conv_output_size = compute_conv2d_output_size(
                conv_output_size, kernel_size=3, stride=2, padding=1
            )
        self.conv_output_size = conv_output_size

        self.fc = nn.Linear(
            in_features=a_dim,
            out_features=32 * conv_output_size[0] * conv_output_size[1],
        )
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.deconv3_mean = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=image_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.deconv3_std = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=image_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(-1, 32, *self.conv_output_size)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x_mean = self.deconv3_mean(x)
        x_std = F.softplus(self.deconv3_std(x))

        return D.Normal(loc=x_mean, scale=x_std)
