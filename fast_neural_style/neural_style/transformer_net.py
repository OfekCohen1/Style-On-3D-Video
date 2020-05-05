import torch


class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers

        self.conv1_left = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1_left = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2_left = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2_left = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3_left = ConvLayer(64, 96, kernel_size=3, stride=2)
        self.in3_left = torch.nn.InstanceNorm2d(96, affine=True)

        self.conv1_right = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1_right = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2_right = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2_right = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3_right = ConvLayer(64, 96, kernel_size=3, stride=2)
        self.in3_right = torch.nn.InstanceNorm2d(96, affine=True)

        # Residual layers
        self.res1 = ResidualBlock(192)
        self.res2 = ResidualBlock(192)
        self.res3 = ResidualBlock(192)
        self.res4 = ResidualBlock(192)
        self.res5 = ResidualBlock(192)

        # Upsampling Layers
        self.deconv1_left = UpsampleConvLayer(192, 96, kernel_size=3, stride=1, upsample=2)
        self.in4_left = torch.nn.InstanceNorm2d(96, affine=True)
        self.deconv2_left = UpsampleConvLayer(96, 48, kernel_size=3, stride=1, upsample=2)
        self.in5_left = torch.nn.InstanceNorm2d(48, affine=True)
        self.deconv3_left = ConvLayer(48, 3, kernel_size=9, stride=1)

        self.deconv1_right = UpsampleConvLayer(192, 96, kernel_size=3, stride=1, upsample=2)
        self.in4_right = torch.nn.InstanceNorm2d(96, affine=True)
        self.deconv2_right = UpsampleConvLayer(96, 48, kernel_size=3, stride=1, upsample=2)
        self.in5_right = torch.nn.InstanceNorm2d(48, affine=True)
        self.deconv3_right = ConvLayer(48, 3, kernel_size=9, stride=1)

        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, img_left, img_right):
        img_left = self.relu(self.in1_left(self.conv1_left(img_left)))
        img_left = self.relu(self.in2_left(self.conv2_left(img_left)))
        img_left = self.relu(self.in3_left(self.conv3_left(img_left)))

        img_right = self.relu(self.in1_right(self.conv1_right(img_right)))
        img_right = self.relu(self.in2_right(self.conv2_right(img_right)))
        img_right = self.relu(self.in3_right(self.conv3_right(img_right)))

        img_combined = torch.cat((img_left, img_right), 1)  # concat in channel dim

        img_combined = self.res1(img_combined)
        img_combined = self.res2(img_combined)
        img_combined = self.res3(img_combined)
        img_combined = self.res4(img_combined)
        img_combined = self.res5(img_combined)

        img_left = self.relu(self.in4_left(self.deconv1_left(img_combined)))
        img_left = self.relu(self.in5_left(self.deconv2_left(img_left)))
        img_left = self.deconv3_left(img_left)

        img_right = self.relu(self.in4_right(self.deconv1_right(img_combined)))
        img_right = self.relu(self.in5_right(self.deconv2_right(img_right)))
        img_right = self.deconv3_right(img_right)
        return img_left, img_right


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
