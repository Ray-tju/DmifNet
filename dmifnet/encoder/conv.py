import torch.nn as nn
# import torch.nn.functional as F
from torchvision import models
from dmifnet.common import normalize_imagenet
from dmifnet.encoder import batchnet as bnet
from dmifnet.encoder import gaussian_conv
import torch
from dmifnet.encoder import channel_attention_1d


class ConvEncoder(nn.Module):
    r''' Simple convolutional encoder network.

    It consists of 5 convolutional layers, each downsampling the input by a
    factor of 2, and a final fully-connected layer projecting the output to
    c_dim dimenions.

    Args:
        c_dim (int): output dimension of latent embedding
    '''

    def __init__(self, c_dim=128):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 32, 3, stride=2)
        self.conv1 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2)
        self.fc_out = nn.Linear(512, c_dim)
        self.actvn = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        net = self.conv0(x)
        net = self.conv1(self.actvn(net))
        net = self.conv2(self.actvn(net))
        net = self.conv3(self.actvn(net))
        net = self.conv4(self.actvn(net))
        net = net.view(batch_size, 512, -1).mean(2)
        out = self.fc_out(self.actvn(net))

        return out


def gray(tensor):
    # TODO: make efficient
    # print(tensor)
    b, c, h, w = tensor.size()
    R = tensor[:, 0, :, :]
    G = tensor[:, 1, :, :]
    B = tensor[:, 2, :, :]
    tem = torch.add(0.299 * R, 0.587 * G)
    tensor_gray = torch.add(tem, 0.114 * B)
    # print(tensor_gray.size())
    return tensor_gray.view(b, 1, h, w)


class Resnet18(nn.Module):
    r''' ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        # self.features = models.resnet18(pretrained=True)
        self.features = bnet.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()
        self.features.fc_head1 = nn.Sequential()
        self.features.fc_head2 = nn.Sequential()
        self.gaussion_conv1 = gaussian_conv.GaussianBlurConv(1, 0.3).cuda()
        self.gaussion_conv2 = gaussian_conv.GaussianBlurConv(1, 0.4).cuda()
        self.gaussion_conv3 = gaussian_conv.GaussianBlurConv(1, 0.5).cuda()
        self.gaussion_conv4 = gaussian_conv.GaussianBlurConv(1, 0.6).cuda()
        self.gaussion_conv5 = gaussian_conv.GaussianBlurConv(1, 0.7).cuda()
        self.gaussion_conv6 = gaussian_conv.GaussianBlurConv(1, 0.8).cuda()
        self.dog_encoder = ConvEncoder(256).to('cuda')
        self.attention = channel_attention_1d.attention_layer(256)

        if use_linear:
            self.fc = nn.Linear(512, c_dim)
            self.fc_head1 = nn.Linear(1024, c_dim)
            self.fc_head2 = nn.Linear(512, c_dim)
            self.fusion_dog_ori = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        x0, x1, x2 = self.features(x)
        gray_x = gray(x)
        gaussian1 = self.gaussion_conv1(gray_x)
        gaussian2 = self.gaussion_conv2(gray_x)
        gaussian3 = self.gaussion_conv3(gray_x)
        gaussian4 = self.gaussion_conv4(gray_x)
        gaussian5 = self.gaussion_conv5(gray_x)
        gaussian6 = self.gaussion_conv6(gray_x)
        dog1 = torch.sub(gaussian2, gaussian1)
        dog2 = torch.sub(gaussian4, gaussian3)
        dog3 = torch.sub(gaussian6, gaussian5)
        dog_tem = torch.cat((dog1, dog2), dim=1)
        dog = torch.cat((dog_tem, dog3), dim=1)
        out_dog = self.dog_encoder(dog)
        out0 = self.fc(x0)
        out1 = self.fc_head1(x1)
        out2 = self.fc_head2(x2)
        out_dog = self.fusion_dog_ori(torch.cat((out_dog, out0), dim=1))
        attention_out_dog = self.attention(out_dog.unsqueeze(dim=-1))
        return out0, out1, out2, attention_out_dog.squeeze(1)


class Resnet34(nn.Module):
    r''' ResNet-34 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet34(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class Resnet50(nn.Module):
    r''' ResNet-50 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(2048, c_dim)
        elif c_dim == 2048:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 2048 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class Resnet101(nn.Module):
    r''' ResNet-101 encoder network.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(2048, c_dim)
        elif c_dim == 2048:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 2048 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out
