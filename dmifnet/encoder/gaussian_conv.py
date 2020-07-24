import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, k=0.3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = self.gauss(3, k)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False).cuda()

    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=1, groups=self.channels).to('cuda')
        return x

    def gauss(self, kernel_size, sigma):
        kernel1 = cv2.getGaussianKernel(kernel_size, sigma)
        kernel2 = cv2.getGaussianKernel(kernel_size, sigma)
        kernel3 = np.multiply(kernel1, np.transpose(kernel2))

        return kernel3