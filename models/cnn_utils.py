import torch
from torch import nn
from torch.nn import functional as F

class CBR(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(nOut, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class CB(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(nOut, momentum=0.95, eps=1e-03)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups)

    def forward(self, input):
        output = self.conv(input)
        return output


class DownSamplerA(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.conv = CBR(nIn, nOut, 3, 2)

    def forward(self, input):
        output = self.conv(input)
        return output


class DownSamplerB(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        k = 4
        n = int(nOut/k)
        n1 = nOut - (k-1)*n
        self.c1 = nn.Sequential(CBR(nIn, n, 1, 1), C(n, n, 3, 2))
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 3)
        self.d8 = CDilated(n, n, 3, 1, 4)
        self.bn = BR(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8

        combine = torch.cat([d1, add1, add2, add3],1)
        if input.size() == combine.size():
            combine = input + combine
        output = self.bn(combine)
        return output


class BR(nn.Module):
    def __init__(self, nOut):
        super().__init__()
        self.bn = nn.BatchNorm3d(nOut, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(inplace=True)  # nn.PReLU(nOut)

    def forward(self, input):
        output = self.bn(input)
        output = self.act(output)
        return output


class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False,
                              dilation=d, groups=groups)
        #self.bn = nn.BatchNorm3d(nOut, momentum=0.95, eps=1e-03)

    def forward(self, input):
        return self.conv(input)
        #return self.bn(output)


class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''

    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            # pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool3d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input


class DilatedParllelResidualBlockB1(nn.Module):  # with k=4
    def __init__(self, nIn, nOut, stride=1):
        super().__init__()
        k = 4
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        self.c1 = CBR(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, stride, 1)
        self.d2 = CDilated(n, n, 3, stride, 1)
        self.d4 = CDilated(n, n, 3, stride, 2)
        self.d8 = CDilated(n, n, 3, stride, 2)
        self.bn = nn.BatchNorm3d(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8

        combine = self.bn(torch.cat([d1, add1, add2, add3], 1))
        if input.size() == combine.size():
            combine = input + combine
        output = F.relu(combine, inplace=True)
        return output

class ASPBlock(nn.Module):  # with k=4
    def __init__(self, nIn, nOut, stride=1):
        super().__init__()
        self.d1 = CB(nIn, nOut, 3, 1)
        self.d2 = CB(nIn, nOut, 5, 1)
        self.d4 = CB(nIn, nOut, 7, 1)
        self.d8 = CB(nIn, nOut, 9, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        d1 = self.d1(input)
        d2 = self.d2(input)
        d3 = self.d4(input)
        d4 = self.d8(input)

        combine = d1 + d2 + d3 + d4
        if input.size() == combine.size():
            combine = input + combine
        output = self.act(combine)
        return output


class UpSampler(nn.Module):
    '''
    Up-sample the feature maps by 2
    '''
    def __init__(self, nIn, nOut):
        super().__init__()
        self.up = CBR(nIn, nOut, 3, 1)

    def forward(self, inp):
        # return F.upsample(self.up(inp), mode='trilinear', scale_factor=2)
        return F.interpolate(self.up(inp), scale_factor=2, mode='trilinear', align_corners=False)


class PSPDec(nn.Module):
    '''
    Inspired or Adapted from Pyramid Scene Network paper
    '''

    def __init__(self, nIn, nOut, downSize):
        super().__init__()
        self.scale = downSize
        self.features = CBR(nIn, nOut, 3, 1)
    def forward(self, x):
        assert x.dim() == 5
        inp_size = x.size()
        out_dim1, out_dim2, out_dim3 = int(inp_size[2] * self.scale), int(inp_size[3] * self.scale), int(inp_size[4] * self.scale)
        x_down = F.adaptive_avg_pool3d(x, output_size=(out_dim1, out_dim2, out_dim3))
        # return F.upsample(self.features(x_down), size=(inp_size[2], inp_size[3], inp_size[4]), mode='trilinear')

        return F.interpolate(self.features(x_down), size=(inp_size[2], inp_size[3], inp_size[4]), mode='trilinear', align_corners=False)
