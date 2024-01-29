"""Adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py"""
# import necessary modules
import numpy as np
import sys
import torch

from glob import glob
from os.path import dirname, join, realpath
from sklearn.metrics import confusion_matrix
from torch import Tensor
from torch.nn import *
from typing import Callable, Optional


def conv3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> Conv1d:
    """3 convolution with padding"""
    return Conv1d(in_planes, out_planes,
                  kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1(in_planes: int, out_planes: int, stride: int = 1) -> Conv1d:
    """1 convolution"""
    return Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[Module] = None,
                 groups: int = 1, base_width: int = 64, dilation: int = 1, norm_layer: Optional[Callable[..., Module]]
                 = None) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = ReLU(inplace=True)

        self.conv2 = conv3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class CADResNet(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        conv_width = config["conv_width"] if "conv_width" in config else 64
        replace_stride_with_dilation = config["replace_stride_with_dilation"] if "replace_stride_with_dilation" in config else None
        zero_init_residual = config["zero_init_residual"] if "zero_init_residual" in config else False

        # properties
        self.base_width = config["width_per_group"] if "width_per_group" in config else 64
        self.block = self.str_to_attr(config["block"])
        self._norm_layer = config["norm_layer"] if "norm_layer" in config else BatchNorm1d
        self.dilation = 1
        self.groups = config["groups"] if "groups" in config else 1
        self.layers = config["layers"]
        self.inplanes = conv_width

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2-stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple,"
                             f" got {replace_stride_with_dilation}")

        # first layer
        self.conv1 = Conv1d(config["n_channels"], self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        if self.layers[0]:
            self.layer1 = self._make_layer(conv_width, self.layers[0])
        if self.layers[1]:
            self.layer2 = self._make_layer(conv_width * 2, self.layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        if self.layers[2]:
            self.layer3 = self._make_layer(conv_width * 4, self.layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        if self.layers[3]:
            self.layer4 = self._make_layer(conv_width * 8, self.layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        # final layer
        self.avgpool = AdaptiveMaxPool1d(1)
        factor = sum(x > 0 for x in self.layers) - 1
        self.fc = Conv1d(conv_width * (2 ** factor) * self.block.expansion, config["n_classes"], kernel_size=1)

        # initialize weights
        for m in self.modules():
            if isinstance(m, Conv1d):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (BatchNorm1d, GroupNorm)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

        # zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # this improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if zero_init_residual:
                init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, planes: int, blocks: int, stride: int = 1, dilate: bool = False) -> Sequential:
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * self.block.expansion:
            downsample = Sequential(conv1(self.inplanes, planes * self.block.expansion, stride),
                                    self._norm_layer(planes * self.block.expansion))

        layers = [self.block(self.inplanes, planes, stride, downsample, self.groups,
                             self.base_width, previous_dilation, self._norm_layer)]
        self.inplanes = planes * self.block.expansion
        for _ in range(1, blocks):
            layers.append(
                self.block(self.inplanes, planes,
                           groups=self.groups, base_width=self.base_width,
                           dilation=self.dilation, norm_layer=self._norm_layer))
        return Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        shape = x.size()
        # -> B x N x C x L

        x = x.view(-1, shape[-2], shape[-1])
        # -> (B x N) x C x L

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.layers[0]:
            x = self.layer1(x)
        if self.layers[1]:
            x = self.layer2(x)
        if self.layers[2]:
            x = self.layer3(x)
        if self.layers[3]:
            x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = x.view(shape[0], shape[1], -1)
        # -> B x N x C

        # select the artery along axis N with the highest overall final activations
        indices = torch.argmax(torch.sum(x, dim=2), dim=1)
        indices = torch.repeat_interleave(indices.view(-1, 1, 1), x.size()[-1], dim=-1)
        x = torch.transpose(torch.gather(x, 1, indices), 1, 2)
        # -> B x C x 1

        return x.squeeze()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def infer(self, loader):
        self.eval()
        self.to(self.device)

        pred = []
        target = []
        with torch.no_grad():
            for i, (x, y, _) in enumerate(loader):
                x = x.to(self.device)
                y = y.to(self.device)

                y_hat = torch.zeros_like(y)
                for model in self.config["weights"]:
                    path = dirname(dirname(realpath(__file__)))
                    ckpt = sorted(glob(join(path, "lightning_logs", model, "*.pt")))[0]
                    self.load(ckpt)

                    y_hat += torch.sigmoid(self(x))

                y_hat /= len(self.config["weights"])
                pred.extend(y_hat.detach().cpu().numpy())
                target.extend(y.detach().cpu().numpy())

        return self.confusion_matrix(np.array(pred), np.array(target)), pred, target

    def confusion_matrix(self, pred, target, threshold=True):
        if threshold:
            # maybe play around with threshold value, lower could be better for higher CAD-RADS due to lack of data
            pred = pred > 0.5
            # pred[:, :3] = pred[:, :3] > 0.5
            # pred[:, 3:] = pred[:, 3:] > 0.4
            pred = pred.sum(axis=1, dtype=int)
        else:
            pred = np.clip(pred, 0, 5)

        target = target.sum(axis=1, dtype=int)
        matrix = confusion_matrix(target, pred, labels=[0, 1, 2, 3, 4, 5])
        return matrix

    def load(self, ckpt, verbose=False):
        state_dict = torch.load(ckpt, map_location="cuda:0")
        log = self.load_state_dict(state_dict, strict=False)
        if verbose:
            print("missing keys:", log.missing_keys)
            print("unexpected keys:", log.unexpected_keys)

    @staticmethod
    def str_to_attr(classname):
        return getattr(sys.modules[__name__], classname)

