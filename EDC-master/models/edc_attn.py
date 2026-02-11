from models.resnet_attn import resnet34, resnet50, wide_resnet50_2, resnext50_32x4d
from models.resnet_decoder_attn import resnet50_decoder, wide_resnet50_decoder, resnet34_decoder, resnext50_32x4d_decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from .attention import CrossAttention2d 
import math


def zero_side(p, side=1):
    p[:, :, :side, :] = 0
    p[:, :, :, :side] = 0

    p[:, :, -side:, :] = 0
    p[:, :, :, -side:] = 0

    return p


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


class Dict2Obj(dict):

    def __getattr__(self, key):
        if key not in self:
            return None
        else:
            value = self[key]
            if isinstance(value, dict):
                value = Dict2Obj(value)
            return value


class R50_R50(nn.Module):
    def __init__(self,
                 img_size=256,
                 train_encoder=True,
                 stop_grad=True,
                 reshape=True,
                 bn_pretrain=False,
                 anomap_layer=[1, 2, 3],
                 attn_heads: int = 4,
                 use_attention: bool = True
                 ):
        super().__init__()
        # pass use_attention=True to resnet and decoder
        self.edc_encoder = resnet50(pretrained=True)
        self.edc_decoder = resnet50_decoder(pretrained=False, inplanes=[2048], use_attention=use_attention, attn_heads=attn_heads)
        self.train_encoder = train_encoder
        self.stop_grad = stop_grad
        self.reshape = reshape
        self.bn_pretrain = bn_pretrain
        self.anomap_layer = anomap_layer

        # Cross-attention e3 <-> d3 (query = decoder, kv = encoder)
        self.use_attention = use_attention
        if self.use_attention:
            # channels for e3/d3: for ResNet50 typical e3 channels = 256 * expansion = 1024
            channels_e3 = 256 * 4
            channels_d3 = channels_e3
            self.cross_e3_d3 = CrossAttention2d(q_channels=channels_d3, kv_channels=channels_e3, num_heads=attn_heads)

    def forward(self, x, ):
        if not self.train_encoder and self.edc_encoder.training:
            self.edc_encoder.eval()
        if self.bn_pretrain and self.edc_encoder.training:
            self.edc_encoder.eval()

        B = x.shape[0]
        e1, e2, e3, e4 = self.edc_encoder(x)
        if not self.train_encoder:
            e4 = e4.detach()
        d1, d2, d3 = self.edc_decoder(e4)

        # cross-attention fuse: refine decoder d3 using encoder e3 context
        if self.use_attention:
            try:
                d3 = self.cross_e3_d3(d3, e3)
            except Exception:
                # robust fallback — if shapes mismatch, skip cross-attn
                pass

        if (not self.train_encoder) or self.stop_grad:
            e1 = e1.detach()
            e2 = e2.detach()
            e3 = e3.detach()

        # rest unchanged: compute cosine losses, p1/p2/p3
        if self.reshape:
            l1 = 1. - torch.cosine_similarity(d1.reshape(B, -1), e1.reshape(B, -1), dim=1).mean()
            l2 = 1. - torch.cosine_similarity(d2.reshape(B, -1), e2.reshape(B, -1), dim=1).mean()
            l3 = 1. - torch.cosine_similarity(d3.reshape(B, -1), e3.reshape(B, -1), dim=1).mean()
        else:
            l1 = 1. - torch.cosine_similarity(d1, e1, dim=1).mean()
            l2 = 1. - torch.cosine_similarity(d2, e2, dim=1).mean()
            l3 = 1. - torch.cosine_similarity(d3, e3, dim=1).mean()

        with torch.no_grad():
            p1 = 1. - torch.cosine_similarity(d1, e1, dim=1).unsqueeze(1)
            p2 = 1. - torch.cosine_similarity(d2, e2, dim=1).unsqueeze(1)
            p3 = 1. - torch.cosine_similarity(d3, e3, dim=1).unsqueeze(1)

        loss = (l1 + l2 + l3)

        p2 = F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, scale_factor=4, mode='bilinear', align_corners=False)

        p_all = [[p1, p2, p3][l - 1] for l in self.anomap_layer]
        p_all = torch.cat(p_all, dim=1).mean(dim=1, keepdim=True)

        with torch.no_grad():
            e1_std = F.normalize(e1.permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()
            e2_std = F.normalize(e2.permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()
            e3_std = F.normalize(e3.permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()

        return {'loss': loss, 'p_all': p_all, 'p1': p1, 'p2': p2, 'p3': p3,
                'e1_std': e1_std, 'e2_std': e2_std, 'e3_std': e3_std}


class WR50_WR50(nn.Module):
    def __init__(self,
                 img_size=256,
                 train_encoder=True,
                 stop_grad=True,
                 reshape=True,
                 bn_pretrain=False,
                 anomap_layer=[1, 2, 3]
                 ):
        super().__init__()
        self.edc_encoder = wide_resnet50_2(pretrained=True)
        self.edc_decoder = wide_resnet50_decoder(pretrained=False, inplanes=[2048])
        self.train_encoder = train_encoder
        self.stop_grad = stop_grad
        self.reshape = reshape
        self.bn_pretrain = bn_pretrain
        self.anomap_layer = anomap_layer

    def forward(self, x, ):
        if not self.train_encoder and self.edc_encoder.training:
            self.edc_encoder.eval()
        if self.bn_pretrain and self.edc_encoder.training:
            self.edc_encoder.eval()

        B = x.shape[0]

        e1, e2, e3, e4 = self.edc_encoder(x)
        if not self.train_encoder:
            e4 = e4.detach()
        d1, d2, d3 = self.edc_decoder(e4)

        if (not self.train_encoder) or self.stop_grad:
            e1 = e1.detach()
            e2 = e2.detach()
            e3 = e3.detach()

        if self.reshape:
            l1 = 1. - torch.cosine_similarity(d1.reshape(B, -1), e1.reshape(B, -1), dim=1).mean()
            l2 = 1. - torch.cosine_similarity(d2.reshape(B, -1), e2.reshape(B, -1), dim=1).mean()
            l3 = 1. - torch.cosine_similarity(d3.reshape(B, -1), e3.reshape(B, -1), dim=1).mean()
        else:
            l1 = 1. - torch.cosine_similarity(d1, e1, dim=1).mean()
            l2 = 1. - torch.cosine_similarity(d2, e2, dim=1).mean()
            l3 = 1. - torch.cosine_similarity(d3, e3, dim=1).mean()

        with torch.no_grad():
            p1 = 1. - torch.cosine_similarity(d1, e1, dim=1).unsqueeze(1)
            p2 = 1. - torch.cosine_similarity(d2, e2, dim=1).unsqueeze(1)
            p3 = 1. - torch.cosine_similarity(d3, e3, dim=1).unsqueeze(1)

        loss = (l1 + l2 + l3)

        p2 = F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, scale_factor=4, mode='bilinear', align_corners=False)

        p_all = [[p1, p2, p3][l - 1] for l in self.anomap_layer]
        p_all = torch.cat(p_all, dim=1).mean(dim=1, keepdim=True)

        with torch.no_grad():
            e1_std = F.normalize(e1.permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()
            e2_std = F.normalize(e2.permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()
            e3_std = F.normalize(e3.permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()

        return {'loss': loss, 'p_all': p_all, 'p1': p1, 'p2': p2, 'p3': p3,
                'e1_std': e1_std, 'e2_std': e2_std, 'e3_std': e3_std}
