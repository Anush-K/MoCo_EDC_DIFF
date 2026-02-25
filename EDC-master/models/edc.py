"""
models/edc.py  — drop-in replacement with MoCo pretrained encoder support.

Key changes vs original:
  - R50_R50 accepts moco_weights (str path or None) and freeze_encoder (bool)
  - If moco_weights given: loads MoCo encoder_q weights, skips MLP head (fc.*)
  - freeze_encoder=True  → Path 1 (frozen MoCo encoder, only decoder trains)
  - freeze_encoder=False → Path 2 (unfrozen MoCo encoder, full fine-tune)
  - moco_weights=None    → original EDC baseline (ImageNet pretrained)
  - WR50_WR50 and all utilities unchanged from original
  - No diffusion code here (lives in edc1.py, already gated by self.diffusion is not None)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet import resnet34, resnet50, wide_resnet50_2, resnext50_32x4d
from models.resnet_decoder import resnet50_decoder, wide_resnet50_decoder, resnet34_decoder, resnext50_32x4d_decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import math


# ─────────────────────────────────────────────────────
# Utility functions (unchanged from original)
# ─────────────────────────────────────────────────────

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
    x_coord = torch.arange(kernel_size)
    x_grid  = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid  = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean     = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(
        in_channels=channels, out_channels=channels,
        kernel_size=kernel_size, groups=channels,
        bias=False, padding=kernel_size // 2
    )
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter


class Dict2Obj(dict):
    def __getattr__(self, key):
        if key not in self:
            return None
        value = self[key]
        if isinstance(value, dict):
            value = Dict2Obj(value)
        return value


# ─────────────────────────────────────────────────────
# MoCo weight loader
# ─────────────────────────────────────────────────────

def load_moco_weights(encoder, moco_path):
    """
    Load MoCo pretrained encoder_q weights into the EDC ResNet50 encoder.

    MoCo checkpoint: {"encoder_q": state_dict}  where encoder_q has a custom
    MLP head (fc.0, fc.2).  EDC encoder's fc is Linear(2048, 1000).
    We load all matching keys by name+shape and skip the fc head automatically.
    """
    checkpoint = torch.load(moco_path, map_location='cpu')
    moco_state = checkpoint.get('encoder_q', checkpoint)  # fallback if no key

    encoder_state = encoder.state_dict()
    new_state = {}
    matched, skipped = [], []

    for k, v in moco_state.items():
        if k in encoder_state and encoder_state[k].shape == v.shape:
            new_state[k] = v
            matched.append(k)
        else:
            skipped.append(k)

    encoder_state.update(new_state)
    encoder.load_state_dict(encoder_state)

    print(f"[MoCo] Loaded {len(matched)} / {len(moco_state)} layers into encoder.")
    print(f"[MoCo] Skipped {len(skipped)} keys (expected: fc MLP head layers).")
    return encoder


# ─────────────────────────────────────────────────────
# R50_R50 — EDC model with MoCo support
# ─────────────────────────────────────────────────────

class R50_R50(nn.Module):
    def __init__(self,
                 img_size=256,
                 train_encoder=True,
                 stop_grad=True,
                 reshape=True,
                 bn_pretrain=False,
                 anomap_layer=[1, 2, 3],
                 moco_weights=None,      # path to MoCo .pth, or None for ImageNet
                 freeze_encoder=False,   # True = Path 1, False = Path 2
                 ):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        if moco_weights is not None:
            self.edc_encoder = resnet50(pretrained=False)
            print(f"[EDC] Loading MoCo pretrained weights from:\n      {moco_weights}")
            self.edc_encoder = load_moco_weights(self.edc_encoder, moco_weights)
        else:
            self.edc_encoder = resnet50(pretrained=True)
            print("[EDC] Using ImageNet pretrained encoder (original EDC baseline).")

        # ── Decoder ──────────────────────────────────────────────────────────
        self.edc_decoder = resnet50_decoder(pretrained=False, inplanes=[2048])

        # ── Flags ─────────────────────────────────────────────────────────────
        self.train_encoder  = train_encoder
        self.stop_grad      = stop_grad
        self.reshape        = reshape
        self.bn_pretrain    = bn_pretrain
        self.anomap_layer   = anomap_layer
        self.freeze_encoder = freeze_encoder

        # ── Path 1: freeze all encoder params ────────────────────────────────
        if self.freeze_encoder:
            for param in self.edc_encoder.parameters():
                param.requires_grad = False
            print("[EDC] Encoder FROZEN  → Path 1 (only decoder trains).")
        else:
            print("[EDC] Encoder UNFROZEN → Path 2 (full fine-tune).")

    def forward(self, x):
        # Keep encoder in eval mode when frozen or train_encoder=False
        if (not self.train_encoder or self.freeze_encoder) and self.edc_encoder.training:
            self.edc_encoder.eval()
        if self.bn_pretrain and self.edc_encoder.training:
            self.edc_encoder.eval()

        B = x.shape[0]

        e1, e2, e3, e4 = self.edc_encoder(x)

        if not self.train_encoder or self.freeze_encoder:
            e4 = e4.detach()

        d1, d2, d3 = self.edc_decoder(e4)

        if (not self.train_encoder or self.freeze_encoder) or self.stop_grad:
            e1 = e1.detach()
            e2 = e2.detach()
            e3 = e3.detach()

        # ── Cosine similarity loss (identical to original) ────────────────────
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

        loss = l1 + l2 + l3

        p2 = F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, scale_factor=4, mode='bilinear', align_corners=False)

        p_all = [[p1, p2, p3][l - 1] for l in self.anomap_layer]
        p_all = torch.cat(p_all, dim=1).mean(dim=1, keepdim=True)

        with torch.no_grad():
            e1_std = F.normalize(e1.permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()
            e2_std = F.normalize(e2.permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()
            e3_std = F.normalize(e3.permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()

        return {
            'loss': loss, 'p_all': p_all,
            'p1': p1, 'p2': p2, 'p3': p3,
            'e1_std': e1_std, 'e2_std': e2_std, 'e3_std': e3_std
        }


# ─────────────────────────────────────────────────────
# WR50_WR50 — unchanged from original
# ─────────────────────────────────────────────────────

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
        self.edc_encoder   = wide_resnet50_2(pretrained=True)
        self.edc_decoder   = wide_resnet50_decoder(pretrained=False, inplanes=[2048])
        self.train_encoder = train_encoder
        self.stop_grad     = stop_grad
        self.reshape       = reshape
        self.bn_pretrain   = bn_pretrain
        self.anomap_layer  = anomap_layer

    def forward(self, x):
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

        loss = l1 + l2 + l3

        p2 = F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, scale_factor=4, mode='bilinear', align_corners=False)

        p_all = [[p1, p2, p3][l - 1] for l in self.anomap_layer]
        p_all = torch.cat(p_all, dim=1).mean(dim=1, keepdim=True)

        with torch.no_grad():
            e1_std = F.normalize(e1.permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()
            e2_std = F.normalize(e2.permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()
            e3_std = F.normalize(e3.permute(1, 0, 2, 3).flatten(1), dim=0).std(dim=1).mean()

        return {
            'loss': loss, 'p_all': p_all,
            'p1': p1, 'p2': p2, 'p3': p3,
            'e1_std': e1_std, 'e2_std': e2_std, 'e3_std': e3_std
        }