import contextlib
from pyparsing import alphas
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# ADD
from models.ema import EMA
# ADD
from copy import deepcopy


# ============================================================
# Time Embedding
# ============================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        """
        t: [B] integer timesteps
        returns: [B, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.mlp(emb)


# ============================================================
# Residual Block (Diffusion UNet)
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        self.time_mlp = nn.Linear(time_dim, channels)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        time_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return x + h


# ============================================================
# Diffusion UNet (NO spatial downsampling)
# ============================================================

class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=1024, base_channels=512, time_dim=512):
        super().__init__()

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.time_embed = TimeEmbedding(time_dim)

        self.block1 = ResBlock(base_channels, time_dim)
        self.block2 = ResBlock(base_channels, time_dim)

        self.mid_conv = nn.Conv2d(base_channels, base_channels // 2, 3, padding=1)

        self.block3 = ResBlock(base_channels // 2, time_dim)
        self.block4 = ResBlock(base_channels // 2, time_dim)

        self.up_conv = nn.Conv2d(base_channels // 2, base_channels, 3, padding=1)

        self.block5 = ResBlock(base_channels, time_dim)
        self.block6 = ResBlock(base_channels, time_dim)

        self.final_conv = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t):
        """
        x: [B, 1024, 16, 16]
        t: [B]
        """
        t_emb = self.time_embed(t)

        x = self.init_conv(x)

        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)

        x = self.mid_conv(x)

        x = self.block3(x, t_emb)
        x = self.block4(x, t_emb)

        x = self.up_conv(x)

        x = self.block5(x, t_emb)
        x = self.block6(x, t_emb)

        return self.final_conv(x)


# ============================================================
# Latent Diffusion Model
# ============================================================

class LatentDiffusion(nn.Module):
    def __init__(
        self,
        channels=1024,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02
    ):
        super().__init__()

        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)


        self.unet = DiffusionUNet(
            in_channels=channels,
            base_channels=512,
            time_dim=512
        )
        
        # ADD: EMA for diffusion UNet
        self.use_ema = False
        self.ema_unet = EMA(self.unet, decay=0.999)

    
    # ADD
    def update_ema(self):
        """Update EMA after optimizer step"""
        self.ema_unet.update()

    @contextlib.contextmanager
    def ema_scope(self):
        """
        Swap UNet weights with EMA weights for evaluation
        """
        # backup current weights
        unet_backup = deepcopy(self.unet.state_dict())

        # load EMA (shadow) weights
        self.unet.load_state_dict(self.ema_unet.shadow.state_dict())

        try:
            yield
        finally:
            # restore original weights
            self.unet.load_state_dict(unet_backup)


    def q_sample(self, x0, t, noise):
        """
        Forward diffusion q(x_t | x_0)
        """
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)

        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    def compute_loss(self, e3):
        """
        Training loss (NORMAL data only)
        """
        B = e3.size(0)
        device = e3.device

        t = torch.randint(
            0, self.timesteps,
            (B,),
            device=device,
            dtype=torch.long
)

        noise = torch.randn_like(e3)

        e3_t = self.q_sample(e3, t, noise)
        # MODIFY
        if self.use_ema:
            noise_pred = self.unet(e3_t, t)
        else:
            noise_pred = self.unet(e3_t, t)

        #loss = F.mse_loss(noise_pred, noise)
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        return loss

    @torch.no_grad()
    def compute_anomaly_score(self, e3, t_list=[200, 400, 600, 800]):
        """
        Diffusion anomaly score
        """
        device = e3.device
        scores = []

        for t_val in t_list:
            t = torch.full((e3.size(0),), t_val, device=device, dtype=torch.long)
            noise = torch.randn_like(e3)
            e3_t = self.q_sample(e3, t, noise)
            # MODIFY
            if self.use_ema:
                noise_pred = self.unet(e3_t, t)
            else:
                noise_pred = self.unet(e3_t, t)

            err = F.mse_loss(noise_pred, noise, reduction='none')
            err = err.mean(dim=(1, 2, 3))
            scores.append(err)

        return torch.stack(scores, dim=0).mean(dim=0)
