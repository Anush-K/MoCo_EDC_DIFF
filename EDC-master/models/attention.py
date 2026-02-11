# # # models/attention.py
# # import torch
# # import torch.nn as nn

# # class SelfAttention2d(nn.Module):
# #     """
# #     Simple self-attention for (B, C, H, W) feature maps using nn.MultiheadAttention.
# #     Operates by flattening spatial dims -> (B, N, C) and applying MHA with residual + LayerNorm.
# #     """
# #     def __init__(self, channels, num_heads=8, dropout=0.0):
# #         super().__init__()
# #         assert channels % num_heads == 0, "channels must be divisible by num_heads"
# #         self.channels = channels
# #         self.num_heads = num_heads

# #         self.norm = nn.LayerNorm(channels)
# #         self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads,
# #                                          dropout=dropout, batch_first=True)
# #         self.ff = nn.Sequential(
# #             nn.LayerNorm(channels),
# #             nn.Linear(channels, channels),
# #             nn.ReLU(inplace=True),
# #             nn.Linear(channels, channels),
# #         )

# #     def forward(self, x):
# #         # x: (B, C, H, W) -> out: same shape
# #         B, C, H, W = x.shape
# #         n = H * W
# #         x_flat = x.permute(0, 2, 3, 1).reshape(B, n, C)  # (B, N, C)
# #         x_norm = self.norm(x_flat)

# #         attn_out, _ = self.mha(x_norm, x_norm, x_norm)  # (B, N, C)
# #         x_res = x_flat + attn_out   # residual

# #         ff_out = self.ff(x_res) + x_res
# #         out = ff_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
# #         return out


# # class CrossAttention2d(nn.Module):
# #     """
# #     Cross-attention between two 2D feature maps.
# #     Query from 'q_feat' (e.g. decoder), key/value from 'kv_feat' (e.g. encoder).
# #     Input shapes: q_feat, kv_feat are (B, C_q, H, W), (B, C_kv, H_kv, W_kv)
# #     This module projects kv to same channel dim as q if needed.
# #     """
# #     def __init__(self, q_channels, kv_channels, num_heads=8, dropout=0.0):
# #         super().__init__()
# #         # we project kv to q_channels if different
# #         self.qc = q_channels
# #         self.kvc = kv_channels
# #         # choose heads consistent with q_channels
# #         assert q_channels % num_heads == 0, "q_channels must be divisible by num_heads"
# #         self.num_heads = num_heads

# #         self.norm_q = nn.LayerNorm(q_channels)
# #         self.norm_kv = nn.LayerNorm(kv_channels)

# #         if kv_channels != q_channels:
# #             self.kv_proj = nn.Linear(kv_channels, q_channels)
# #         else:
# #             self.kv_proj = None

# #         self.mha = nn.MultiheadAttention(embed_dim=q_channels, num_heads=num_heads,
# #                                          dropout=dropout, batch_first=True)

# #         self.ff = nn.Sequential(
# #             nn.LayerNorm(q_channels),
# #             nn.Linear(q_channels, q_channels),
# #             nn.ReLU(inplace=True),
# #             nn.Linear(q_channels, q_channels),
# #         )

# #     def forward(self, q_feat, kv_feat):
# #         # q_feat: (B, Cq, Hq, Wq) ; kv_feat: (B, Ckv, Hkv, Wkv)
# #         B, Cq, Hq, Wq = q_feat.shape
# #         _, Ckv, Hkv, Wkv = kv_feat.shape

# #         q_n = Hq * Wq
# #         kv_n = Hkv * Wkv

# #         q_flat = q_feat.permute(0, 2, 3, 1).reshape(B, q_n, Cq)     # (B, Nq, Cq)
# #         kv_flat = kv_feat.permute(0, 2, 3, 1).reshape(B, kv_n, Ckv)  # (B, Nk, Ckv)

# #         q_norm = self.norm_q(q_flat)
# #         kv_norm = self.norm_kv(kv_flat)

# #         if self.kv_proj is not None:
# #             kv_norm = self.kv_proj(kv_norm)  # (B, Nk, Cq)

# #         attn_out, _ = self.mha(q_norm, kv_norm, kv_norm)  # (B, Nq, Cq)
# #         q_res = q_flat + attn_out
# #         ff_out = self.ff(q_res) + q_res

# #         out = ff_out.reshape(B, Hq, Wq, Cq).permute(0, 3, 1, 2).contiguous()
# #         return out


# # models/attention.py  -- MPS-friendly conv-based attention
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# class SelfAttention2d(nn.Module):
#     """
#     MPS-friendly self-attention for (B,C,H,W).
#     Uses conv1x1 for Q,K,V, scaled dot-product and feed-forward projection.
#     Avoids nn.MultiheadAttention to stay on-device.
#     """
#     def __init__(self, channels, num_heads=8, head_dim=None, dropout=0.0):
#         super().__init__()
#         assert channels % num_heads == 0, "channels must be divisible by num_heads"
#         self.C = channels
#         self.num_heads = num_heads
#         self.head_dim = channels // num_heads

#         self.qkv_proj = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=True)
#         self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

#         # small FFN for locality (1x1 conv)
#         self.ffn = nn.Sequential(
#             nn.Conv2d(channels, channels, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, 1)
#         )

#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         # x: (B, C, H, W)
#         B, C, H, W = x.shape
#         n = H * W

#         qkv = self.qkv_proj(x)  # (B, 3C, H, W)
#         q, k, v = torch.chunk(qkv, 3, dim=1)  # each (B, C, H, W)

#         # reshape into (B, heads, head_dim, N)
#         q = q.reshape(B, self.num_heads, self.head_dim, n).permute(0,1,3,2)  # (B, H, N, Dh)
#         k = k.reshape(B, self.num_heads, self.head_dim, n).permute(0,1,2,3)  # (B, H, Dh, N)
#         v = v.reshape(B, self.num_heads, self.head_dim, n).permute(0,1,3,2)  # (B, H, N, Dh)

#         # scaled dot-product: (B, H, N, N)
#         scale = 1.0 / math.sqrt(self.head_dim)
#         attn = torch.matmul(q, k) * scale  # (B, H, N, N)
#         attn = F.softmax(attn, dim=-1)
#         attn = self.dropout(attn)

#         out = torch.matmul(attn, v)  # (B, H, N, Dh)
#         out = out.permute(0,1,3,2).reshape(B, C, H, W)  # (B, C, H, W)

#         out = self.out_proj(out)
#         out = out + x  # residual
#         out = out + self.ffn(out)
#         return out


# class CrossAttention2d(nn.Module):
#     """
#     MPS-friendly cross-attention: query from q_feat (decoder), key/value from kv_feat (encoder).
#     Projects kv to q-channels if required using conv.
#     """
#     def __init__(self, q_channels, kv_channels, num_heads=8, head_dim=None, dropout=0.0):
#         super().__init__()
#         assert q_channels % num_heads == 0, "q_channels must be divisible by num_heads"
#         self.qc = q_channels
#         self.kvc = kv_channels
#         self.num_heads = num_heads
#         self.head_dim = q_channels // num_heads

#         self.q_proj = nn.Conv2d(q_channels, q_channels, kernel_size=1, bias=True)
#         self.kv_proj = nn.Conv2d(kv_channels, q_channels * 2, kernel_size=1, bias=True)  # k and v
#         self.out_proj = nn.Conv2d(q_channels, q_channels, kernel_size=1, bias=True)

#         self.ffn = nn.Sequential(
#             nn.Conv2d(q_channels, q_channels, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(q_channels, q_channels, 1),
#         )
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, q_feat, kv_feat):
#         # q_feat: (B, Cq, Hq, Wq) ; kv_feat: (B, Ckv, Hkv, Wkv)
#         B, Cq, Hq, Wq = q_feat.shape
#         _, Ckv, Hkv, Wkv = kv_feat.shape
#         Nq = Hq * Wq
#         Nk = Hkv * Wkv

#         q = self.q_proj(q_feat).reshape(B, self.num_heads, self.head_dim, Nq).permute(0,1,3,2)  # (B,H,Nq,D)
#         kv = self.kv_proj(kv_feat)  # (B, 2*Cq, Hkv, Wkv)
#         k, v = torch.chunk(kv, 2, dim=1)
#         k = k.reshape(B, self.num_heads, self.head_dim, Nk)  # (B,H,D,Nk)
#         v = v.reshape(B, self.num_heads, self.head_dim, Nk).permute(0,1,3,2)  # (B,H,Nk,D)

#         scale = 1.0 / math.sqrt(self.head_dim)
#         attn = torch.matmul(q, k) * scale   # (B,H,Nq,Nk)
#         attn = F.softmax(attn, dim=-1)
#         attn = self.dropout(attn)

#         out = torch.matmul(attn, v)  # (B,H,Nq,D)
#         out = out.permute(0,1,3,2).reshape(B, Cq, Hq, Wq)
#         out = self.out_proj(out)
#         out = out + q_feat  # residual
#         out = out + self.ffn(out)
#         return out
