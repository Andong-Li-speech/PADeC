import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm import *


class GaussianFourierProjection(nn.Module):
   """Gaussian Fourier embeddings for noise levels."""

   def __init__(self, embedding_size=256, scale=16.0):
      super().__init__()
      self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
      self.mlp = nn.Sequential(
         nn.Linear(embedding_size * 2, embedding_size * 2, bias=True),
         nn.SiLU(),
         nn.Linear(embedding_size * 2, embedding_size, bias=True),
      )

   def forward(self, x):
      x_proj = torch.log(x[:, None]) * self.W[None, :] * 2 * np.pi  # 这里要取log
      x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
      x = self.mlp(x)
      return x


class PositionalTimestepEmbedder(nn.Module):
   """
   Embeds scalar timesteps into vector representations.
   """
   def __init__(self, 
                hidden_size, 
                frequency_embedding_size=256, 
                pe_type="positional",
                scale=1000,
                out_size=None):
      super().__init__()
      if out_size is None:
         out_size = hidden_size
      self.mlp = nn.Sequential(
         nn.Linear(frequency_embedding_size, hidden_size, bias=True),
         nn.SiLU(),
         nn.Linear(hidden_size, out_size, bias=True),
      )
      self.frequency_embedding_size = frequency_embedding_size
      self.scale = scale

   def forward(self, t):
      t_freq = timestep_embedding(t, self.frequency_embedding_size, scale=self.scale).type(
         self.mlp[0].weight.dtype)
      t_emb = self.mlp(t_freq)
      return t_emb


def timestep_embedding(timesteps, dim, max_period=10000, scale=1000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None] * scale  # 通过scale将timestep从[0, 1]放大
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Conv2FormerUnit(nn.Module):
   def __init__(self,
                input_channel: int,
                hidden_channel: int,
                f_kernel_size: int,
                t_kernel_size: int,
                mlp_ratio: int = 1,
                act_type: str = "gelu",
                use_band_attention: bool = False,
               ):
      super(Conv2FormerUnit, self).__init__()
      self.input_channel = input_channel
      self.hidden_channel = hidden_channel
      self.f_kernel_size = f_kernel_size
      self.t_kernel_size = t_kernel_size
      self.mlp_ratio = mlp_ratio
      self.act_type = act_type
      self.use_band_attention = use_band_attention
      #
      self.norm1 = ChannelNormalization(self.input_channel, ndim=4)
      self.act = self.set_act_layer()
      
      pad_ = nn.ConstantPad2d([t_kernel_size//2, t_kernel_size//2, f_kernel_size//2, f_kernel_size//2], value=0.)
      self.attn = nn.Sequential(
         nn.Conv2d(self.input_channel, self.hidden_channel, 1),
         self.act,
         pad_,
         nn.Conv2d(self.hidden_channel, self.hidden_channel, kernel_size=(self.f_kernel_size, self.t_kernel_size), groups=self.hidden_channel)
      )
      self.v = nn.Conv2d(self.input_channel, self.hidden_channel, 1)
      self.proj = nn.Conv2d(self.hidden_channel, self.input_channel, 1)
      
      if self.use_band_attention:
         self.band_shuffle = BandAttention(input_size=self.input_channel,
                                           hidden_size=self.hidden_channel,
                                           )
      
      # Feedforward
      self.norm2 = ChannelNormalization(self.input_channel, ndim=4)
      self.fc1 = nn.Sequential(
         nn.Conv2d(self.input_channel, self.input_channel * self.mlp_ratio, 1),
         self.act
      )
      
      if self.f_kernel_size == 1:
         if self.t_kernel_size == 1:
            self.dw_conv = nn.Sequential(
            nn.Conv2d(self.input_channel * self.mlp_ratio, self.input_channel * self.mlp_ratio, (1, 1), groups=self.input_channel * self.mlp_ratio),
            self.act
            )
         else:    
            pad_ = nn.ConstantPad2d([1, 1, 0, 0], value=0.)
            self.dw_conv = nn.Sequential(
               pad_,
               nn.Conv2d(self.input_channel * self.mlp_ratio, self.input_channel * self.mlp_ratio, (1, 3), groups=self.input_channel * self.mlp_ratio),
               self.act
            )
      else:
         if self.t_kernel_size == 1:
            pad_ = nn.ConstantPad2d([0, 0, 1, 1], value=0.)
            self.dw_conv = nn.Sequential(
               pad_,
               nn.Conv2d(self.input_channel * self.mlp_ratio, self.input_channel * self.mlp_ratio, (3, 1), groups=self.input_channel * self.mlp_ratio),
               self.act
            )
         else:  
            pad_ = nn.ConstantPad2d([1, 1, 1, 1], value=0.)
            self.dw_conv = nn.Sequential(
               pad_,
               nn.Conv2d(self.input_channel * self.mlp_ratio, self.input_channel * self.mlp_ratio, (3, 3), groups=self.input_channel * self.mlp_ratio),
               self.act
            )
      self.fc2 = nn.Conv2d(self.input_channel * self.mlp_ratio, self.input_channel, 1)

   def set_act_layer(self):
      if self.act_type.lower() == "relu":
         return nn.ReLU()
      elif self.act_type.lower() == "silu":
         return nn.SiLU()
      elif self.act_type.lower() == "gelu":
         return nn.GELU()

   def forward(self, x):
      """
      x: (B, C, nband, T)
      """
      # conv-attention
      x_res = x
      x = self.norm1(x)
      x = self.attn(x) * self.v(x)
      x = self.proj(x)
      x = x_res + x
      
      # band shuffle (optional)
      if self.use_band_attention:
         b_size, c, nband, seq_len = x.shape
         x = x.permute(0, 3, 1, 2).contiguous().view(b_size*seq_len, c, nband)
         x, _ = self.band_shuffle(x)
         x = x.view(b_size, seq_len, c, nband).permute(0, 2, 3, 1).contiguous()
      
      # mlp
      x_res = x
      x = self.norm2(x)
      x = self.fc1(x)
      x = x + self.dw_conv(x)
      x = self.fc2(x)
      out = x_res + x

      return out


class HorUnit(nn.Module):
   def __init__(self,
                input_channel: int,
                hidden_channel: int,
                f_kernel_size: int,
                t_kernel_size: int,
                mlp_ratio: int,
                act_type: str = "gelu",
                order: int = 4,
                causal: bool = False,
                use_band_attention: bool = False,
                ):
      super(HorUnit, self).__init__()
      self.input_channel = input_channel
      self.hidden_channel = hidden_channel
      self.order = order
      self.f_kernel_size = f_kernel_size
      self.t_kernel_size = t_kernel_size
      self.mlp_ratio = mlp_ratio
      self.act_type = act_type
      self.causal = causal
      self.use_band_attention = use_band_attention
      
      #
      self.norm1 = ChannelNormalization(self.input_channel, ndim=4)
      if causal:
         pad_ = nn.ConstantPad2d((self.t_kernel_size-1, 0, self.f_kernel_size//2, self.f_kernel_size//2), value=0.)
      else:
         pad_ = nn.ConstantPad2d((self.t_kernel_size//2, self.t_kernel_size//2, self.f_kernel_size//2, self.f_kernel_size//2), value=0.)

      self.dims = [self.input_channel // 2 ** i for i in range(self.order)]
      self.dims.reverse()
      self.proj_in = nn.Conv2d(self.input_channel, 2 * self.hidden_channel, 1)
      self.scale = 1 / 3
      self.dw_conv1 = nn.Sequential(
         pad_,
         nn.Conv2d(sum(self.dims), sum(self.dims), (self.f_kernel_size, self.t_kernel_size), groups=sum(self.dims))
      )
      self.proj_out = nn.Conv2d(self.hidden_channel, self.input_channel, 1)
      
      self.pws = nn.ModuleList(
         [nn.Conv2d(self.dims[i], self.dims[i+1], 1) for i in range(order - 1)]
      )

      self.act = self.set_act_layer()
      
      # 
      if self.use_band_attention:
         self.band_shuffle = BandAttention(input_size=self.input_channel,
                                           hidden_size=self.hidden_channel,
                                           num_head=self.hidden_channel // 64,
                                           )
      
      # Feedforward
      self.norm2 = ChannelNormalization(self.input_channel, ndim=4)
      self.fc1 = nn.Sequential(
         nn.Conv2d(self.hidden_channel, self.hidden_channel * self.mlp_ratio, 1),
         self.act
      )
      if self.causal:
         pad_ = nn.ConstantPad2d([2, 0, 1, 1], value=0.)
      else:
         pad_ = nn.ConstantPad2d([1, 1, 1, 1], value=0.)
      self.dw_conv2 = nn.Sequential(
         pad_,
         nn.Conv2d(self.hidden_channel * self.mlp_ratio, self.hidden_channel * self.mlp_ratio, 3, groups=self.hidden_channel * self.mlp_ratio),
         self.act
      )
      self.fc2 = nn.Conv2d(self.hidden_channel * self.mlp_ratio, self.input_channel, 1)

   def set_act_layer(self):
      if self.act_type.lower() == "relu":
         return nn.ReLU()
      elif self.act_type.lower() == "silu":
         return nn.SiLU()
      elif self.act_type.lower() == "gelu":
         return nn.GELU()

   def forward(self, x):
      """
      x: (B, C, nband, T)
      """
      x_res = x   
      x = self.norm1(x)
      fused_x = self.proj_in(x)
      pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
      dw_abc = self.dw_conv1(abc)
      
      dw_list = torch.split(dw_abc, self.dims, dim=1)
      x = pwa * dw_list[0]
      
      for i in range(self.order - 1):
         x = self.pws[i](x) * dw_list[i + 1]
      x = x_res + self.proj_out(x)
      
      # band shuffle (optional)
      if self.use_band_attention:
         b_size, c, nband, seq_len = x.shape
         x = x.permute(0, 3, 1, 2).contiguous().view(b_size*seq_len, c, nband)
         x, _ = self.band_shuffle(x)
         x = x.view(b_size, seq_len, c, nband).permute(0, 2, 3, 1).contiguous()
      
      # mlp
      x_res = x
      x = self.norm2(x)
      x = self.fc1(x)
      x = x + self.dw_conv2(x)
      x = self.fc2(x)
      out = x_res + x
      
      return out


class BandAttention(nn.Module):
   """
   Transformer with rotary positional embedding.
   """
   def __init__(self, input_size, hidden_size, num_head=4, theta=10000, window=10000, 
               input_drop=0., attention_drop=0., causal=False):
      super().__init__()

      self.input_size = input_size
      self.hidden_size = hidden_size // num_head
      self.num_head = num_head
      self.theta = theta  # base frequency for RoPE
      self.window = window
      # pre-calculate rotary embeddings
      cos_freq, sin_freq = self._calc_rotary_emb()
      self.register_buffer("cos_freq", cos_freq)  # win, N
      self.register_buffer("sin_freq", sin_freq)  # win, N
      
      self.attention_drop = attention_drop
      self.causal = causal
      self.eps = 1e-5

      self.input_norm = RMSNorm(self.input_size)
      self.input_drop = nn.Dropout(p=input_drop)
      self.weight = nn.Conv1d(self.input_size, self.hidden_size*self.num_head*3, 1, bias=False)
      self.output = nn.Conv1d(self.hidden_size*self.num_head, self.input_size, 1, bias=False)

   def _calc_rotary_emb(self):
      freq = 1. / (self.theta ** (torch.arange(0, self.hidden_size, 2)[:(self.hidden_size // 2)] / self.hidden_size))  # theta_i
      freq = freq.reshape(1, -1)  # 1, N//2
      pos = torch.arange(0, self.window).reshape(-1, 1)  # win, 1
      cos_freq = torch.cos(pos*freq)  # win, N//2
      sin_freq = torch.sin(pos*freq)  # win, N//2
      cos_freq = torch.stack([cos_freq]*2, -1).reshape(self.window, self.hidden_size)  # win, N
      sin_freq = torch.stack([sin_freq]*2, -1).reshape(self.window, self.hidden_size)  # win, N

      return cos_freq, sin_freq
   
   def _add_rotary_emb(self, feature, pos):
      # feature shape: ..., N
      N = feature.shape[-1]

      feature_reshape = feature.reshape(-1, N)
      pos = min(pos, self.window-1)
      cos_freq = self.cos_freq[pos]
      sin_freq = self.sin_freq[pos]
      reverse_sign = torch.from_numpy(np.asarray([-1, 1])).to(feature.device).type(feature.dtype)
      feature_reshape_neg = (torch.flip(feature_reshape.reshape(-1, N//2, 2), [-1]) * reverse_sign.reshape(1, 1, 2)).reshape(-1, N)
      feature_rope = feature_reshape * cos_freq.unsqueeze(0) + feature_reshape_neg * sin_freq.unsqueeze(0)
   
      return feature_rope.reshape(feature.shape)

   def _add_rotary_sequence(self, feature):
      # feature shape: ..., T, N
      T, N = feature.shape[-2:]
      feature_reshape = feature.reshape(-1, T, N)

      cos_freq = self.cos_freq[:T]
      sin_freq = self.sin_freq[:T]
      reverse_sign = torch.from_numpy(np.asarray([-1, 1])).to(feature.device).type(feature.dtype)
      feature_reshape_neg = (torch.flip(feature_reshape.reshape(-1, N//2, 2), [-1]) * reverse_sign.reshape(1, 1, 2)).reshape(-1, T, N)
      feature_rope = feature_reshape * cos_freq.unsqueeze(0) + feature_reshape_neg * sin_freq.unsqueeze(0)
   
      return feature_rope.reshape(feature.shape)
   
   def forward(self, input):
      # input shape: B*T, C, nband

      B, _, T = input.shape

      weight = self.weight(self.input_drop(self.input_norm(input))).reshape(B, self.num_head, self.hidden_size*3, T).mT
      Q, K, V = torch.split(weight, self.hidden_size, dim=-1)  # B, num_head, T, N
      
      # rotary positional embedding
      Q_rot = self._add_rotary_sequence(Q)
      K_rot = self._add_rotary_sequence(K)

      attention_output = F.scaled_dot_product_attention(Q_rot.contiguous(), K_rot.contiguous(), V.contiguous(), dropout_p=self.attention_drop, is_causal=self.causal)  # B, num_head, T, N
      attention_output = attention_output.mT.reshape(B, -1, T)
      output = self.output(attention_output) + input

      return output, (K_rot, V)