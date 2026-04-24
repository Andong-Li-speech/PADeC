import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import init

from .norm import *


class HorUnit(nn.Module):
   def __init__(self,
                nb_num: int,
                input_channel: int,
                hidden_channel: int,
                order: int,
                f_kernel_size: int,
                t_kernel_size: int,
                mlp_ratio: int,
                act_type: str = "gelu",
                causal: bool = False,
                ):
      super(HorUnit, self).__init__()
      self.nb_num = nb_num
      self.input_channel = input_channel
      self.hidden_channel = hidden_channel
      self.order = order
      self.f_kernel_size = f_kernel_size
      self.t_kernel_size = t_kernel_size
      self.mlp_ratio = mlp_ratio
      self.act_type = act_type
      self.causal = causal

      #
      self.norm1 = BandwiseC2LayerNorm(nband=self.nb_num, feature_dim=self.input_channel)

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
      # FFN
      self.act = self.set_act_layer()
      # Feedforward
      self.norm2 = BandwiseC2LayerNorm(self.nb_num, self.hidden_channel)
      # self.norm2 = ChannelNormalization(self.input_channel, ndim=4)
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
      time_token: (B, C)
      time_ada: (B, C) or None
      return: (B, C, nband, T)
      """
      # Horn
      x_res = x
      x = self.norm1(x)
      fused_x = self.proj_in(x)
      pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
      dw_abc = self.dw_conv1(abc)
      
      dw_list = torch.split(dw_abc, self.dims, dim=1)
      x = pwa * dw_list[0]
      
      for i in range(self.order - 1):
         x = self.pws[i](x) * dw_list[i + 1]
      x = self.proj_out(x) + x_res

      # FFN
      x_res = x
      x = self.norm2(x)
      x = self.fc1(x)
      x = x + self.dw_conv2(x)
      x = self.fc2(x)
      out = x_res + x

      return out


class Conv2FormerModule(nn.Module):
   def __init__(self,
                nband: int,
                input_channel: int,
                hidden_channel: int,
                f_kernel_size: int,
                t_kernel_size: int,
                mlp_ratio: int = 1,
                causal: bool = False,
                ):
      super(Conv2FormerModule, self).__init__()
      self.nband = nband
      self.input_channel = input_channel
      self.hidden_channel = hidden_channel
      self.f_kernel_size = f_kernel_size
      self.t_kernel_size = t_kernel_size
      self.mlp_ratio = mlp_ratio
      self.causal = causal
      if self.causal:
         pad_ = nn.ConstantPad2d([t_kernel_size-1, 0, f_kernel_size//2, f_kernel_size//2], value=0.)
      else:
         pad_ = nn.ConstantPad2d([t_kernel_size//2, t_kernel_size//2, f_kernel_size//2, f_kernel_size//2], value=0.)
      # spatial attention
      self.attn = nn.Sequential(
         BandwiseC2LayerNorm(nband=nband, feature_dim=self.input_channel),
         nn.Conv2d(self.input_channel, self.hidden_channel, 1),
         nn.GELU(),
         pad_,
         nn.Conv2d(self.hidden_channel, self.hidden_channel, kernel_size=(f_kernel_size, t_kernel_size), groups=self.hidden_channel)
      )
      self.v = nn.Conv2d(self.input_channel, self.hidden_channel, 1)
      self.proj = nn.Conv2d(self.hidden_channel, self.input_channel, 1)

      # Feedforward
      self.fc1 = nn.Sequential(
         BandwiseC2LayerNorm(nband=nband, feature_dim=self.input_channel),
         nn.Conv2d(self.input_channel, self.input_channel * self.mlp_ratio, 1),
         nn.GELU()
      )
      if self.causal:
         pad_ = nn.ConstantPad2d([2, 0, 1, 1], value=0.)
      else:
         pad_ = nn.ConstantPad2d([1, 1, 0, 0], value=0.)
      self.dw_conv = nn.Sequential(
         pad_,
         nn.Conv2d(self.input_channel * self.mlp_ratio, self.input_channel * self.mlp_ratio, 3, groups=self.input_channel * self.mlp_ratio),
         nn.GELU()
      )
      self.fc2 =  nn.Conv2d(self.input_channel * self.mlp_ratio, self.input_channel, 1)

   def forward(self, x):
      """
      inpt: (B, C, nband, T)
      return: (B, C, nband, T)
      """
      # attn
      x_res = x
      x = self.attn(x) * self.v(x)
      x = self.proj(x)
      x = x_res + x
      # mlp
      x_res = x
      x = self.fc1(x)
      x = x + self.dw_conv(x)
      x = self.fc2(x)
      x = x_res + x
      return x


class ConvNextV2(nn.Module):
   def __init__(self,
                input_channel: int,
                hidden_channel: int,
                f_kernel_size: int,
                t_kernel_size: int,
                causal: bool = False,
                ):
      super(ConvNextV2, self).__init__()
      self.input_channel = input_channel
      self.hidden_channel = hidden_channel
      self.f_kernel_size = f_kernel_size
      self.t_kernel_size = t_kernel_size
      self.causal = causal
      #
      if self.causal:
         pad_ = nn.ConstantPad2d([t_kernel_size-1, 0, f_kernel_size//2, f_kernel_size//2], value=0.)
      else:
         pad_ = nn.ConstantPad2d([t_kernel_size//2, t_kernel_size//2, f_kernel_size//2, f_kernel_size//2], value=0.)
      self.dwconv = nn.Sequential(
         pad_,
         nn.Conv2d(input_channel, input_channel, kernel_size=(self.f_kernel_size, self.t_kernel_size), groups=input_channel)
      )
      self.norm = nn.LayerNorm(input_channel)
      self.pwconv1 = nn.Linear(input_channel, hidden_channel)
      self.act = nn.GELU()
      self.grn = GRN2d(hidden_channel)
      self.pwconv2 = nn.Linear(hidden_channel, input_channel)
   
   def forward(self, x):
      """
      inpt: (B, C, nband, T)
      return: (B, C, nband, T)
      """
      inpt = x
      x = self.dwconv(x)
      x = x.permute(0, 2, 3, 1)  # (B, C, nband, T)-> (B, nband, T, C)
      x = self.norm(x)
      x = self.pwconv1(x)
      x = self.act(x)
      x = self.grn(x)
      x = self.pwconv2(x)
      x = x.permute(0, 3, 1, 2).contiguous()
      x = inpt + x
      return x


class GRN(nn.Module):
   """GRN (Global Response Normalization) layer"""

   def __init__(self, dim):
      super().__init__()
      self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
      self.beta = nn.Parameter(torch.zeros(1, dim, 1))

   def forward(self, x):
      Gx = torch.norm(x, p=2, dim=-1, keepdim=True)
      Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
      return self.gamma * (x * Nx) + self.beta + x


class GRN2d(nn.Module):
   def __init__(self, dim):
      super(GRN2d, self).__init__()
      self.dim = dim
      self.gamma = nn.Parameter(torch.zeros(1, 1, 1, self.dim))
      self.beta = nn.Parameter(torch.zeros(1, 1, 1, self.dim))

   def forward(self, x):
      """
      (B, nband, T, C)
      return: (B, nband, T, C)
      """
      Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
      Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
      return self.gamma * (x * Nx) + self.beta + x
  

class LinearGroup(nn.Module):
   def __init__(self, in_features: int, out_features: int, num_groups: int, bias: bool = True):
      super(LinearGroup, self).__init__()
      self.in_features = in_features
      self.out_features = out_features
      self.num_groups = num_groups
      self.weight = Parameter(torch.empty([num_groups, out_features, in_features]))
      if bias:
         self.bias = Parameter(torch.empty([num_groups, out_features]))
      else:
         self.register_parameter('bias', None)
      self.reset_parameters()
   
   def reset_parameters(self):
      init.kaiming_uniform_(self.weight, a=math.sqrt(5))
      if self.bias is not None:
         fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
         init.uniform_(self.bias, -bound, bound)

   def forward(self, input):
      """
      input: (BT, G, nband)
      return: (BT, G, nband)
      """
      x = torch.einsum('...gh,gkh->...gk', [input, self.weight])
      if self.bias is not None:
         x = x + self.bias[None, ...]

      return x


class BandSplit_22k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int,
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64,
               ):
      super(BandSplit_22k, self).__init__()
      self.sr = sr      
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.eps = torch.finfo(torch.float32).eps

      fft_reso = sr / n_fft
      bw_250 = int(np.floor(250 / fft_reso))  # 11 bands
      bw_500 = int(np.floor(500 / fft_reso))  # 23 bands
      bw_1k = int(np.floor(1000 / fft_reso))  # 46 bands

      # total 24 bands
      self.band_width = [bw_250] * 12  # 3k  0~11
      self.band_width += [bw_500] * 8  # 4k  12~19
      self.band_width += [bw_1k] * 3  # 3k   20~22
      self.band_width.append(self.n_fft // 2 + 1 - np.sum(self.band_width))  # remains

      self.nband = len(self.band_width)
      print(f'Totally splitting {len(self.band_width)} bands for sampling rate: 22.05k.')
      
      self.encoder = nn.ModuleList([])
      for i in range(self.nband):
            self.encoder.append(
               nn.Sequential(
                  ChannelNormalization(self.band_width[i] * 2 + 1),
                  nn.Conv1d(self.band_width[i] * 2 + 1, self.feature_dim, 1)
               )
            )

   def get_nband(self):
      return self.nband

   def forward(self, input=None):
      """
      input: (B, F, T, 2)
      log_input: (B, F, T)
      return: (B, nband, C, T)
      """
      subband_spec_list = []
      band_idx = 0
      for i in range(len(self.band_width)):
            cur_subband_spec = input[:, band_idx: band_idx + self.band_width[i]].transpose(1, 2).contiguous() # (B, T, fw, 2)
            cur_subband_spec_power = torch.sqrt(torch.norm(cur_subband_spec, dim=-1, keepdim=True).pow(2).sum(dim=-2, keepdim=True) + self.eps)  # (B, T, 1, 1)
            b_size, seq_len, _, _ = cur_subband_spec.shape
            cur_subband_spec_ = (cur_subband_spec / cur_subband_spec_power).view(b_size, seq_len, -1)
            cur_subband_spec_ = torch.cat([cur_subband_spec_, torch.log(cur_subband_spec_power.squeeze(-1))], dim=-1).transpose(-2, -1).contiguous()
            subband_spec_list.append(self.encoder[i](cur_subband_spec_))  
            band_idx += self.band_width[i]

      out = torch.stack(subband_spec_list, dim=1)  # (B, nband, C, T)

      return out


class BandSplit_24k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int,
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64,
               ):
      super(BandSplit_24k, self).__init__()
      self.sr = sr      
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.eps = torch.finfo(torch.float32).eps

      fft_reso = sr / n_fft
      bw_250 = int(np.floor(250 / fft_reso))  # 10 bands
      bw_500 = int(np.floor(500 / fft_reso))  # 21 bands
      bw_1k = int(np.floor(1000 / fft_reso))  # 42 bands

      # total 24 bands
      self.band_width = [12] * 12  # 3k  0~11
      self.band_width += [24] * 8  # 4k  12~19
      self.band_width += [44] * 3  # 4k   20~22
      self.band_width.append(self.n_fft // 2 + 1 - np.sum(self.band_width))
      # self.band_width.append(self.n_fft // 2 + 1 - np.sum(self.band_width))  # remains

      self.nband = len(self.band_width)
      print(f'Totally splitting {len(self.band_width)} bands for sampling rate: 24k.')

      self.encoder = nn.ModuleList([])
      for i in range(self.nband):
            self.encoder.append(
               nn.Sequential(
                  ChannelNormalization(self.band_width[i] * 2),
                  nn.Conv1d(self.band_width[i] * 2, self.feature_dim, 1)
               )
            )

   def get_nband(self):
      return self.nband

   def forward(self, input=None):
      """
      input: (B, F, T, 2)
      log_input: (B, F, T)
      return: (B, C, nband, T)
      """
      b_size, _, seq_len, _ = input.shape
      subband_spec_list = []
      band_idx = 0
      for i in range(len(self.band_width)):
            cur_subband_spec = input[:, band_idx: band_idx + self.band_width[i]].transpose(-2, -1).contiguous().view(b_size, -1, seq_len) # (B, T, fw, 2)
            subband_spec_list.append(self.encoder[i](cur_subband_spec))  
            band_idx += self.band_width[i]

      out = torch.stack(subband_spec_list, dim=-2)  # (B, C, nband, T)

      return out


class SharedBandSplit_NB24_22k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int,
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64,
               ):
      super(SharedBandSplit_NB24_22k, self).__init__()
      self.sr = sr      
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.eps = torch.finfo(torch.float32).eps

      self.reg1_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(12, 3), stride=(12, 1)),
          BandwiseC2LayerNorm(nband=12, feature_dim=self.feature_dim)
      )
      self.reg2_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(24, 3), stride=(24, 1)),
          BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim)
      )
      self.reg3_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(44, 3), stride=(44, 1)),
          BandwiseC2LayerNorm(nband=4, feature_dim=self.feature_dim)
      )

      self.nband = 12 + 8 + 4
      print(f'Totally splitting {self.nband} bands for sampling rate: 22.05k.')

   def get_nband(self):
      return self.nband

   def forward(self, input=None):
      """
      input: (B, F, T, 2)
      log_input: (B, F, T)
      return: (B, nband, C, T)
      """
      input = input.permute(0, 3, 1, 2).contiguous()
      x1, x2, x3 = input[..., :144, :], input[..., 144:336, :], input[..., 336:-1, :]
      y1, y2, y3 = self.reg1_encoder(x1), self.reg2_encoder(x2), self.reg3_encoder(x3)

      out = torch.cat([y1, y2, y3], dim=-2)  # (B, C, nband, C)

      return out
   

class SharedBandSplit_NB24_even_22k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int,
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64,
               ):
      super(SharedBandSplit_NB24_even_22k, self).__init__()
      self.sr = sr      
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.eps = torch.finfo(torch.float32).eps

      self.reg1_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(21, 3), stride=(21, 1)),
          BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim)
      )
      self.reg2_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(21, 3), stride=(21, 1)),
          BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim)
      )
      self.reg3_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(21, 3), stride=(21, 1)),
          BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim)
      )

      self.nband = 8 + 8 + 8
      print(f'Totally evenly splitting {self.nband} bands for sampling rate: 22.05k.')

   def get_nband(self):
      return self.nband

   def forward(self, input=None):
      """
      input: (B, F, T, 2)
      log_input: (B, F, T)
      return: (B, nband, C, T)
      """
      input = input.permute(0, 3, 1, 2).contiguous()
      x1, x2, x3 = input[..., :168, :], input[..., 168:336, :], input[..., 336:-1, :]
      y1, y2, y3 = self.reg1_encoder(x1), self.reg2_encoder(x2), self.reg3_encoder(x3)

      out = torch.cat([y1, y2, y3], dim=-2)  # (B, C, nband, C)

      return out


class SharedBandSplit_NB6_22k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int,
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64,
               ):
      super(SharedBandSplit_NB6_22k, self).__init__()
      self.sr = sr      
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.eps = torch.finfo(torch.float32).eps

      self.reg1_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(48, 3), stride=(48, 1)),
          BandwiseC2LayerNorm(nband=3, feature_dim=self.feature_dim)
      )
      self.reg2_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(96, 3), stride=(96, 1)),
          BandwiseC2LayerNorm(nband=2, feature_dim=self.feature_dim)
      )
      self.reg3_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(176, 3), stride=(176, 1)),
          BandwiseC2LayerNorm(nband=1, feature_dim=self.feature_dim)
      )

      self.nband = 3 + 2 + 1
      print(f'Totally splitting {self.nband} bands for sampling rate: 22.05k.')

   def get_nband(self):
      return self.nband

   def forward(self, input=None):
      """
      input: (B, F, T, 2)
      log_input: (B, F, T)
      return: (B, nband, C, T)
      """
      input = input.permute(0, 3, 1, 2).contiguous()
      x1, x2, x3 = input[..., :144, :], input[..., 144:336, :], input[..., 336:-1, :]
      y1, y2, y3 = self.reg1_encoder(x1), self.reg2_encoder(x2), self.reg3_encoder(x3)

      out = torch.cat([y1, y2, y3], dim=-2)  # (B, C, nband, C)

      return out


class SharedBandSplit_NB12_22k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int,
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64,
               ):
      super(SharedBandSplit_NB12_22k, self).__init__()
      self.sr = sr      
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.eps = torch.finfo(torch.float32).eps

      self.reg1_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(24, 3), stride=(24, 1)),
          BandwiseC2LayerNorm(nband=6, feature_dim=self.feature_dim)
      )
      self.reg2_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(48, 3), stride=(48, 1)),
          BandwiseC2LayerNorm(nband=4, feature_dim=self.feature_dim)
      )
      self.reg3_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(88, 3), stride=(88, 1)),
          BandwiseC2LayerNorm(nband=2, feature_dim=self.feature_dim)
      )

      self.nband = 6 + 4 + 2
      print(f'Totally splitting {self.nband} bands for sampling rate: 22.05k.')

   def get_nband(self):
      return self.nband

   def forward(self, input=None):
      """
      input: (B, F, T, 2)
      log_input: (B, F, T)
      return: (B, nband, C, T)
      """
      input = input.permute(0, 3, 1, 2).contiguous()
      x1, x2, x3 = input[..., :144, :], input[..., 144:336, :], input[..., 336:-1, :]
      y1, y2, y3 = self.reg1_encoder(x1), self.reg2_encoder(x2), self.reg3_encoder(x3)

      out = torch.cat([y1, y2, y3], dim=-2)  # (B, C, nband, C)

      return out


class SharedBandSplit_NB48_22k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int,
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64,
               ):
      super(SharedBandSplit_NB48_22k, self).__init__()
      self.sr = sr      
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.eps = torch.finfo(torch.float32).eps

      self.reg1_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(6, 3), stride=(6, 1)),
          BandwiseC2LayerNorm(nband=24, feature_dim=self.feature_dim)
      )
      self.reg2_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(12, 3), stride=(12, 1)),
          BandwiseC2LayerNorm(nband=16, feature_dim=self.feature_dim)
      )
      self.reg3_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(22, 3), stride=(22, 1)),
          BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim)
      )

      self.nband = 24 + 16 + 8
      print(f'Totally splitting {self.nband} bands for sampling rate: 22.05k.')

   def get_nband(self):
      return self.nband

   def forward(self, input=None):
      """
      input: (B, F, T, 2)
      log_input: (B, F, T)
      return: (B, nband, C, T)
      """
      input = input.permute(0, 3, 1, 2).contiguous()
      x1, x2, x3 = input[..., :144, :], input[..., 144:336, :], input[..., 336:-1, :]
      y1, y2, y3 = self.reg1_encoder(x1), self.reg2_encoder(x2), self.reg3_encoder(x3)

      out = torch.cat([y1, y2, y3], dim=-2)  # (B, C, nband, C)

      return out


class SharedBandSplit_NB96_22k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int,
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64,
               ):
      super(SharedBandSplit_NB96_22k, self).__init__()
      self.sr = sr      
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.eps = torch.finfo(torch.float32).eps

      self.reg1_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(3, 3), stride=(3, 1)),
          BandwiseC2LayerNorm(nband=48, feature_dim=self.feature_dim)
      )
      self.reg2_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(6, 3), stride=(6, 1)),
          BandwiseC2LayerNorm(nband=32, feature_dim=self.feature_dim)
      )
      self.reg3_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(11, 3), stride=(11, 1)),
          BandwiseC2LayerNorm(nband=16, feature_dim=self.feature_dim)
      )

      self.nband = 48 + 32 + 16
      print(f'Totally splitting {self.nband} bands for sampling rate: 22.05k.')

   def get_nband(self):
      return self.nband

   def forward(self, input=None):
      """
      input: (B, F, T, 2)
      log_input: (B, F, T)
      return: (B, nband, C, T)
      """
      input = input.permute(0, 3, 1, 2).contiguous()
      x1, x2, x3 = input[..., :144, :], input[..., 144:336, :], input[..., 336:-1, :]
      y1, y2, y3 = self.reg1_encoder(x1), self.reg2_encoder(x2), self.reg3_encoder(x3)

      out = torch.cat([y1, y2, y3], dim=-2)  # (B, C, nband, C)

      return out


class BandMerge_22k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int, 
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64,
               decode_type: str = 'mag+phase',  
               ):
      super(BandMerge_22k, self).__init__()
      self.sr = sr       
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.decode_type = decode_type
      self.eps = torch.finfo(torch.float32).eps

      fft_reso = sr / n_fft
      bw_250 = int(np.floor(250 / fft_reso))  # 5 bands
      bw_500 = int(np.floor(500 / fft_reso))  # 10 bands
      bw_1k = int(np.floor(1000 / fft_reso))  # 20 bands

      # total 24 bands
      self.band_width = [12] * 12  # 3k
      self.band_width += [24] * 8  # 4k 
      self.band_width += [44] * 3  # 3k
      self.band_width.append(self.n_fft // 2 + 1 - np.sum(self.band_width))  # remains

      self.nband = len(self.band_width)
      print(f'Totally Merge {len(self.band_width)} bands for sampling rate: 22.05k.')
      if decode_type.lower() == 'mag+phase':
         self.decoder_mag, self.decoder_phase = nn.ModuleList([]), nn.ModuleList([])
         for i in range(self.nband):
            self.decoder_mag.append(
               nn.Sequential(
                  ChannelNormalization(self.feature_dim),
                  nn.Conv1d(self.feature_dim, 2 * self.feature_dim, 1),
                  nn.GELU(),
                  nn.Conv1d(2 * self.feature_dim, int(self.band_width[i]), 1),
                  )
               )
            self.decoder_phase.append(
               nn.Sequential(
                  ChannelNormalization(self.feature_dim),
                  nn.Conv1d(self.feature_dim, 2 * self.feature_dim, 1),
                  nn.GELU(),
                  nn.Conv1d(2 * self.feature_dim, int(self.band_width[i]) * 2, 1)
                  )
               )
       
      elif decode_type.lower() == 'phase':
         self.decoder_phase = nn.ModuleList([])
         for i in range(self.nband):
            self.decoder_phase.append(
               nn.Sequential(
                  ChannelNormalization(self.feature_dim),
                  nn.Conv1d(self.feature_dim, 2 * self.feature_dim, 1),
                  nn.GELU(),
                  nn.Conv1d(2 * self.feature_dim, int(self.band_width[i]) * 2, 1)
               )
            )

   def forward(self, emb_input):
      """
      emb_input: (B, nband, C, T)
      return:
         mag: (B, F, T)
         phase: (B, F, T)
      """
      if self.decode_type.lower() == 'mag+phase':
         decode_mag_list, decode_phase_list = [], []
         for i in range(len(self.band_width)):
            # mag
            this_mag = torch.exp(self.decoder_mag[i](emb_input[:, i].contiguous()))
            # phase
            this_comp = self.decoder_phase[i](emb_input[:, i].contiguous())
            this_real, this_imag = this_comp.chunk(2, dim=1)
            this_phase = torch.atan2(this_imag, this_real)

            decode_mag_list.append(this_mag)
            decode_phase_list.append(this_phase)
         mag, phase = torch.cat(decode_mag_list, dim=1), torch.cat(decode_phase_list, dim=1)  # (B, F, T)
         return mag, phase
      elif self.decode_type.lower() == 'phase':
         decode_phase_list = []
         for i in range(len(self.band_width)):
            # phase
            this_comp = self.decoder_phase[i](emb_input[:, i].contiguous())
            this_real, this_imag = this_comp.chunk(2, dim=1)
            this_phase = torch.atan2(this_imag, this_real)
            decode_phase_list.append(this_phase)
         phase = torch.cat(decode_phase_list, dim=1)
         return phase


class BandMerge_24k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int, 
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64,
               decode_type: str = 'mag+phase',  
               ):
      super(BandMerge_24k, self).__init__()
      self.sr = sr       
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.decode_type = decode_type
      self.eps = torch.finfo(torch.float32).eps

      fft_reso = sr / n_fft
      bw_250 = int(np.floor(250 / fft_reso))  # 5 bands
      bw_500 = int(np.floor(500 / fft_reso))  # 10 bands
      bw_1k = int(np.floor(1000 / fft_reso))  # 20 bands

      # total 24 bands
      self.band_width = [12] * 12  # 3k
      self.band_width += [24] * 8  # 4k 
      self.band_width += [44] * 3  # 3k
      self.band_width.append(self.n_fft // 2 + 1 - np.sum(self.band_width))  # remains

      self.nband = len(self.band_width)
      print(f'Totally Merge {len(self.band_width)} bands for sampling rate: 24k.')
      if decode_type.lower() in ["mag+phase", "res+phase"]:
         self.decoder_mag, self.decoder_phase = nn.ModuleList([]), nn.ModuleList([])
         for i in range(self.nband):
            self.decoder_mag.append(
               nn.Sequential(
                  ChannelNormalization(self.feature_dim),
                  nn.Conv1d(self.feature_dim, 2 * self.feature_dim, 1),
                  nn.GELU(),
                  nn.Conv1d(2 * self.feature_dim, int(self.band_width[i]), 1),
                  )
               )
            self.decoder_phase.append(
               nn.Sequential(
                  ChannelNormalization(self.feature_dim),
                  nn.Conv1d(self.feature_dim, 2 * self.feature_dim, 1),
                  nn.GELU(),
                  nn.Conv1d(2 * self.feature_dim, int(self.band_width[i]) * 2, 1)
                  )
               )
       
      elif decode_type.lower() == 'phase':
         self.decoder_phase = nn.ModuleList([])
         for i in range(self.nband):
            self.decoder_phase.append(
               nn.Sequential(
                  ChannelNormalization(self.feature_dim),
                  nn.Conv1d(self.feature_dim, 2 * self.feature_dim, 1),
                  nn.GELU(),
                  nn.Conv1d(2 * self.feature_dim, int(self.band_width[i]) * 2, 1)
               )
            )

   def forward(self, emb_input):
      """
      emb_input: (B, C, nband, T)
      return:
         mag: (B, F, T)
         phase: (B, F, T)
      """
      if self.decode_type.lower() in ["mag+phase", "res+phase"]:
         decode_mag_list, decode_phase_list = [], []
         for i in range(len(self.band_width)):
            # mag
            this_mag = torch.exp(self.decoder_mag[i](emb_input[:, :, i].contiguous()))
            # phase
            this_comp = self.decoder_phase[i](emb_input[:, :, i].contiguous())
            this_real, this_imag = this_comp.chunk(2, dim=1)
            this_phase = torch.atan2(this_imag, this_real)

            decode_mag_list.append(this_mag)
            decode_phase_list.append(this_phase)
         mag, phase = torch.cat(decode_mag_list, dim=1), torch.cat(decode_phase_list, dim=1)  # (B, F, T)
         return mag, phase


class SharedBandMerge_NB24_even_22k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int, 
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64,
               decode_type: str = 'mag+phase',  
               ):
      super(SharedBandMerge_NB24_even_22k, self).__init__()
      self.sr = sr       
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.decode_type = decode_type
      self.eps = torch.finfo(torch.float32).eps
      if self.decode_type.lower() in ["mag+phase", "res+phase"]:
         self.reg1_mag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(21, 1), stride=(21, 1))
         )
         self.reg2_mag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(21, 1), stride=(21, 1))
         )
         self.reg3_mag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(21, 1), stride=(21, 1))
         )
         self.reg1_phase_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=2, kernel_size=(21, 1), stride=(21, 1))
         )
         self.reg2_phase_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=2, kernel_size=(21, 1), stride=(21, 1))
         )
         self.reg3_phase_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=2, kernel_size=(21, 1), stride=(21, 1))
         )
      elif self.decode_type.lower() in ["ri", "res_ri"]:
         pass
         

   def forward(self, emb_input):
      """
      emb_input: (B, C, nband, T)
      return:
         mag: (B, F, T)
         phase: (B, F, T)
      """
      if self.decode_type.lower() in ["mag+phase", "res+phase"]:
         x1, x2, x3 = emb_input[:, :, :8].contiguous(), \
                      emb_input[:, :, 8:16].contiguous(), \
                      emb_input[:, :, 16:].contiguous()
         mag1, mag2, mag3 = self.reg1_mag_decoder(x1), self.reg2_mag_decoder(x2), self.reg3_mag_decoder(x3)
         com1, com2, com3 = self.reg1_phase_decoder(x1), self.reg2_phase_decoder(x2), self.reg3_phase_decoder(x3)
         mag = torch.exp(torch.cat([mag1, mag2, mag3], dim=-2))  # exp operation
         com = torch.cat([com1, com2, com3], dim=-2)
         last_mag, last_com = mag[..., -1, :].unsqueeze(-2).repeat(1, 1, 9, 1).contiguous(), com[..., -1, :].unsqueeze(-2).repeat(1, 1, 9, 1).contiguous()
         mag, com = torch.cat([mag, last_mag], dim=-2), torch.cat([com, last_com], dim=-2)
         pha = torch.atan2(com[:, -1], com[:, 0])
         return mag.squeeze(1), pha
      elif self.decode_type.lower() in ["ri", "res_ri"]:
         x1, x2, x3 = emb_input[:, :, :12].contiguous(), \
                      emb_input[:, :, 12:20].contiguous(), \
                      emb_input[:, :, 20:].contiguous()
         real1, real2, real3 = self.reg1_real_decoder(x1), self.reg2_real_decoder(x2), self.reg3_real_decoder(x3)
         imag1, imag2, imag3 = self.reg1_imag_decoder(x1), self.reg2_imag_decoder(x2), self.reg3_imag_decoder(x3)
         real = torch.cat([real1, real2, real3], dim=-2)
         imag = torch.cat([imag1, imag2, imag3], dim=-2)
         last_real, last_imag = real[..., -1, :].unsqueeze(-2), imag[..., -1, :].unsqueeze(-2)
         real, imag = torch.cat([real, last_real], dim=-2), torch.cat([imag, last_imag], dim=-2)
         return real.squeeze(1), imag.squeeze(1)


class SharedBandMerge_NB24_22k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int, 
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64,
               decode_type: str = 'mag+phase',  
               ):
      super(SharedBandMerge_NB24_22k, self).__init__()
      self.sr = sr       
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.decode_type = decode_type
      self.eps = torch.finfo(torch.float32).eps
      if self.decode_type.lower() in ["mag+phase", "res+phase"]:
         self.reg1_mag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=12, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg2_mag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg3_mag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=4, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(44, 1), stride=(44, 1))
         )
         self.reg1_phase_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=12, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=2, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg2_phase_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=2, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg3_phase_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=4, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=2, kernel_size=(44, 1), stride=(44, 1))
         )
      elif self.decode_type.lower() in ["ri", "res_ri"]:
         self.reg1_real_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=12, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg2_real_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg3_real_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=4, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(44, 1), stride=(44, 1))
         )
         self.reg1_imag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=12, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg2_imag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg3_imag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=4, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(44, 1), stride=(44, 1))
         )

   def forward(self, emb_input):
      """
      emb_input: (B, C, nband, T)
      return:
         mag: (B, F, T)
         phase: (B, F, T)
      """
      if self.decode_type.lower() in ["mag+phase", "res+phase"]:
         x1, x2, x3 = emb_input[:, :, :12].contiguous(), \
                      emb_input[:, :, 12:20].contiguous(), \
                      emb_input[:, :, 20:].contiguous()
         mag1, mag2, mag3 = self.reg1_mag_decoder(x1), self.reg2_mag_decoder(x2), self.reg3_mag_decoder(x3)
         com1, com2, com3 = self.reg1_phase_decoder(x1), self.reg2_phase_decoder(x2), self.reg3_phase_decoder(x3)
         mag = torch.exp(torch.cat([mag1, mag2, mag3], dim=-2))  # exp operation
         com = torch.cat([com1, com2, com3], dim=-2)
         last_mag, last_com = mag[..., -1, :].unsqueeze(-2), com[..., -1, :].unsqueeze(-2)
         mag, com = torch.cat([mag, last_mag], dim=-2), torch.cat([com, last_com], dim=-2)
         pha = torch.atan2(com[:, -1], com[:, 0])
         return mag.squeeze(1), pha
      elif self.decode_type.lower() in ["ri", "res_ri"]:
         x1, x2, x3 = emb_input[:, :, :12].contiguous(), \
                      emb_input[:, :, 12:20].contiguous(), \
                      emb_input[:, :, 20:].contiguous()
         real1, real2, real3 = self.reg1_real_decoder(x1), self.reg2_real_decoder(x2), self.reg3_real_decoder(x3)
         imag1, imag2, imag3 = self.reg1_imag_decoder(x1), self.reg2_imag_decoder(x2), self.reg3_imag_decoder(x3)
         real = torch.cat([real1, real2, real3], dim=-2)
         imag = torch.cat([imag1, imag2, imag3], dim=-2)
         last_real, last_imag = real[..., -1, :].unsqueeze(-2), imag[..., -1, :].unsqueeze(-2)
         real, imag = torch.cat([real, last_real], dim=-2), torch.cat([imag, last_imag], dim=-2)
         return real.squeeze(1), imag.squeeze(1)


class SharedBandMerge_NB6_22k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int, 
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64,
               decode_type: str = 'mag+phase',  
               ):
      super(SharedBandMerge_NB6_22k, self).__init__()
      self.sr = sr       
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.decode_type = decode_type
      self.eps = torch.finfo(torch.float32).eps
      if self.decode_type.lower() in ["mag+phase", "res+phase"]:
         self.reg1_mag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=3, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(48, 1), stride=(48, 1))
         )
         self.reg2_mag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=2, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(96, 1), stride=(96, 1))
         )
         self.reg3_mag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=1, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(176, 1), stride=(176, 1))
         )
         self.reg1_phase_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=3, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=2, kernel_size=(48, 1), stride=(48, 1))
         )
         self.reg2_phase_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=2, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=2, kernel_size=(96, 1), stride=(96, 1))
         )
         self.reg3_phase_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=1, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=2, kernel_size=(176, 1), stride=(176, 1))
         )
      elif self.decode_type.lower() in ["ri", "res_ri"]:
         self.reg1_real_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=12, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg2_real_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg3_real_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=4, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(44, 1), stride=(44, 1))
         )
         self.reg1_imag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=12, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg2_imag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg3_imag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=4, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(44, 1), stride=(44, 1))
         )
         

   def forward(self, emb_input):
      """
      emb_input: (B, C, nband, T)
      return:
         mag: (B, F, T)
         phase: (B, F, T)
      """
      if self.decode_type.lower() in ["mag+phase", "res+phase"]:
         x1, x2, x3 = emb_input[:, :, :3].contiguous(), \
                      emb_input[:, :, 3:5].contiguous(), \
                      emb_input[:, :, 5:].contiguous()
         mag1, mag2, mag3 = self.reg1_mag_decoder(x1), self.reg2_mag_decoder(x2), self.reg3_mag_decoder(x3)
         com1, com2, com3 = self.reg1_phase_decoder(x1), self.reg2_phase_decoder(x2), self.reg3_phase_decoder(x3)
         mag = torch.exp(torch.cat([mag1, mag2, mag3], dim=-2))  # exp operation
         com = torch.cat([com1, com2, com3], dim=-2)
         last_mag, last_com = mag[..., -1, :].unsqueeze(-2), com[..., -1, :].unsqueeze(-2)
         mag, com = torch.cat([mag, last_mag], dim=-2), torch.cat([com, last_com], dim=-2)
         pha = torch.atan2(com[:, -1], com[:, 0])
         return mag.squeeze(1), pha
      elif self.decode_type.lower() in ["ri", "res_ri"]:
         x1, x2, x3 = emb_input[:, :, :12].contiguous(), \
                      emb_input[:, :, 12:20].contiguous(), \
                      emb_input[:, :, 20:].contiguous()
         real1, real2, real3 = self.reg1_real_decoder(x1), self.reg2_real_decoder(x2), self.reg3_real_decoder(x3)
         imag1, imag2, imag3 = self.reg1_imag_decoder(x1), self.reg2_imag_decoder(x2), self.reg3_imag_decoder(x3)
         real = torch.cat([real1, real2, real3], dim=-2)
         imag = torch.cat([imag1, imag2, imag3], dim=-2)
         last_real, last_imag = real[..., -1, :].unsqueeze(-2), imag[..., -1, :].unsqueeze(-2)
         real, imag = torch.cat([real, last_real], dim=-2), torch.cat([imag, last_imag], dim=-2)
         return real.squeeze(1), imag.squeeze(1)


class SharedBandMerge_NB12_22k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int, 
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64,
               decode_type: str = 'mag+phase',  
               ):
      super(SharedBandMerge_NB12_22k, self).__init__()
      self.sr = sr       
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.decode_type = decode_type
      self.eps = torch.finfo(torch.float32).eps
      if self.decode_type.lower() in ["mag+phase", "res+phase"]:
         self.reg1_mag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=6, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg2_mag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=4, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(48, 1), stride=(48, 1))
         )
         self.reg3_mag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=2, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(88, 1), stride=(88, 1))
         )
         self.reg1_phase_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=6, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=2, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg2_phase_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=4, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=2, kernel_size=(48, 1), stride=(48, 1))
         )
         self.reg3_phase_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=2, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=2, kernel_size=(88, 1), stride=(88, 1))
         )
      elif self.decode_type.lower() in ["ri", "res_ri"]:
         self.reg1_real_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=12, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg2_real_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg3_real_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=4, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(44, 1), stride=(44, 1))
         )
         self.reg1_imag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=12, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg2_imag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg3_imag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=4, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(44, 1), stride=(44, 1))
         )
         

   def forward(self, emb_input):
      """
      emb_input: (B, C, nband, T)
      return:
         mag: (B, F, T)
         phase: (B, F, T)
      """
      if self.decode_type.lower() in ["mag+phase", "res+phase"]:
         x1, x2, x3 = emb_input[:, :, :6].contiguous(), \
                      emb_input[:, :, 6:10].contiguous(), \
                      emb_input[:, :, 10:].contiguous()
         mag1, mag2, mag3 = self.reg1_mag_decoder(x1), self.reg2_mag_decoder(x2), self.reg3_mag_decoder(x3)
         com1, com2, com3 = self.reg1_phase_decoder(x1), self.reg2_phase_decoder(x2), self.reg3_phase_decoder(x3)
         mag = torch.exp(torch.cat([mag1, mag2, mag3], dim=-2))  # exp operation
         com = torch.cat([com1, com2, com3], dim=-2)
         last_mag, last_com = mag[..., -1, :].unsqueeze(-2), com[..., -1, :].unsqueeze(-2)
         mag, com = torch.cat([mag, last_mag], dim=-2), torch.cat([com, last_com], dim=-2)
         pha = torch.atan2(com[:, -1], com[:, 0])
         return mag.squeeze(1), pha
      elif self.decode_type.lower() in ["ri", "res_ri"]:
         x1, x2, x3 = emb_input[:, :, :12].contiguous(), \
                      emb_input[:, :, 12:20].contiguous(), \
                      emb_input[:, :, 20:].contiguous()
         real1, real2, real3 = self.reg1_real_decoder(x1), self.reg2_real_decoder(x2), self.reg3_real_decoder(x3)
         imag1, imag2, imag3 = self.reg1_imag_decoder(x1), self.reg2_imag_decoder(x2), self.reg3_imag_decoder(x3)
         real = torch.cat([real1, real2, real3], dim=-2)
         imag = torch.cat([imag1, imag2, imag3], dim=-2)
         last_real, last_imag = real[..., -1, :].unsqueeze(-2), imag[..., -1, :].unsqueeze(-2)
         real, imag = torch.cat([real, last_real], dim=-2), torch.cat([imag, last_imag], dim=-2)
         return real.squeeze(1), imag.squeeze(1)


class SharedBandMerge_NB48_22k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int, 
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64,
               decode_type: str = 'mag+phase',  
               ):
      super(SharedBandMerge_NB48_22k, self).__init__()
      self.sr = sr       
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.decode_type = decode_type
      self.eps = torch.finfo(torch.float32).eps
      if self.decode_type.lower() in ["mag+phase", "res+phase"]:
         self.reg1_mag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=24, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(6, 1), stride=(6, 1))
         )
         self.reg2_mag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=16, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg3_mag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(22, 1), stride=(22, 1))
         )
         self.reg1_phase_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=24, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=2, kernel_size=(6, 1), stride=(6, 1))
         )
         self.reg2_phase_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=16, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=2, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg3_phase_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=2, kernel_size=(22, 1), stride=(22, 1))
         )
      elif self.decode_type.lower() in ["ri", "res_ri"]:
         self.reg1_real_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=12, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg2_real_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg3_real_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=4, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(44, 1), stride=(44, 1))
         )
         self.reg1_imag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=12, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg2_imag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg3_imag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=4, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(44, 1), stride=(44, 1))
         )
         

   def forward(self, emb_input):
      """
      emb_input: (B, C, nband, T)
      return:
         mag: (B, F, T)
         phase: (B, F, T)
      """
      if self.decode_type.lower() in ["mag+phase", "res+phase"]:
         x1, x2, x3 = emb_input[:, :, :24].contiguous(), \
                      emb_input[:, :, 24:40].contiguous(), \
                      emb_input[:, :, 40:].contiguous()
         mag1, mag2, mag3 = self.reg1_mag_decoder(x1), self.reg2_mag_decoder(x2), self.reg3_mag_decoder(x3)
         com1, com2, com3 = self.reg1_phase_decoder(x1), self.reg2_phase_decoder(x2), self.reg3_phase_decoder(x3)
         mag = torch.exp(torch.cat([mag1, mag2, mag3], dim=-2))  # exp operation
         com = torch.cat([com1, com2, com3], dim=-2)
         last_mag, last_com = mag[..., -1, :].unsqueeze(-2), com[..., -1, :].unsqueeze(-2)
         mag, com = torch.cat([mag, last_mag], dim=-2), torch.cat([com, last_com], dim=-2)
         pha = torch.atan2(com[:, -1], com[:, 0])
         return mag.squeeze(1), pha
      elif self.decode_type.lower() in ["ri", "res_ri"]:
         x1, x2, x3 = emb_input[:, :, :12].contiguous(), \
                      emb_input[:, :, 12:20].contiguous(), \
                      emb_input[:, :, 20:].contiguous()
         real1, real2, real3 = self.reg1_real_decoder(x1), self.reg2_real_decoder(x2), self.reg3_real_decoder(x3)
         imag1, imag2, imag3 = self.reg1_imag_decoder(x1), self.reg2_imag_decoder(x2), self.reg3_imag_decoder(x3)
         real = torch.cat([real1, real2, real3], dim=-2)
         imag = torch.cat([imag1, imag2, imag3], dim=-2)
         last_real, last_imag = real[..., -1, :].unsqueeze(-2), imag[..., -1, :].unsqueeze(-2)
         real, imag = torch.cat([real, last_real], dim=-2), torch.cat([imag, last_imag], dim=-2)
         return real.squeeze(1), imag.squeeze(1)


class SharedBandMerge_NB96_22k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int, 
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64,
               decode_type: str = 'mag+phase',  
               ):
      super(SharedBandMerge_NB96_22k, self).__init__()
      self.sr = sr       
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.decode_type = decode_type
      self.eps = torch.finfo(torch.float32).eps
      if self.decode_type.lower() in ["mag+phase", "res+phase"]:
         self.reg1_mag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=48, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(3, 1), stride=(3, 1))
         )
         self.reg2_mag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=32, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(6, 1), stride=(6, 1))
         )
         self.reg3_mag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=16, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(11, 1), stride=(11, 1))
         )
         self.reg1_phase_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=48, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=2, kernel_size=(3, 1), stride=(3, 1))
         )
         self.reg2_phase_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=32, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=2, kernel_size=(6, 1), stride=(6, 1))
         )
         self.reg3_phase_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=16, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=2, kernel_size=(11, 1), stride=(11, 1))
         )
      elif self.decode_type.lower() in ["ri", "res_ri"]:
         self.reg1_real_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=12, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg2_real_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg3_real_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=4, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(44, 1), stride=(44, 1))
         )
         self.reg1_imag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=12, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg2_imag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg3_imag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=4, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(44, 1), stride=(44, 1))
         )
         
   def forward(self, emb_input):
      """
      emb_input: (B, C, nband, T)
      return:
         mag: (B, F, T)
         phase: (B, F, T)
      """
      if self.decode_type.lower() in ["mag+phase", "res+phase"]:
         x1, x2, x3 = emb_input[:, :, :48].contiguous(), \
                      emb_input[:, :, 48:80].contiguous(), \
                      emb_input[:, :, 80:].contiguous()
         mag1, mag2, mag3 = self.reg1_mag_decoder(x1), self.reg2_mag_decoder(x2), self.reg3_mag_decoder(x3)
         com1, com2, com3 = self.reg1_phase_decoder(x1), self.reg2_phase_decoder(x2), self.reg3_phase_decoder(x3)
         mag = torch.exp(torch.cat([mag1, mag2, mag3], dim=-2))  # exp operation
         com = torch.cat([com1, com2, com3], dim=-2)
         last_mag, last_com = mag[..., -1, :].unsqueeze(-2), com[..., -1, :].unsqueeze(-2)
         mag, com = torch.cat([mag, last_mag], dim=-2), torch.cat([com, last_com], dim=-2)
         pha = torch.atan2(com[:, -1], com[:, 0])
         return mag.squeeze(1), pha
      elif self.decode_type.lower() in ["ri", "res_ri"]:
         x1, x2, x3 = emb_input[:, :, :12].contiguous(), \
                      emb_input[:, :, 12:20].contiguous(), \
                      emb_input[:, :, 20:].contiguous()
         real1, real2, real3 = self.reg1_real_decoder(x1), self.reg2_real_decoder(x2), self.reg3_real_decoder(x3)
         imag1, imag2, imag3 = self.reg1_imag_decoder(x1), self.reg2_imag_decoder(x2), self.reg3_imag_decoder(x3)
         real = torch.cat([real1, real2, real3], dim=-2)
         imag = torch.cat([imag1, imag2, imag3], dim=-2)
         last_real, last_imag = real[..., -1, :].unsqueeze(-2), imag[..., -1, :].unsqueeze(-2)
         real, imag = torch.cat([real, last_real], dim=-2), torch.cat([imag, last_imag], dim=-2)
         return real.squeeze(1), imag.squeeze(1)


class StarConv(nn.Module):
   def __init__(self, 
                 input_channel: int,
                 kernel_size: int,
                 causal: bool,
                 ):
      super(StarConv, self).__init__()
      self.input_channel = input_channel
      self.kernel_size = kernel_size
      self.causal = causal
      if not self.causal:
         self.conv1 = nn.Conv1d(input_channel, input_channel, kernel_size=kernel_size, groups=input_channel, padding="same", padding_mode="zeros")
         self.conv2 = nn.Conv1d(input_channel, input_channel, kernel_size=kernel_size, groups=input_channel, padding="same", padding_mode="zeros")
      else:
         self.conv1 = nn.Sequential(
             nn.ConstantPad1d([kernel_size - 1, 0], value=0.),
             nn.Conv1d(input_channel, input_channel, kernel_size=kernel_size, groups=input_channel)
         )
         self.conv2 = nn.Sequential(
             nn.ConstantPad1d([kernel_size - 1, 0], value=0.),
             nn.Conv1d(input_channel, input_channel, kernel_size=kernel_size, groups=input_channel)
         )

   def forward(self, inpt):
      return self.conv1(inpt) * self.conv2(inpt)


class BandShuffler(nn.Module):
   """
   The structure is from https://github.com/Audio-WestlakeU/NBSS/blob/main/models/arch/SpatialNet.py
   """
   def __init__(self, 
               nband: int,
               input_size: int,
               squeeze_size: int=64,
               f_kernel_size: int=3,
               f_conv_groups: int=8,
               ):
      super(BandShuffler, self).__init__()
      self.nband = nband
      self.input_size = input_size
      self.squeeze_size = squeeze_size
      self.f_kernel_size = f_kernel_size
      self.f_conv_groups = f_conv_groups
      #
      self.fconv1 = nn.Sequential(
         ChannelNormalization(input_size),
         nn.Conv1d(input_size, input_size, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode='zeros'),
         nn.PReLU(input_size)
      )
      self.fconv2 = nn.Sequential(
         ChannelNormalization(input_size),
         nn.Conv1d(input_size, input_size, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode='zeros'),
         nn.PReLU(input_size)
      )
      self.squeeze = nn.Sequential(nn.Conv1d(in_channels=input_size, out_channels=squeeze_size, kernel_size=1), nn.SiLU())
      self.unsqueeze = nn.Sequential(nn.Conv1d(in_channels=squeeze_size, out_channels=input_size, kernel_size=1), nn.SiLU())
      self.full = LinearGroup(nband, nband, squeeze_size)

   def forward(self, input):
      """
      input: (B, T, C, nband)
      return: (B, T, C, nband)
      """
      b_size, seq_len, c, nband = input.shape
      x = input.view(b_size * seq_len, c, nband)  # (B*T, C, nband)
      # f-conv1
      resi = x
      x = self.fconv1(x)
      x = resi + x
      # group
      resi = x
      x = self.squeeze(x)
      x = self.full(x)
      x = self.unsqueeze(x)
      x = resi + x
      # f-conv2
      resi = x
      x = self.fconv2(x)
      x = resi + x
      # reshape
      out = x.view(*input.shape)
      return out


class TimeResRNN(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 dropout: float = 0.,
                 causal: bool = True,
                 residual: bool = True,
                 ):
        super(TimeResRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.causal = causal
        self.residual = residual
        self.eps = torch.finfo(torch.float32).eps
        if not causal:
            self.norm = TimeGlobalNormalization(input_size, ndim=4)
        else:
            self.norm = nn.LayerNorm(input_size)

        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=not causal)

        # linear projection layer
        self.proj = nn.Linear(hidden_size*(int(not causal) + 1), input_size)

    def forward(self, input):
        """
        input: (B, nband, C, T)
        return: (B, nband, C, T)
        """
        batch_size, t1, E, t2 = input.shape
        x = input.transpose(-2, -1).contiguous()
        x = self.norm(x)
        x = x.view(batch_size * t1, t2, E)
        rnn_output, _ = self.rnn(self.dropout(x))
        rnn_output = self.proj(rnn_output).transpose(-2, -1).contiguous().view(*input.shape)
        if self.residual:
            return input + rnn_output
        else:
            return rnn_output


class FreqResRNN(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 dropout: float = 0.,
                 causal: bool = True,
                 residual: bool = True,
                 ):
        super(FreqResRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.causal = causal
        self.residual = residual
        self.eps = torch.finfo(torch.float32).eps
        self.norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=not causal)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * (int(not causal) + 1), input_size)

    def forward(self, input):
        """
        input: (B, T, C, nband)
        return: (B, T, C, nband)
        """
        batch_size, t1, E, t2 = input.shape
        x= input.transpose(-2, -1).contiguous()
        x = self.norm(x)
        x = x.view(batch_size * t1, t2, E)
        rnn_output, _ = self.rnn(self.dropout(x))
        rnn_output = self.proj(rnn_output).transpose(-2, -1).contiguous().view(*input.shape)
        if self.residual:
            return input + rnn_output
        else:
            return rnn_output


class BandWiseTimeModule(nn.Module):
   def __init__(self, 
                nband: int,
                nrep: int,
                input_channel: int,
                hidden_channel: int,
                kernel_size: int,
                causal: bool = False,
                ):
      super(BandWiseTimeModule, self).__init__()
      self.nband = nband
      self.nrep = nrep
      self.input_channel = input_channel
      self.hidden_channel = hidden_channel
      self.kernel_size = kernel_size
      self.causal = causal

      band_timenet_list = []
      for _ in range(self.nrep):
         band_timenet_list.append(
            nn.Sequential(
               nn.Conv1d(input_channel, input_channel, kernel_size, padding="same", padding_mode="zeros", groups=input_channel),
               BandwiseLayerNorm(self.nband, input_channel),
               nn.Conv1d(input_channel, hidden_channel, 1),
               nn.GELU(),
               GRN(hidden_channel),
               nn.Conv1d(hidden_channel, input_channel, 1)
            )
         )
      self.Ttband_timenet_list = nn.ModuleList(band_timenet_list)

   def forward(self, input):
      """
      inpt: (B, nband, C, T)
      return: (B, nband, C, T)
      """
      #
      b_size, nband, nch, seq_len = input.shape
      x = input.view(b_size*nband, nch, -1)
      for timenet in self.Ttband_timenet_list:
         bot = x.clone()
         x = timenet(x)
         x = x + bot
      out = x.view(b_size, nband, nch, seq_len)

      return out
