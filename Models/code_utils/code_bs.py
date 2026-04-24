import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm import *


class BandSplit_code320_24k(nn.Module):
   def __init__(self,
               nband: int,
               code_dim: int,
               rank_reduce: int = 4,
               feature_dim: int = 64,
               ):
      super(BandSplit_code320_24k, self).__init__()
      self.nband = nband    
      self.code_dim = code_dim
      self.rank_reduce = rank_reduce
      self.feature_dim = feature_dim
      self.eps = torch.finfo(torch.float32).eps

      print(f'Totally splitting {self.nband} bands for sampling rate: 24k.')

      cond_dim = self.feature_dim
      self.cond_encoder = nn.ModuleList([])
      for _ in range(self.nband):
            self.cond_encoder.append(
               nn.Sequential(
                  ChannelNormalization(self.code_dim, ndim=3),
                  nn.Conv1d(self.code_dim, self.code_dim // self.rank_reduce, 1, bias=False),
                  nn.Conv1d(self.code_dim // self.rank_reduce, cond_dim, 1)
               )
            )

   def get_nband(self):
      return self.nband

   def forward(self, cond=None):
      """
      cond: (B, C, T)
      xt: (B, 2, F, T) or None
      time_ada_begin: (B, C) or None
      return: (B, C, nband, T)
      """
      # cond encoder
      if self.training:
         cond_list = []
         for layer in self.cond_encoder:
            cond_list.append(layer(cond))
         out = torch.stack(cond_list, dim=-2)  # (B, C, nband, T)
      else:
         # 只算一次共享统计量
         eps = self.cond_encoder[0][0].eps  # 这里要用 ChannelNormalization 的 eps=1e-5
         mean_ = cond.mean(dim=1, keepdim=True)
         std_ = torch.sqrt(cond.var(dim=1, keepdim=True, unbiased=False) + eps)
         cond_hat = (cond - mean_) / std_

         B, _, T = cond.shape
         out = cond.new_empty(B, self.feature_dim, self.nband, T)

         for i, layer in enumerate(self.cond_encoder):
            cn, conv1, conv2 = layer
            x = cond_hat
            if cn.affine:
                  x = x * cn.gain + cn.bias
            x = conv1(x)
            x = conv2(x)
            out[:, :, i, :] = x

      return out


class BandSplit_code320_NB24_Var(nn.Module):
   def __init__(self,
                nband: int,
                code_dim: int,
                rank_reduce: int = 4,
                feature_dim: int = 64,
                compensate_type: str = "pad",
               ):
      super(BandSplit_code320_NB24_Var, self).__init__()
      self.nband = nband    
      self.code_dim = code_dim
      self.rank_reduce = rank_reduce
      self.feature_dim = feature_dim
      self.compensate_type = compensate_type
      self.eps = torch.finfo(torch.float32).eps

      cond_dim = self.feature_dim
      self.cond_encoder = nn.ModuleList([])
      for _ in range(self.nband):
            self.cond_encoder.append(
               nn.Sequential(
                  ChannelNormalization(self.code_dim, ndim=3),
                  nn.Conv1d(self.code_dim, self.code_dim // self.rank_reduce, 1, bias=False),
                  nn.Conv1d(self.code_dim // self.rank_reduce, cond_dim, 1)
               )
            )

   def get_nband(self):
      return self.nband

   def forward(self, cond=None, sr=None):
      """
      cond: (B, C, T)
      sr: float or None
      return: (B, C, nband, T)
      """
      # cond encoder
      cond_list = []
      if self.compensate_type == "zero_padding":
         for i in range(24):
            cond_list.append(self.cond_encoder[i](cond))
         out = torch.stack(cond_list, dim=-2)
         if sr == 44100 or sr is None:
            pad_out = torch.zeros([out.shape[0], out.shape[1], 8, out.shape[-1]], device=out.device)
            out = torch.cat([out, pad_out], dim=-2)
      elif self.compensate_type == "code_padding":
         if sr is None or sr == 44100:
            nb = self.nband
         elif sr == 24000:
            nb = 24
         for i in range(nb):
            cond_list.append(self.cond_encoder[i](cond))
         out = torch.stack(cond_list, dim=-2)  # (B, C, nband, T)

      return out


class SharedBandSplit_NB24_24k(nn.Module):
   def __init__(self,
               feature_dim: int = 64,
               ):
      super(SharedBandSplit_NB24_24k, self).__init__()    
      self.feature_dim = feature_dim
      self.eps = torch.finfo(torch.float32).eps

      self.reg1_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(16, 3), stride=(16, 1)),
          BandwiseC2LayerNorm(nband=12, feature_dim=self.feature_dim)
      )
      self.reg2_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(32, 3), stride=(32, 1)),
          BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim)
      )
      self.reg3_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(48, 3), stride=(48, 1)),
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
      x1, x2, x3 = input[..., :192, :], input[..., 192:448, :], input[..., 448:-1, :]
      y1, y2, y3 = self.reg1_encoder(x1), self.reg2_encoder(x2), self.reg3_encoder(x3)

      out = torch.cat([y1, y2, y3], dim=-2)  # (B, C, nband, C)

      return out


class BandMerge_code320_NB6_24k(nn.Module):
   def __init__(self,
               feature_dim: int = 64,
               decode_type: str = "mag+phase",
               ):
      super(BandMerge_code320_NB6_24k, self).__init__()
      self.feature_dim = feature_dim
      self.decode_type = decode_type
      self.eps = torch.finfo(torch.float32).eps
      
      self.norm1 = ChannelNormalization(feature_dim, ndim=4)
      self.norm2 = ChannelNormalization(feature_dim, ndim=4)
      if self.decode_type == "mag+phase":
         self.reg1_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(64, 1), stride=(64, 1))
         )
         self.reg2_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(128, 1), stride=(128, 1))
         )
         self.reg3_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(192, 1), stride=(192, 1))
         )
         self.reg1_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(64, 1), stride=(64, 1))
         )
         self.reg2_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(128, 1), stride=(128, 1))
         )
         self.reg3_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(192, 1), stride=(192, 1))
         )
      elif self.decode_type == "ri":
         self.reg1_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(64, 1), stride=(64, 1))
         )
         self.reg2_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(128, 1), stride=(128, 1))
         )
         self.reg3_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(192, 1), stride=(192, 1))
         )
         self.reg1_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(64, 1), stride=(64, 1))
         )
         self.reg2_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(128, 1), stride=(128, 1))
         )
         self.reg3_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(192, 1), stride=(192, 1))
         )

   def forward(self, emb_input):
      """
      emb_input: (B, C, nband, T)
      return:
         mag: (B, F, T)
         phase: (B, F, T)
      """
      b_size = emb_input.shape[0]
      emb_input1, emb_input2 = self.norm1(emb_input), self.norm2(emb_input)

      x1_1, x1_2, x1_3 = emb_input1[:, :, :3].contiguous(), \
                           emb_input1[:, :, 3:5].contiguous(), \
                           emb_input1[:, :, 5:].contiguous()
      x2_1, x2_2, x2_3 = emb_input2[:, :, :3].contiguous(), \
                           emb_input2[:, :, 3:5].contiguous(), \
                           emb_input2[:, :, 5:].contiguous()

      if self.decode_type == "mag+phase":
         mag1, mag2, mag3 = self.reg1_mag_decoder(x1_1), self.reg2_mag_decoder(x1_2), self.reg3_mag_decoder(x1_3)
         com1, com2, com3 = self.reg1_pha_decoder(x2_1), self.reg2_pha_decoder(x2_2), self.reg3_pha_decoder(x2_3)
         mag = F.softplus(torch.cat([mag1, mag2, mag3], dim=-2))
         com = torch.cat([com1, com2, com3], dim=-2)
         last_mag, last_com = mag[..., -1, :].unsqueeze(-2), com[..., -1, :].unsqueeze(-2)
         mag, com = torch.cat([mag, last_mag], dim=-2), torch.cat([com, last_com], dim=-2)
         pha = torch.atan2(com[:, -1], com[:, 0])

         return mag.squeeze(1), pha
      elif self.decode_type == "ri":
         rea1, rea2, rea3 = self.reg1_rea_decoder(x1_1), self.reg2_rea_decoder(x1_2), self.reg3_rea_decoder(x1_3)
         imag1, imag2, imag3 = self.reg1_imag_decoder(x2_1), self.reg2_imag_decoder(x2_2), self.reg3_imag_decoder(x2_3)
         rea = torch.cat([rea1, rea2, rea3], dim=-2) 
         imag = torch.cat([imag1, imag2, imag3], dim=-2)
         last_rea, last_imag = rea[..., -1, :].unsqueeze(-2), imag[..., -1, :].unsqueeze(-2)
         rea, imag = torch.cat([rea, last_rea], dim=-2), torch.cat([imag, last_imag], dim=-2)
         
         return rea.squeeze(1), imag.squeeze(1)


class BandMerge_code320_NB12_24k(nn.Module):
   def __init__(self,
               feature_dim: int = 64,
               decode_type: str = "ri",
               ):
      super(BandMerge_code320_NB12_24k, self).__init__()
      self.feature_dim = feature_dim
      self.decode_type = decode_type
      self.eps = torch.finfo(torch.float32).eps
      
      self.norm1 = ChannelNormalization(feature_dim, ndim=4)
      self.norm2 = ChannelNormalization(feature_dim, ndim=4)
      if self.decode_type == "mag+phase":
         self.reg1_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(32, 1), stride=(32, 1))
         )
         self.reg2_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(64, 1), stride=(64, 1))
         )
         self.reg3_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(96, 1), stride=(96, 1))
         )
         self.reg1_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(32, 1), stride=(32, 1))
         )
         self.reg2_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(64, 1), stride=(64, 1))
         )
         self.reg3_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(96, 1), stride=(96, 1))
         )
      elif self.decode_type == "ri":
         self.reg1_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(32, 1), stride=(32, 1))
         )
         self.reg2_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(64, 1), stride=(64, 1))
         )
         self.reg3_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(96, 1), stride=(96, 1))
         )
         self.reg1_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(32, 1), stride=(32, 1))
         )
         self.reg2_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(64, 1), stride=(64, 1))
         )
         self.reg3_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(96, 1), stride=(96, 1))
         )

   def forward(self, emb_input):
      """
      emb_input: (B, C, nband, T)
      return:
         mag: (B, F, T)
         phase: (B, F, T)
      """
      b_size = emb_input.shape[0]
      emb_input1, emb_input2 = self.norm1(emb_input), self.norm2(emb_input)

      x1_1, x1_2, x1_3 = emb_input1[:, :, :6].contiguous(), \
                           emb_input1[:, :, 6:10].contiguous(), \
                           emb_input1[:, :, 10:].contiguous()
      x2_1, x2_2, x2_3 = emb_input2[:, :, :6].contiguous(), \
                           emb_input2[:, :, 6:10].contiguous(), \
                           emb_input2[:, :, 10:].contiguous()

      if self.decode_type == "mag+phase":
         mag1, mag2, mag3 = self.reg1_mag_decoder(x1_1), self.reg2_mag_decoder(x1_2), self.reg3_mag_decoder(x1_3)
         com1, com2, com3 = self.reg1_pha_decoder(x2_1), self.reg2_pha_decoder(x2_2), self.reg3_pha_decoder(x2_3)
         mag = F.softplus(torch.cat([mag1, mag2, mag3], dim=-2)) 
         com = torch.cat([com1, com2, com3], dim=-2)
         last_mag, last_com = mag[..., -1, :].unsqueeze(-2), com[..., -1, :].unsqueeze(-2)
         mag, com = torch.cat([mag, last_mag], dim=-2), torch.cat([com, last_com], dim=-2)
         pha = torch.atan2(com[:, -1], com[:, 0])

         return mag.squeeze(1), pha
      elif self.decode_type == "ri":
         rea1, rea2, rea3 = self.reg1_rea_decoder(x1_1), self.reg2_rea_decoder(x1_2), self.reg3_rea_decoder(x1_3)
         imag1, imag2, imag3 = self.reg1_imag_decoder(x2_1), self.reg2_imag_decoder(x2_2), self.reg3_imag_decoder(x2_3)
         rea = torch.cat([rea1, rea2, rea3], dim=-2) 
         imag = torch.cat([imag1, imag2, imag3], dim=-2)
         last_rea, last_imag = rea[..., -1, :].unsqueeze(-2), imag[..., -1, :].unsqueeze(-2)
         rea, imag = torch.cat([rea, last_rea], dim=-2), torch.cat([imag, last_imag], dim=-2)
         
         return rea.squeeze(1), imag.squeeze(1)


class BandMerge_code320_NB24_24k(nn.Module):
   def __init__(self,
               feature_dim: int = 64,
               decode_type: str = "mag+phase",
               act_type: str = "tanh",
               mag_act_type: str = "exp",
               ):
      super(BandMerge_code320_NB24_24k, self).__init__()
      self.feature_dim = feature_dim
      self.decode_type = decode_type
      self.act_type = act_type
      self.mag_act_type = mag_act_type
      self.eps = torch.finfo(torch.float32).eps
      
      if self.act_type == "gelu":
         act = nn.GELU()
      elif self.act_type == "tanh":
         act = nn.Tanh()
      
      self.norm1 = ChannelNormalization(feature_dim, ndim=4)
      self.norm2 = ChannelNormalization(feature_dim, ndim=4)
      if self.decode_type == "mag+phase":
         self.reg1_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(16, 1), stride=(16, 1))
         )
         self.reg2_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(32, 1), stride=(32, 1))
         )
         self.reg3_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(48, 1), stride=(48, 1))
         )
         self.reg1_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(16, 1), stride=(16, 1))
         )
         self.reg2_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(32, 1), stride=(32, 1))
         )
         self.reg3_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(48, 1), stride=(48, 1))
         )
      elif self.decode_type == "ri":
         self.reg1_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(16, 1), stride=(16, 1))
         )
         self.reg2_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(32, 1), stride=(32, 1))
         )
         self.reg3_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(48, 1), stride=(48, 1))
         )
         self.reg1_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(16, 1), stride=(16, 1))
         )
         self.reg2_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(32, 1), stride=(32, 1))
         )
         self.reg3_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(48, 1), stride=(48, 1))
         )

   def forward(self, emb_input):
      """
      emb_input: (B, C, nband, T)
      return:
         mag: (B, F, T)
         phase: (B, F, T)
      """
      emb_input1, emb_input2 = self.norm1(emb_input), self.norm2(emb_input)

      x1_1, x1_2, x1_3 = emb_input1[:, :, :12].contiguous(), \
                           emb_input1[:, :, 12:20].contiguous(), \
                           emb_input1[:, :, 20:].contiguous()
      x2_1, x2_2, x2_3 = emb_input2[:, :, :12].contiguous(), \
                           emb_input2[:, :, 12:20].contiguous(), \
                           emb_input2[:, :, 20:].contiguous()

      if self.decode_type == "mag+phase":
         mag1, mag2, mag3 = self.reg1_mag_decoder(x1_1), self.reg2_mag_decoder(x1_2), self.reg3_mag_decoder(x1_3)
         com1, com2, com3 = self.reg1_pha_decoder(x2_1), self.reg2_pha_decoder(x2_2), self.reg3_pha_decoder(x2_3)
         
         # 原始mag生成逻辑
         if self.mag_act_type == "softplus":
            mag = F.softplus(torch.cat([mag1, mag2, mag3], dim=-2))
         elif self.mag_act_type == "exp": 
            mag = torch.exp(torch.cat([mag1, mag2, mag3], dim=-2))

         com = torch.cat([com1, com2, com3], dim=-2)
         last_mag, last_com = mag[..., -1, :].unsqueeze(-2), com[..., -1, :].unsqueeze(-2)
         mag, com = torch.cat([mag, last_mag], dim=-2), torch.cat([com, last_com], dim=-2)
         pha = torch.atan2(com[:, -1], com[:, 0])

         return mag.squeeze(1), pha
      elif self.decode_type == "ri":
         rea1, rea2, rea3 = self.reg1_rea_decoder(x1_1), self.reg2_rea_decoder(x1_2), self.reg3_rea_decoder(x1_3)
         imag1, imag2, imag3 = self.reg1_imag_decoder(x2_1), self.reg2_imag_decoder(x2_2), self.reg3_imag_decoder(x2_3)
         rea = torch.cat([rea1, rea2, rea3], dim=-2) 
         imag = torch.cat([imag1, imag2, imag3], dim=-2)
         last_rea, last_imag = rea[..., -1, :].unsqueeze(-2), imag[..., -1, :].unsqueeze(-2)
         rea, imag = torch.cat([rea, last_rea], dim=-2), torch.cat([imag, last_imag], dim=-2)
         
         mag = torch.sqrt(rea ** 2 + imag ** 2 + self.eps)
         pha = torch.atan2(imag, rea)
         
         return mag.squeeze(1), pha.squeeze(1)


class BandMerge_code320_NB24_even_24k(nn.Module):
   def __init__(self,
               feature_dim: int = 64,
               decode_type: str = "ri",
               ):
      super(BandMerge_code320_NB24_even_24k, self).__init__()
      self.feature_dim = feature_dim
      self.decode_type = decode_type
      self.eps = torch.finfo(torch.float32).eps
      
      self.norm1 = ChannelNormalization(feature_dim, ndim=4)
      self.norm2 = ChannelNormalization(feature_dim, ndim=4)
      if self.decode_type == "mag+phase":
         self.reg1_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(27, 1), stride=(27, 1))
         )
         self.reg2_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(27, 1), stride=(27, 1))
         )
         self.reg3_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(27, 1), stride=(27, 1))
         )
         self.reg1_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(27, 1), stride=(27, 1))
         )
         self.reg2_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(27, 1), stride=(27, 1))
         )
         self.reg3_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(27, 1), stride=(27, 1))
         )
      elif self.decode_type == "ri":
         self.reg1_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(27, 1), stride=(27, 1))
         )
         self.reg2_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(27, 1), stride=(27, 1))
         )
         self.reg3_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(27, 1), stride=(27, 1))
         )
         self.reg1_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(27, 1), stride=(27, 1))
         )
         self.reg2_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(27, 1), stride=(27, 1))
         )
         self.reg3_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(27, 1), stride=(27, 1))
         )

   def forward(self, emb_input):
      """
      emb_input: (B, C, nband, T)
      return:
         mag: (B, F, T)
         phase: (B, F, T)
      """
      emb_input1, emb_input2 = self.norm1(emb_input), self.norm2(emb_input)

      x1_1, x1_2, x1_3 = emb_input1[:, :, :8].contiguous(), \
                           emb_input1[:, :, 8:16].contiguous(), \
                           emb_input1[:, :, 16:].contiguous()
      x2_1, x2_2, x2_3 = emb_input2[:, :, :8].contiguous(), \
                           emb_input2[:, :, 8:16].contiguous(), \
                           emb_input2[:, :, 16:].contiguous()

      if self.decode_type == "mag+phase":
         mag1, mag2, mag3 = self.reg1_mag_decoder(x1_1), self.reg2_mag_decoder(x1_2), self.reg3_mag_decoder(x1_3)
         com1, com2, com3 = self.reg1_pha_decoder(x2_1), self.reg2_pha_decoder(x2_2), self.reg3_pha_decoder(x2_3)
         mag = F.softplus(torch.cat([mag1, mag2, mag3], dim=-2)) 
         com = torch.cat([com1, com2, com3], dim=-2)
         mag, com = mag[..., :-7, :].contiguous(), com[..., :-7, :].contiguous()
         pha = torch.atan2(com[:, -1], com[:, 0])

         return mag.squeeze(1), pha
      elif self.decode_type == "ri":
         rea1, rea2, rea3 = self.reg1_rea_decoder(x1_1), self.reg2_rea_decoder(x1_2), self.reg3_rea_decoder(x1_3)
         imag1, imag2, imag3 = self.reg1_imag_decoder(x2_1), self.reg2_imag_decoder(x2_2), self.reg3_imag_decoder(x2_3)
         rea = torch.cat([rea1, rea2, rea3], dim=-2) 
         imag = torch.cat([imag1, imag2, imag3], dim=-2)
         rea, imag = rea[..., :-7, :].contiguous(), imag[..., :-7, :].contiguous()
         
         return rea.squeeze(1), imag.squeeze(1)


class BandMerge_code320_NB48_24k(nn.Module):
   def __init__(self,
               feature_dim: int = 64,
               decode_type: str = "ri",
               ):
      super(BandMerge_code320_NB48_24k, self).__init__()
      self.feature_dim = feature_dim
      self.decode_type = decode_type
      self.eps = torch.finfo(torch.float32).eps
      
      self.norm1 = ChannelNormalization(feature_dim, ndim=4)
      self.norm2 = ChannelNormalization(feature_dim, ndim=4)
      if self.decode_type == "mag+phase":
         self.reg1_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(8, 1), stride=(8, 1))
         )
         self.reg2_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(16, 1), stride=(16, 1))
         )
         self.reg3_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg1_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(8, 1), stride=(8, 1))
         )
         self.reg2_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(16, 1), stride=(16, 1))
         )
         self.reg3_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(24, 1), stride=(24, 1))
         )
      elif self.decode_type == "ri":
         self.reg1_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(8, 1), stride=(8, 1))
         )
         self.reg2_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(16, 1), stride=(16, 1))
         )
         self.reg3_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg1_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(8, 1), stride=(8, 1))
         )
         self.reg2_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(16, 1), stride=(16, 1))
         )
         self.reg3_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(24, 1), stride=(24, 1))
         )

   def forward(self, emb_input):
      """
      emb_input: (B, C, nband, T)
      return:
         mag: (B, F, T)
         phase: (B, F, T)
      """
      emb_input1, emb_input2 = self.norm1(emb_input), self.norm2(emb_input)

      x1_1, x1_2, x1_3 = emb_input1[:, :, :24].contiguous(), \
                           emb_input1[:, :, 24:40].contiguous(), \
                           emb_input1[:, :, 40:].contiguous()
      x2_1, x2_2, x2_3 = emb_input2[:, :, :24].contiguous(), \
                           emb_input2[:, :, 24:40].contiguous(), \
                           emb_input2[:, :, 40:].contiguous()

      if self.decode_type == "mag+phase":
         mag1, mag2, mag3 = self.reg1_mag_decoder(x1_1), self.reg2_mag_decoder(x1_2), self.reg3_mag_decoder(x1_3)
         com1, com2, com3 = self.reg1_pha_decoder(x2_1), self.reg2_pha_decoder(x2_2), self.reg3_pha_decoder(x2_3)
         mag = F.softplus(torch.cat([mag1, mag2, mag3], dim=-2)) 
         com = torch.cat([com1, com2, com3], dim=-2)
         last_mag, last_com = mag[..., -1, :].unsqueeze(-2), com[..., -1, :].unsqueeze(-2)
         mag, com = torch.cat([mag, last_mag], dim=-2), torch.cat([com, last_com], dim=-2)
         pha = torch.atan2(com[:, -1], com[:, 0])

         return mag.squeeze(1), pha
      elif self.decode_type == "ri":
         rea1, rea2, rea3 = self.reg1_rea_decoder(x1_1), self.reg2_rea_decoder(x1_2), self.reg3_rea_decoder(x1_3)
         imag1, imag2, imag3 = self.reg1_imag_decoder(x2_1), self.reg2_imag_decoder(x2_2), self.reg3_imag_decoder(x2_3)
         rea = torch.cat([rea1, rea2, rea3], dim=-2) 
         imag = torch.cat([imag1, imag2, imag3], dim=-2)
         last_rea, last_imag = rea[..., -1, :].unsqueeze(-2), imag[..., -1, :].unsqueeze(-2)
         rea, imag = torch.cat([rea, last_rea], dim=-2), torch.cat([imag, last_imag], dim=-2)
         
         return rea.squeeze(1), imag.squeeze(1)


class BandMerge_code320_NB96_24k(nn.Module):
   def __init__(self,
               feature_dim: int = 64,
               decode_type: str = "ri",
               ):
      super(BandMerge_code320_NB96_24k, self).__init__()
      self.feature_dim = feature_dim
      self.decode_type = decode_type
      self.eps = torch.finfo(torch.float32).eps
      
      self.norm1 = ChannelNormalization(feature_dim, ndim=4)
      self.norm2 = ChannelNormalization(feature_dim, ndim=4)
      if self.decode_type == "mag+phase":
         self.reg1_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(4, 1), stride=(4, 1))
         )
         self.reg2_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(8, 1), stride=(8, 1))
         )
         self.reg3_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg1_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(4, 1), stride=(4, 1))
         )
         self.reg2_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(8, 1), stride=(8, 1))
         )
         self.reg3_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(12, 1), stride=(12, 1))
         )
      elif self.decode_type == "ri":
         self.reg1_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(4, 1), stride=(4, 1))
         )
         self.reg2_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(8, 1), stride=(8, 1))
         )
         self.reg3_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg1_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(4, 1), stride=(4, 1))
         )
         self.reg2_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(8, 1), stride=(8, 1))
         )
         self.reg3_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(12, 1), stride=(12, 1))
         )

   def forward(self, emb_input):
      """
      emb_input: (B, C, nband, T)
      return:
         mag: (B, F, T)
         phase: (B, F, T)
      """
      emb_input1, emb_input2 = self.norm1(emb_input), self.norm2(emb_input)

      x1_1, x1_2, x1_3 = emb_input1[:, :, :48].contiguous(), \
                           emb_input1[:, :, 48:80].contiguous(), \
                           emb_input1[:, :, 80:].contiguous()
      x2_1, x2_2, x2_3 = emb_input2[:, :, :48].contiguous(), \
                           emb_input2[:, :, 48:80].contiguous(), \
                           emb_input2[:, :, 80:].contiguous()

      if self.decode_type == "mag+phase":
         mag1, mag2, mag3 = self.reg1_mag_decoder(x1_1), self.reg2_mag_decoder(x1_2), self.reg3_mag_decoder(x1_3)
         com1, com2, com3 = self.reg1_pha_decoder(x2_1), self.reg2_pha_decoder(x2_2), self.reg3_pha_decoder(x2_3)
         mag = F.softplus(torch.cat([mag1, mag2, mag3], dim=-2)) 
         com = torch.cat([com1, com2, com3], dim=-2)
         last_mag, last_com = mag[..., -1, :].unsqueeze(-2), com[..., -1, :].unsqueeze(-2)
         mag, com = torch.cat([mag, last_mag], dim=-2), torch.cat([com, last_com], dim=-2)
         pha = torch.atan2(com[:, -1], com[:, 0])

         return mag.squeeze(1), pha
      elif self.decode_type == "ri":
         rea1, rea2, rea3 = self.reg1_rea_decoder(x1_1), self.reg2_rea_decoder(x1_2), self.reg3_rea_decoder(x1_3)
         imag1, imag2, imag3 = self.reg1_imag_decoder(x2_1), self.reg2_imag_decoder(x2_2), self.reg3_imag_decoder(x2_3)
         rea = torch.cat([rea1, rea2, rea3], dim=-2) 
         imag = torch.cat([imag1, imag2, imag3], dim=-2)
         last_rea, last_imag = rea[..., -1, :].unsqueeze(-2), imag[..., -1, :].unsqueeze(-2)
         rea, imag = torch.cat([rea, last_rea], dim=-2), torch.cat([imag, last_imag], dim=-2)
         
         return rea.squeeze(1), imag.squeeze(1)


class BandMerge_code320_NB24_Var(nn.Module):
   def __init__(self,
               feature_dim: int = 64,
               decode_type: str = "mag+phase",
               act_type="gelu",
               ):
      super(BandMerge_code320_NB24_Var, self).__init__()
      self.feature_dim = feature_dim
      self.decode_type = decode_type
      self.act_type = act_type
      self.eps = torch.finfo(torch.float32).eps
      
      if self.act_type == "gelu":
         act = nn.GELU()
      elif self.act_type == "tanh":
         act = nn.Tanh()
      
      self.norm1 = ChannelNormalization(feature_dim, ndim=4)
      self.norm2 = ChannelNormalization(feature_dim, ndim=4)
      if self.decode_type == "mag+phase":
         self.reg1_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(16, 1), stride=(16, 1))
         )
         self.reg2_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(32, 1), stride=(32, 1))
         )
         self.reg3_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(48, 1), stride=(48, 1))
         )
         self.reg4_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(64, 1), stride=(64, 1))
         )
         self.reg5_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(92, 1), stride=(92, 1))
         )
         self.reg1_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(16, 1), stride=(16, 1))
         )
         self.reg2_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(32, 1), stride=(32, 1))
         )
         self.reg3_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(48, 1), stride=(48, 1))
         )
         self.reg4_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(64, 1), stride=(64, 1))
         )
         self.reg5_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(92, 1), stride=(92, 1))
         )
      elif self.decode_type == "ri":
         self.reg1_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(16, 1), stride=(16, 1))
         )
         self.reg2_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(32, 1), stride=(32, 1))
         )
         self.reg3_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(48, 1), stride=(48, 1))
         )
         self.reg4_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(64, 1), stride=(64, 1))
         )
         self.reg5_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(92, 1), stride=(92, 1))
         )
         self.reg1_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(16, 1), stride=(16, 1))
         )
         self.reg2_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(32, 1), stride=(32, 1))
         )
         self.reg3_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(48, 1), stride=(48, 1))
         )
         self.reg4_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(64, 1), stride=(64, 1))
         )
         self.reg5_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(92, 1), stride=(92, 1))
         )

   def forward(self, emb_input, sr=None):
      """
      emb_input: (B, C, nband, T)
      return:
         mag: (B, F, T)
         phase: (B, F, T)
      """
      emb_input1, emb_input2 = self.norm1(emb_input), self.norm2(emb_input)

      if sr == 24000:
         x1_1, x1_2, x1_3 = emb_input1[:, :, :12].contiguous(), \
                            emb_input1[:, :, 12:20].contiguous(), \
                            emb_input1[:, :, 20:].contiguous()
         x2_1, x2_2, x2_3 = emb_input2[:, :, :12].contiguous(), \
                            emb_input2[:, :, 12:20].contiguous(), \
                            emb_input2[:, :, 20:].contiguous()

         if self.decode_type == "mag+phase":
            mag1, mag2, mag3 = self.reg1_mag_decoder(x1_1), self.reg2_mag_decoder(x1_2), self.reg3_mag_decoder(x1_3)
            com1, com2, com3 = self.reg1_pha_decoder(x2_1), self.reg2_pha_decoder(x2_2), self.reg3_pha_decoder(x2_3)
            mag = torch.exp(torch.cat([mag1, mag2, mag3], dim=-2)) 
            com = torch.cat([com1, com2, com3], dim=-2)
            last_mag, last_com = mag[..., -1, :].unsqueeze(-2), com[..., -1, :].unsqueeze(-2)
            mag, com = torch.cat([mag, last_mag], dim=-2), torch.cat([com, last_com], dim=-2)
            pha = torch.atan2(com[:, -1], com[:, 0])

            return mag.squeeze(1), pha
         elif self.decode_type == "ri":
            rea1, rea2, rea3 = self.reg1_rea_decoder(x1_1), self.reg2_rea_decoder(x1_2), self.reg3_rea_decoder(x1_3)
            imag1, imag2, imag3 = self.reg1_imag_decoder(x2_1), self.reg2_imag_decoder(x2_2), self.reg3_imag_decoder(x2_3)
            rea = torch.cat([rea1, rea2, rea3], dim=-2) 
            imag = torch.cat([imag1, imag2, imag3], dim=-2)
            last_rea, last_imag = rea[..., -1, :].unsqueeze(-2), imag[..., -1, :].unsqueeze(-2)
            rea, imag = torch.cat([rea, last_rea], dim=-2), torch.cat([imag, last_imag], dim=-2)
            
            return rea.squeeze(1), imag.squeeze(1)
      elif sr == 44100 or sr is None:
         x1_1, x1_2, x1_3, x1_4, x1_5 = emb_input1[:, :, :12].contiguous(), \
                                        emb_input1[:, :, 12:20].contiguous(), \
                                        emb_input1[:, :, 20:26].contiguous(), \
                                        emb_input1[:, :, 26:30].contiguous(), \
                                        emb_input1[:, :, 30:].contiguous()

         x2_1, x2_2, x2_3, x2_4, x2_5 = emb_input2[:, :, :12].contiguous(), \
                                        emb_input2[:, :, 12:20].contiguous(), \
                                        emb_input2[:, :, 20:26].contiguous(), \
                                        emb_input2[:, :, 26:30].contiguous(), \
                                        emb_input2[:, :, 30:].contiguous()

         if self.decode_type == "mag+phase":
            mag1, mag2, mag3, mag4, mag5 = self.reg1_mag_decoder(x1_1), self.reg2_mag_decoder(x1_2), self.reg3_mag_decoder(x1_3), self.reg4_mag_decoder(x1_4), self.reg5_mag_decoder(x1_5)
            com1, com2, com3, com4, com5 = self.reg1_pha_decoder(x2_1), self.reg2_pha_decoder(x2_2), self.reg3_pha_decoder(x2_3), self.reg4_pha_decoder(x2_4), self.reg5_pha_decoder(x2_5)
            mag = torch.exp((torch.cat([mag1, mag2, mag3, mag4, mag5], dim=-2)).clamp_(min=-13.8, max=4.61)) 
            com = torch.cat([com1, com2, com3, com4, com5], dim=-2)
            last_mag, last_com = mag[..., -1, :].unsqueeze(-2), com[..., -1, :].unsqueeze(-2)
            mag, com = torch.cat([mag, last_mag], dim=-2), torch.cat([com, last_com], dim=-2)
            pha = torch.atan2(com[:, -1], com[:, 0])

            return mag.squeeze(1), pha
         elif self.decode_type == "ri":
            rea1, rea2, rea3, rea4, rea5 = self.reg1_rea_decoder(x1_1), self.reg2_rea_decoder(x1_2), self.reg3_rea_decoder(x1_3), self.reg4_rea_decoder(x1_4), self.reg5_rea_decoder(x1_5)
            imag1, imag2, imag3, imag4, imag5 = self.reg1_imag_decoder(x2_1), self.reg2_imag_decoder(x2_2), self.reg3_imag_decoder(x2_3), self.reg4_imag_decoder(x2_4), self.reg5_imag_decoder(x2_5)
            rea = torch.cat([rea1, rea2, rea3, rea4, rea5], dim=-2) 
            imag = torch.cat([imag1, imag2, imag3, imag4, imag5], dim=-2)
            last_rea, last_imag = rea[..., -1, :].unsqueeze(-2), imag[..., -1, :].unsqueeze(-2)
            rea, imag = torch.cat([rea, last_rea], dim=-2), torch.cat([imag, last_imag], dim=-2)
            
            return rea.squeeze(1), imag.squeeze(1)
   
   
class BandMerge_code588_NB32(nn.Module):
   def __init__(self,
               feature_dim: int = 64,
               decode_type: str = "mag+phase",
               act_type="gelu",
               ):
      super(BandMerge_code588_NB32, self).__init__()
      self.feature_dim = feature_dim
      self.decode_type = decode_type
      self.act_type = act_type
      self.eps = torch.finfo(torch.float32).eps
      
      if self.act_type == "gelu":
         act = nn.GELU()
      elif self.act_type == "tanh":
         act = nn.Tanh()
      
      self.norm1 = ChannelNormalization(feature_dim, ndim=4)
      self.norm2 = ChannelNormalization(feature_dim, ndim=4)
      if self.decode_type == "mag+phase":
         self.reg1_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(16, 1), stride=(16, 1))
         )
         self.reg2_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(32, 1), stride=(32, 1))
         )
         self.reg3_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(48, 1), stride=(48, 1))
         )
         self.reg4_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(64, 1), stride=(64, 1))
         )
         self.reg5_mag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(92, 1), stride=(92, 1))
         )
         self.reg1_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(16, 1), stride=(16, 1))
         )
         self.reg2_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(32, 1), stride=(32, 1))
         )
         self.reg3_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(48, 1), stride=(48, 1))
         )
         self.reg4_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(64, 1), stride=(64, 1))
         )
         self.reg5_pha_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=2, kernel_size=(92, 1), stride=(92, 1))
         )
      elif self.decode_type == "ri":
         self.reg1_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(16, 1), stride=(16, 1))
         )
         self.reg2_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(32, 1), stride=(32, 1))
         )
         self.reg3_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(48, 1), stride=(48, 1))
         )
         self.reg4_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(64, 1), stride=(64, 1))
         )
         self.reg5_rea_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(92, 1), stride=(92, 1))
         )
         self.reg1_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(16, 1), stride=(16, 1))
         )
         self.reg2_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(32, 1), stride=(32, 1))
         )
         self.reg3_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(48, 1), stride=(48, 1))
         )
         self.reg4_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(64, 1), stride=(64, 1))
         )
         self.reg5_imag_decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=(1, 1)),
            act,
            nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1, kernel_size=(92, 1), stride=(92, 1))
         )

   def forward(self, emb_input):
      """
      emb_input: (B, C, nband, T)
      return:
         mag: (B, F, T)
         phase: (B, F, T)
      """
      emb_input1, emb_input2 = self.norm1(emb_input), self.norm2(emb_input)
      x1_1, x1_2, x1_3, x1_4, x1_5 = emb_input1[:, :, :12].contiguous(), \
                                     emb_input1[:, :, 12:20].contiguous(), \
                                     emb_input1[:, :, 20:26].contiguous(), \
                                     emb_input1[:, :, 26:30].contiguous(), \
                                     emb_input1[:, :, 30:].contiguous()

      x2_1, x2_2, x2_3, x2_4, x2_5 = emb_input2[:, :, :12].contiguous(), \
                                     emb_input2[:, :, 12:20].contiguous(), \
                                     emb_input2[:, :, 20:26].contiguous(), \
                                     emb_input2[:, :, 26:30].contiguous(), \
                                     emb_input2[:, :, 30:].contiguous()

      if self.decode_type == "mag+phase":
         mag1, mag2, mag3, mag4, mag5 = self.reg1_mag_decoder(x1_1), self.reg2_mag_decoder(x1_2), self.reg3_mag_decoder(x1_3), self.reg4_mag_decoder(x1_4), self.reg5_mag_decoder(x1_5)
         com1, com2, com3, com4, com5 = self.reg1_pha_decoder(x2_1), self.reg2_pha_decoder(x2_2), self.reg3_pha_decoder(x2_3), self.reg4_pha_decoder(x2_4), self.reg5_pha_decoder(x2_5)
         mag = torch.exp(torch.cat([mag1, mag2, mag3, mag4, mag5], dim=-2))
         com = torch.cat([com1, com2, com3, com4, com5], dim=-2)
         last_mag, last_com = mag[..., -1, :].unsqueeze(-2), com[..., -1, :].unsqueeze(-2)
         mag, com = torch.cat([mag, last_mag], dim=-2), torch.cat([com, last_com], dim=-2)
         pha = torch.atan2(com[:, -1], com[:, 0])

         return mag.squeeze(1), pha
      elif self.decode_type == "ri":
         rea1, rea2, rea3, rea4, rea5 = self.reg1_rea_decoder(x1_1), self.reg2_rea_decoder(x1_2), self.reg3_rea_decoder(x1_3), self.reg4_rea_decoder(x1_4), self.reg5_rea_decoder(x1_5)
         imag1, imag2, imag3, imag4, imag5 = self.reg1_imag_decoder(x2_1), self.reg2_imag_decoder(x2_2), self.reg3_imag_decoder(x2_3), self.reg4_imag_decoder(x2_4), self.reg5_imag_decoder(x2_5)
         rea = torch.cat([rea1, rea2, rea3, rea4, rea5], dim=-2) 
         imag = torch.cat([imag1, imag2, imag3, imag4, imag5], dim=-2)
         last_rea, last_imag = rea[..., -1, :].unsqueeze(-2), imag[..., -1, :].unsqueeze(-2)
         rea, imag = torch.cat([rea, last_rea], dim=-2), torch.cat([imag, last_imag], dim=-2)
         
         return rea.squeeze(1), imag.squeeze(1)