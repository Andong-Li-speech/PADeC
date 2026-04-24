import torch
import torch.nn as nn

from .code_utils.code_bs import *
from .rnd_utils.norm import *
from .code_utils.basic_arch import *


class PADeC320_24k(nn.Module):
   """
   Input:
      cond: (B, C, T), e.g., summed EnCodec RVQ embeddings.
   Output:
      logamp_g: (B, F, T)
      pha_g: (B, F, T)
      ri_g: (B, 2, F, T)
      y_g: (B, L)
   """
   def __init__(self, h):
      super().__init__()
      self.h = h
      self.nblocks = h.nblocks
      self.hidden_channel = h.hidden_channel
      self.rank_reduce = h.rank_reduce
      self.f_kernel_size = h.f_kernel_size
      self.t_kernel_size = h.t_kernel_size
      self.mlp_ratio = h.mlp_ratio
      self.act_type = h.act_type
      self.basic_type = h.basic_type
      self.nb_num = h.nb_num
      self.use_even_split = h.use_even_split
      self.decode_type = h.decode_type
      self.use_band_attention = h.use_band_attention
      self.code_dim = h.code_dim

      self.n_fft = h.n_fft
      self.win_size = h.win_size
      self.hop_size = h.hop_size
      self.eps = torch.finfo(torch.float32).eps
      self.register_buffer("istft_window", torch.hann_window(self.win_size), persistent=False)

      self.enc = BandSplit_code320_24k(
         nband=self.nb_num,
         code_dim=self.code_dim,
         rank_reduce=self.rank_reduce,
         feature_dim=self.hidden_channel,
      )

      main_net = []
      for _ in range(self.nblocks):
         if self.basic_type == "conv2former":
            main_net.append(
               Conv2FormerUnit(
                  input_channel=self.hidden_channel,
                  hidden_channel=self.hidden_channel,
                  f_kernel_size=self.f_kernel_size,
                  t_kernel_size=self.t_kernel_size,
                  mlp_ratio=self.mlp_ratio,
                  act_type=self.act_type,
                  use_band_attention=self.use_band_attention,
               )
            )
         elif self.basic_type == "hor":
            main_net.append(
               HorUnit(
                  input_channel=self.hidden_channel,
                  hidden_channel=self.hidden_channel,
                  f_kernel_size=self.f_kernel_size,
                  t_kernel_size=self.t_kernel_size,
                  mlp_ratio=self.mlp_ratio,
                  act_type=self.act_type,
                  use_band_attention=self.use_band_attention,
               )
            )
         else:
            raise ValueError(f"Unsupported basic_type: {self.basic_type}")
      self.main_net = nn.ModuleList(main_net)

      if self.nb_num == 6:
         self.dec = BandMerge_code320_NB6_24k(feature_dim=self.hidden_channel, decode_type=self.decode_type)
      elif self.nb_num == 12:
         self.dec = BandMerge_code320_NB12_24k(feature_dim=self.hidden_channel, decode_type=self.decode_type)
      elif self.nb_num == 24:
         if self.use_even_split:
            self.dec = BandMerge_code320_NB24_even_24k(feature_dim=self.hidden_channel, decode_type=self.decode_type)
         else:
            self.dec = BandMerge_code320_NB24_24k(
               feature_dim=self.hidden_channel,
               decode_type=self.decode_type,
               act_type="tanh",
            )
      elif self.nb_num == 48:
         self.dec = BandMerge_code320_NB48_24k(feature_dim=self.hidden_channel, decode_type=self.decode_type)
      elif self.nb_num == 96:
         self.dec = BandMerge_code320_NB96_24k(feature_dim=self.hidden_channel, decode_type=self.decode_type)
      else:
         raise ValueError(f"Unsupported nb_num: {self.nb_num}. Supported: 6, 12, 24, 48, 96.")

      self.initialize_weights()

   def initialize_weights(self):
      def _basic_init(module):
         if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
               nn.init.constant_(module.bias, 0)
      self.apply(_basic_init)

   def forward(self, cond):
      enc_x = self.enc(cond)

      x = enc_x
      for layer in self.main_net:
         x = layer(x)

      amp_g, pha_g = self.dec(x)
      amp_g = amp_g.clamp_min(1e-6)

      logamp_g = torch.log(amp_g)
      rea_g = amp_g * torch.cos(pha_g)
      imag_g = amp_g * torch.sin(pha_g)
      ri_g = torch.stack([rea_g, imag_g], dim=1)

      y_g = torch.istft(
         torch.complex(rea_g, imag_g),
         n_fft=self.n_fft,
         hop_length=self.hop_size,
         win_length=self.win_size,
         window=self.istft_window.to(cond.device),
      )
      return logamp_g, pha_g, ri_g, y_g
