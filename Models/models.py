import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from torch.nn import Conv2d
from torch.nn.utils import weight_norm, spectral_norm
from utils import get_padding
from dataset_code import  mel_spectrogram
from typing import *
import numpy as np

LRELU_SLOPE = 0.1


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        16,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        16,
                        64,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        64,
                        256,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        256,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(Conv2d(512, 512, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(512, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        if x.ndim == 2:
            x= x.unsqueeze(1)  # (B, 1, L)

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, mpd_reshapes):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(mpd_reshapes[0]),
                DiscriminatorP(mpd_reshapes[1]),
                DiscriminatorP(mpd_reshapes[2]),
                DiscriminatorP(mpd_reshapes[3]),
                DiscriminatorP(mpd_reshapes[-1]),
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for _, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def phase_loss(phase_r, phase_g, n_fft, frames):
    GD_matrix = (
        torch.triu(torch.ones(n_fft // 2 + 1, n_fft // 2 + 1), diagonal=1)
        - torch.triu(torch.ones(n_fft // 2 + 1, n_fft // 2 + 1), diagonal=2)
        - torch.eye(n_fft // 2 + 1)
    )
    GD_matrix = GD_matrix.to(phase_g.device)

    GD_r = torch.matmul(phase_r.permute(0, 2, 1), GD_matrix)
    GD_g = torch.matmul(phase_g.permute(0, 2, 1), GD_matrix)

    PTD_matrix = (
        torch.triu(torch.ones(frames, frames), diagonal=1)
        - torch.triu(torch.ones(frames, frames), diagonal=2)
        - torch.eye(frames)
    )
    PTD_matrix = PTD_matrix.to(phase_g.device)

    PTD_r = torch.matmul(phase_r, PTD_matrix)
    PTD_g = torch.matmul(phase_g, PTD_matrix)

    IP_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    GD_loss = torch.mean(anti_wrapping_function(GD_r - GD_g))
    PTD_loss = torch.mean(anti_wrapping_function(PTD_r - PTD_g))
    loss = IP_loss + GD_loss + PTD_loss
    return loss


# for omni-phase-loss related
def linear_anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

def cos_anti_wrapping_function(x):
    return np.pi * (1 - torch.cos(x)) / 2

def square_anti_wrapping_function(x):
    return torch.square(x - torch.round(x / (2 * np.pi)) * 2 * np.pi) / np.pi

def log_anti_wrapping_function(x):
    return np.pi * torch.log(torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi) + 1) / np.log(np.pi + 1)

def cubic_anti_wrapping_function(x):
    return (np.pi / 2) + ((torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi) - np.pi / 2) ** 3) * 4 / (np.pi ** 2)


class Weighted_OmniPhaseLoss(nn.Module):
    def __init__(self, anti_type="linear", use_mag_weighted=True, alpha=100):
        super(Weighted_OmniPhaseLoss, self).__init__()
        self.anti_type = anti_type
        self.use_mag_weighted = use_mag_weighted
        self.alpha = alpha
        kernel1 = torch.from_numpy(np.array([[-1., 0, 0], [0, 1, 0], [0, 0, 0]], dtype='float32'))
        kernel2 = torch.from_numpy(np.array([[0, -1., 0], [0, 1., 0], [0, 0, 0]], dtype='float32'))
        kernel3 = torch.from_numpy(np.array([[0, 0, -1.], [0, 1., 0], [0, 0, 0]], dtype='float32'))
        kernel4 = torch.from_numpy(np.array([[0, 0, 0], [-1., 1., 0], [0, 0, 0]], dtype='float32'))
        kernel5 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., 0], [0, 0, 0]], dtype='float32'))
        kernel6 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., -1.], [0, 0, 0]], dtype='float32'))
        kernel7 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., 0], [-1., 0, 0]], dtype='float32'))
        kernel8 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., 0], [0, -1., 0]], dtype='float32'))
        kernel9 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., 0], [0, 0, -1.]], dtype='float32'))
        kernels = torch.stack([kernel1, kernel2, kernel3, kernel4, kernel5, kernel6, kernel7, kernel8, kernel9], dim=0)  # (out_nch, 3, 3)
        kernels = kernels.unsqueeze(1)
        self.filters = kernels

    def forward(self, phase_r, phase_g, mag_r=None):
        """
        phase_r: (B, F, T)
        phase_g: (B, F, T)
        mag_r: (B, F, T)
        """
        if self.anti_type.lower() == "linear":
            func = "linear_anti_wrapping_function"
        elif self.anti_type.lower() == "cos":
            func = "cos_anti_wrapping_function"
        elif self.anti_type.lower() == "square":
            func = "square_anti_wrapping_function"
        elif self.anti_type.lower() == "log":
            func = "log_anti_wrapping_function"
        elif self.anti_type.lower() == "cubic_anti_wrapping_function":
            func = "cubic_anti_wrapping_function"

        if mag_r.ndim == 3:
            mag_r = mag_r.unsqueeze(1)  # (B, 1, F, T)

        mag_r = (mag_r / (torch.max(mag_r) + 1e-6) * self.alpha).transpose(-2, -1).contiguous()

        phase_r = phase_r.transpose(-2, -1).unsqueeze(1)  # (B,1,T,F)
        phase_g = phase_g.transpose(-2, -1).unsqueeze(1)  # (B,1,T,F)
        loss = 0
        
        phase_r = F.conv2d(phase_r, self.filters.to(phase_r.device), bias=None, stride=1, padding=1)  # (B,9,T,F)
        phase_g = F.conv2d(phase_g, self.filters.to(phase_r.device), bias=None, stride=1, padding=1)  # (B,9,T,F)
        if self.use_mag_weighted:
            loss = loss + 3 * torch.mean(mag_r * eval(func)(phase_g - phase_r))
        else:
            loss = loss + 3 * torch.mean(eval(func)(phase_g - phase_r))

        return loss


# for omni-ri-loss related
def norm_linear_anti_wrapping_function(x, gamma=0):
    """
    range: (gamma, gamma + 1)
    """
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi) / np.pi + gamma

def norm_cos_anti_wrapping_function(x, gamma=0):
    """
    range: (gamma, gamma + 1)
    """
    return (1 - torch.cos(x)) / 2 + gamma

def norm_square_anti_wrapping_function(x, gamma=0):
    """
    range: (gamma, gamma + 1)
    """
    return torch.square(x - torch.round(x / (2 * np.pi)) * 2 * np.pi) / (np.pi ** 2) + gamma

def norm_log_anti_wrapping_function(x, gamma=0):
    """
    range: (gamma, gamma + 1)
    """
    return torch.log(torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi) + 1) / np.log(np.pi + 1) + gamma

def norm_cubic_anti_wrapping_function(x, gamma=0):
    """
    range: (gamma, gamma + 1)
    """
    return (1 / 2) + ((torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi) - np.pi / 2) ** 3) * 4 / (np.pi ** 3) + gamma


class OmniRILoss(nn.Module):
    def __init__(self, anti_type="linear"):
        super(OmniRILoss, self).__init__()
        self.anti_type = anti_type

        kernel1 = torch.from_numpy(np.array([[-1., 0, 0], [0, 1, 0], [0, 0, 0]], dtype='float32'))
        kernel2 = torch.from_numpy(np.array([[0, -1., 0], [0, 1., 0], [0, 0, 0]], dtype='float32'))
        kernel3 = torch.from_numpy(np.array([[0, 0, -1.], [0, 1., 0], [0, 0, 0]], dtype='float32'))
        kernel4 = torch.from_numpy(np.array([[0, 0, 0], [-1., 1., 0], [0, 0, 0]], dtype='float32'))
        kernel5 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., 0], [0, 0, 0]], dtype='float32'))
        kernel6 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., -1.], [0, 0, 0]], dtype='float32'))
        kernel7 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., 0], [-1., 0, 0]], dtype='float32'))
        kernel8 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., 0], [0, -1., 0]], dtype='float32'))
        kernel9 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., 0], [0, 0, -1.]], dtype='float32'))
        kernels = torch.stack([kernel1, kernel2, kernel3, kernel4, kernel5, kernel6, kernel7, kernel8, kernel9], dim=0)  # (out_nch, 3, 3)
        kernels = kernels.unsqueeze(1)
        self.filters = kernels

    def forward(self, rea, imag, rea_g, imag_g):
        """
        rea: target real, (B, F, T)
        imag: target imaginary, (B, F, T)
        rea_g: estimate real, (B, F, T)
        imag_g: estimate imaginary, (B, F, T)
        """
        mag, mag_g = torch.sqrt(rea ** 2 + imag ** 2 + 1e-8).unsqueeze(1).repeat(1, self.filters.shape[0], 1, 1), \
                     torch.sqrt(rea_g ** 2 + imag_g ** 2 + 1e-8).unsqueeze(1).repeat(1, self.filters.shape[0], 1, 1)
        pha, pha_g = torch.atan2(imag, rea).unsqueeze(1), \
                     torch.atan2(imag_g, rea_g).unsqueeze(1)

        pha = F.conv2d(pha, self.filters.to(pha.device), bias=None, stride=1, padding=1)
        pha_g = F.conv2d(pha_g, self.filters.to(pha.device), bias=None, stride=1, padding=1)
        
        com = torch.cat([mag * torch.cos(pha), mag * torch.sin(pha)], dim=1)
        com_g = torch.cat([mag_g * torch.cos(pha_g), mag_g * torch.sin(pha_g)], dim=1)
        
        loss = torch.mean(torch.abs(com - com_g))
            
        return loss


class RILoss(nn.Module):
    def __init__(self):
        super(RILoss, self).__init__()
    
    def forward(self, rea, imag, rea_g, imag_g):
        """
        rea: target real, (B, F, T)
        imag: target imaginary, (B, F, T)
        rea_g: estimate real, (B, F, T)
        imag_g: estimate imaginary, (B, F, T)
        """
        loss_r = torch.abs(rea - rea_g).mean()
        loss_g = torch.abs(imag - imag_g).mean()
        return (loss_r + loss_g) / 2


# Multi-scale mel loss
class MultiResolutionMelLoss(nn.Module):
    def __init__(self,
                 resolutions=((32, 8, 32, 5),
                              (64, 16, 64, 10),
                              (128, 32, 128, 20),
                              (256, 64, 256, 40),
                              (512, 128, 512, 80),
                              (1024, 256, 1024, 160),
                              (2048, 512, 2048, 320),
                              ),
                sampling_rate=24000,
    ):
        super(MultiResolutionMelLoss, self).__init__()
        self.resolutions = resolutions
        self.sampling_rate = sampling_rate
    
    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        loss_tot = 0.
        for i, cur_reso in enumerate(self.resolutions):
            y_mel = mel_spectrogram(y, 
                                    n_fft=cur_reso[0], 
                                    num_mels=cur_reso[-1],
                                    sampling_rate=self.sampling_rate,
                                    hop_size=cur_reso[1],
                                    win_size=cur_reso[2],
                                    fmin=0,
                                    fmax=self.sampling_rate / 2,
                                    )
            y_hat_mel = mel_spectrogram(y_hat, 
                                        n_fft=cur_reso[0], 
                                        num_mels=cur_reso[-1],
                                        sampling_rate=self.sampling_rate,
                                        hop_size=cur_reso[1],
                                        win_size=cur_reso[2],
                                        fmin=0,
                                        fmax=self.sampling_rate / 2,
                                        )
            loss_tot = loss_tot + torch.abs(y_mel - y_hat_mel).mean()
        loss_tot = loss_tot / len(self.resolutions)
        return loss_tot


class MultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        resolutions=(
                    (1024, 256, 1024), 
                    (2048, 512, 2048), 
                    (512, 128, 512),
                    (256, 64, 256)
                    ),
        num_embeddings: int = None,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorR(resolution=r, num_embeddings=num_embeddings)
                for r in resolutions
            ]
        )

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor, bandwidth_id: torch.Tensor = None
    ):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y, cond_embedding_id=bandwidth_id)
            y_d_g, fmap_g = d(x=y_hat, cond_embedding_id=bandwidth_id)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
    def __init__(
        self,
        resolution,
        channels: int = 64,
        in_channels: int = 1,
        num_embeddings: int = None,
        lrelu_slope: float = 0.1,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.lrelu_slope = lrelu_slope
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv2d(
                        in_channels,
                        channels,
                        kernel_size=(7, 5),
                        stride=(2, 2),
                        padding=(3, 2),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(5, 3),
                        stride=(2, 1),
                        padding=(2, 1),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(5, 3),
                        stride=(2, 2),
                        padding=(2, 1),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels, channels, kernel_size=3, stride=(2, 1), padding=1
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels, channels, kernel_size=3, stride=(2, 2), padding=1
                    )
                ),
            ]
        )
        if num_embeddings is not None:
            self.emb = torch.nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=channels
            )
            torch.nn.init.zeros_(self.emb.weight)
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor = None):
        fmap = []
        if x.ndim == 3:
            x = x.squeeze(1)

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        if cond_embedding_id is not None:
            emb = self.emb(cond_embedding_id)
            h = (emb.view(1, -1, 1, 1) * x).sum(dim=1, keepdims=True)
        else:
            h = 0
        x = self.conv_post(x)
        fmap.append(x)
        x += h
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x: torch.Tensor):
        n_fft, hop_length, win_length = self.resolution
        magnitude_spectrogram = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length).to(x.device),  # interestingly rectangular window kind of works here
            center=True,
            return_complex=True,
        ).abs()

        return magnitude_spectrogram


def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * math.pi)) * 2 * math.pi)


def amplitude_loss(log_amplitude_r, log_amplitude_g):
    MSELoss = torch.nn.MSELoss()

    amplitude_loss = MSELoss(log_amplitude_r, log_amplitude_g)

    return amplitude_loss


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean(torch.nan_to_num_(torch.clamp(1 - dr, min=0)))
        g_loss = torch.mean(torch.nan_to_num_(torch.clamp(1 + dg, min=0)))
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean(torch.nan_to_num_(torch.clamp(1 - dg, min=0)))
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def ls_discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def ls_generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean(torch.nan_to_num_(torch.abs(1 - dg)) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def STFT_consistency_loss(rea_r, rea_g, imag_r, imag_g):
    C_loss = torch.mean(
        torch.mean((rea_r - rea_g) ** 2 + (imag_r - imag_g) ** 2, (1, 2))
    )

    return C_loss
