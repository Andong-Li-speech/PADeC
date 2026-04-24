from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
from torch import nn
from librosa.filters import mel as librosa_mel_fn


def _to_bandwidth_index(bandwidth_id) -> int:
    if isinstance(bandwidth_id, torch.Tensor):
        if bandwidth_id.numel() != 1:
            raise ValueError("bandwidth_id must be a scalar or a single-element tensor.")
        return int(bandwidth_id.detach().cpu().item())
    return int(bandwidth_id)


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes):
    return dynamic_range_decompression_torch(magnitudes)


_mel_cache = {}
_inv_mel_cache = {}


def _cache_key(sampling_rate, n_fft, num_mels, fmin, fmax, win_size, device):
    return f"{sampling_rate}-{n_fft}-{num_mels}-{fmin}-{fmax}-{win_size}-{device}"


def mel_spectrogram(
    y,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
):
    device = y.device
    key = _cache_key(sampling_rate, n_fft, num_mels, fmin, fmax, win_size, device)
    if key not in _mel_cache:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis = torch.from_numpy(mel).float().to(device)
        hann_window = torch.hann_window(win_size).to(device)
        _mel_cache[key] = (mel_basis, hann_window)
    mel_basis, hann_window = _mel_cache[key]

    spec = torch.stft(
        y.to(device),
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=True,
        return_complex=True,
    )
    spec = mel_basis @ spec.abs()
    return spectral_normalize_torch(spec)


def inverse_mel(
    mel,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
):
    device = mel.device
    key = _cache_key(sampling_rate, n_fft, num_mels, fmin, fmax, win_size, device)
    if key not in _inv_mel_cache:
        if key not in _mel_cache:
            mel_np = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
            mel_basis = torch.from_numpy(mel_np).float().to(device)
            hann_window = torch.hann_window(win_size).to(device)
            _mel_cache[key] = (mel_basis, hann_window)
        mel_basis, _ = _mel_cache[key]
        _inv_mel_cache[key] = mel_basis.pinverse()
    mel_basis, _ = _mel_cache[key]
    inv_basis = _inv_mel_cache[key]
    return mel_basis.to(device), inv_basis.to(device), inv_basis.to(device) @ spectral_de_normalize_torch(mel.to(device))


class MelSpectrogramFeatures(nn.Module):
    def __init__(
        self,
        sample_rate=24000,
        n_fft=1024,
        win_size=1024,
        hop_size=256,
        num_mels=100,
        fmin=0,
        fmax=12000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.num_mels = num_mels

    def mel_forward(self, audio):
        return mel_spectrogram(
            audio,
            n_fft=self.n_fft,
            num_mels=self.num_mels,
            sampling_rate=self.sample_rate,
            hop_size=self.hop_size,
            win_size=self.win_size,
            fmin=self.fmin,
            fmax=self.fmax,
        )

    def inverse_mel_forward(self, inpt):
        return inverse_mel(
            inpt,
            n_fft=self.n_fft,
            num_mels=self.num_mels,
            sampling_rate=self.sample_rate,
            hop_size=self.hop_size,
            win_size=self.win_size,
            fmin=self.fmin,
            fmax=self.fmax,
        )[-1].abs().clamp_min_(1e-6)


class EncodecFeatures(nn.Module):
    """Extract continuous EnCodec latent features from audio.

    The module keeps the official EnCodec encoder and quantizer frozen, then
    converts discrete RVQ codes back to summed codebook embeddings. The output
    shape is ``(B, C, T)``.
    """
    def __init__(
        self,
        encodec_model: str = "encodec_24khz",
        bandwidths: Sequence[float] = (1.5, 3.0, 6.0, 12.0),
        train_codebooks: bool = False,
    ):
        super().__init__()
        try:
            from encodec import EncodecModel
        except ImportError as exc:
            raise ImportError(
                "EncodecFeatures requires the `encodec` package. "
                "Install it with `pip install encodec`."
            ) from exc

        if encodec_model == "encodec_24khz":
            model_fn = EncodecModel.encodec_model_24khz
        elif encodec_model == "encodec_48khz":
            model_fn = EncodecModel.encodec_model_48khz
        else:
            raise ValueError(
                f"Unsupported encodec_model: {encodec_model}. "
                "Use 'encodec_24khz' or 'encodec_48khz'."
            )

        self.encodec = model_fn(pretrained=True)
        for param in self.encodec.parameters():
            param.requires_grad = False

        self.bandwidths = list(bandwidths)
        self.num_q = self.encodec.quantizer.get_num_quantizers_for_bandwidth(
            self.encodec.frame_rate, bandwidth=max(self.bandwidths)
        )
        codebook_weights = torch.cat(
            [vq.codebook for vq in self.encodec.quantizer.vq.layers[: self.num_q]],
            dim=0,
        )
        self.codebook_weights = nn.Parameter(codebook_weights, requires_grad=train_codebooks)

    @torch.no_grad()
    def get_encodec_codes(self, audio):
        if audio.ndim != 2:
            raise ValueError(f"Expected audio shape (B, L), got {tuple(audio.shape)}")
        emb = self.encodec.encoder(audio.unsqueeze(1))
        codes = self.encodec.quantizer.encode(emb, self.encodec.frame_rate, self.encodec.bandwidth)
        return codes

    def forward(self, audio: torch.Tensor, *, bandwidth_id=None):
        if bandwidth_id is None:
            raise ValueError("The `bandwidth_id` argument is required.")
        bandwidth_idx = _to_bandwidth_index(bandwidth_id)
        if bandwidth_idx < 0 or bandwidth_idx >= len(self.bandwidths):
            raise ValueError(f"bandwidth_id={bandwidth_idx} is out of range for {self.bandwidths}.")

        self.encodec.eval()
        self.encodec.set_target_bandwidth(self.bandwidths[bandwidth_idx])
        codes = self.get_encodec_codes(audio)

        offsets = torch.arange(
            0,
            self.encodec.quantizer.bins * len(codes),
            self.encodec.quantizer.bins,
            device=audio.device,
        )
        embedding_idxs = codes + offsets.view(-1, 1, 1)
        features = torch.nn.functional.embedding(embedding_idxs, self.codebook_weights).sum(dim=0)
        return features.transpose(1, 2)  # (B, C, T)

    @torch.no_grad()
    def decode(self, emb):
        return self.encodec.decoder(emb).squeeze(1)


class DACFeatures(nn.Module):
    """Optional DAC feature extractor.

    DAC dependencies are loaded lazily so users who only use EnCodec do not need
    to install DAC.
    """
    def __init__(
        self,
        sample_rate: int,
        pretrained_path: str,
        bandwidths: Sequence[float] = (1.5, 3.0, 6.0, 12.0),
    ):
        super().__init__()
        try:
            import dac
        except ImportError as exc:
            raise ImportError(
                "DACFeatures requires the `descript-audio-codec` package. "
                "Install DAC only if you plan to use codec_type='dac'."
            ) from exc

        self.sample_rate = sample_rate
        self.bandwidths = list(bandwidths)
        self.model = dac.DAC.load(pretrained_path)
        for param in self.model.parameters():
            param.requires_grad = False
        self.frame_rate = sample_rate / np.prod(self.model.encoder_rates)
        self.kbps = self.frame_rate * np.log2(self.model.codebook_size) / 1000

    @torch.no_grad()
    def forward(self, audio: torch.Tensor, *, bandwidth_id=None):
        if bandwidth_id is None:
            raise ValueError("The `bandwidth_id` argument is required.")
        bandwidth_idx = _to_bandwidth_index(bandwidth_id)
        num_quantizers = int(self.bandwidths[bandwidth_idx] / self.kbps)
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        x = self.model.preprocess(audio, self.sample_rate)
        return self.model.encode(x, n_quantizers=num_quantizers)[0]


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val))
