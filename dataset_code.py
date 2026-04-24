import os
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import librosa
import numpy as np
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

try:
    import torchaudio
except ImportError:
    torchaudio = None


def load_wav(full_path, sample_rate):
    data, _ = librosa.load(full_path, sr=sample_rate, mono=True)
    return data.astype(np.float32)


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
    center=True,
    in_dataset=False,
):
    device = torch.device("cpu") if in_dataset else y.device
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
        center=center,
        return_complex=True,
    )
    spec = mel_basis @ spec.abs()
    return spectral_normalize_torch(spec)


def spectrogram(
    y,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    center=True,
    in_dataset=False,
):
    device = torch.device("cpu") if in_dataset else y.device
    key = _cache_key(sampling_rate, n_fft, num_mels, fmin, fmax, win_size, device)
    if key not in _mel_cache:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis = torch.from_numpy(mel).float().to(device)
        hann_window = torch.hann_window(win_size).to(device)
        _mel_cache[key] = (mel_basis, hann_window)
    _, hann_window = _mel_cache[key]

    spec = torch.stft(
        y.to(device),
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        return_complex=True,
    ).abs()
    return torch.log(spec.clamp_min_(1e-5))


def inverse_mel(
    mel,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    in_dataset=False,
):
    device = torch.device("cpu") if in_dataset else mel.device
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


def amp_pha_specturm(y, n_fft, hop_size, win_size):
    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=True,
        return_complex=True,
    )
    log_amplitude = torch.log(stft_spec.abs() + 1e-7)
    phase = torch.atan2(stft_spec.imag, stft_spec.real)
    return log_amplitude, phase, stft_spec.real, stft_spec.imag


def _resolve_line_to_wav(line: str, dataset_type: str, raw_wavfile_path: str) -> str:
    line = line.strip()
    if not line:
        return ""

    # Allow optional metadata after '|'.
    item = line.split("|")[0]
    dataset_type = dataset_type.lower()

    if os.path.isabs(item) and os.path.isfile(item):
        return item

    if dataset_type == "libritts":
        # LibriTTS lists often omit the .wav suffix.
        if not item.endswith(".wav"):
            item = f"{item}.wav"
        return os.path.join(raw_wavfile_path, item)

    # Unified lists are expected to contain absolute paths. If users provide
    # relative paths, resolve them relative to raw_wavfile_path when given.
    if raw_wavfile_path and not os.path.isabs(item):
        return os.path.join(raw_wavfile_path, item)
    return item


def read_filelist(path: str, dataset_type: str = "unified", raw_wavfile_path: str = ""):
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        files = [_resolve_line_to_wav(line, dataset_type, raw_wavfile_path) for line in f]
    return [p for p in files if p]


def get_dataset_filelist(training_dataset_type, validation_dataset_type, input_training_wav_list, input_validation_wav_list, raw_wavfile_path):
    training_files = read_filelist(input_training_wav_list, training_dataset_type, raw_wavfile_path)
    validation_files = read_filelist(input_validation_wav_list, validation_dataset_type, raw_wavfile_path)
    return training_files, validation_files


def peak_normalize(audio: torch.Tensor, gain_db: float, eps: float = 1e-8):
    target_peak = 10.0 ** (gain_db / 20.0)
    peak = audio.abs().max().clamp_min(eps)
    return audio / peak * target_peak


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        train_type,
        training_files,
        segment_size,
        sampling_rate,
        split=True,
        shuffle=True,
        device=None,
        max_segment_size=None,
        file_num=-1,
        n_cache_reuse=1,
        seed=39087,
        normalize=True,
        train_gain_range=(-10.0, -1.0),
        valid_gain=-3.0,
    ):
        self.type = train_type
        self.audio_files = list(training_files)
        self.rng = random.Random(seed)
        if shuffle:
            self.rng.shuffle(self.audio_files)

        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.device = device
        self.max_segment_size = max_segment_size
        self.file_num = file_num
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.normalize = normalize
        self.train_gain_range = train_gain_range
        self.valid_gain = valid_gain

    def __len__(self):
        return len(self.audio_files) if self.file_num <= 0 else self.file_num

    def _load_audio(self, index):
        filename = self.audio_files[index % len(self.audio_files)]
        if self._cache_ref_count == 0:
            audio = load_wav(filename, self.sampling_rate)
            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1
        return torch.from_numpy(audio).float().unsqueeze(0)

    def __getitem__(self, index):
        audio = self._load_audio(index)

        if self.split:
            if audio.size(1) >= self.segment_size:
                max_audio_start = audio.size(1) - self.segment_size
                audio_start = self.rng.randint(0, max_audio_start)
                audio = audio[:, audio_start: audio_start + self.segment_size]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), "constant")

        if self.normalize:
            gain = self.rng.uniform(*self.train_gain_range) if self.type == "train" else self.valid_gain
            # Prefer sox's norm when available, but keep a pure PyTorch fallback
            # so the dataset works even on systems without sox support.
            if torchaudio is not None:
                try:
                    audio, _ = torchaudio.sox_effects.apply_effects_tensor(
                        audio, self.sampling_rate, [["norm", f"{gain:.2f}"]]
                    )
                except Exception:
                    audio = peak_normalize(audio, gain)
            else:
                audio = peak_normalize(audio, gain)

        if self.max_segment_size is not None and audio.shape[-1] > self.max_segment_size:
            audio = audio[:, : self.max_segment_size]

        return audio.squeeze(0)
