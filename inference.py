from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List

import librosa as lib
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from Models.padec_24k import PADeC320_24k
from tool_code.feature_extractor import EncodecFeatures
from utils import clean_state_dict, load_checkpoint, load_config


MODEL_REGISTRY = {
    "PADeC320_24k": PADeC320_24k,
}


def get_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device(device_arg)


def build_generator(h, device):
    if h.model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model_name={h.model_name}. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[h.model_name](h).to(device)


def build_codec_feature_extractor(h, device):
    codec_type = getattr(h, "codec_type", "encodec").lower()
    if codec_type == "encodec":
        return EncodecFeatures(
            encodec_model=getattr(h, "encodec_model", "encodec_24khz"),
            bandwidths=getattr(h, "bandwidths", [1.5, 3.0, 6.0, 12.0]),
            train_codebooks=getattr(h, "train_codebooks", False),
        ).to(device)
    raise ValueError(f"Unsupported codec_type={codec_type}. Supported: encodec, dac.")


def load_generator_weights(model, checkpoint_path, device):
    ckpt = load_checkpoint(checkpoint_path, device)
    state = ckpt["generator"] if isinstance(ckpt, dict) and "generator" in ckpt else ckpt
    state = clean_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Warning] Missing keys: {missing}")
    if unexpected:
        print(f"[Warning] Unexpected keys: {unexpected}")


def ensure_mono(audio: np.ndarray):
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)

def load_audio(path: str, sampling_rate: int):
    audio, sr = sf.read(path)
    audio = ensure_mono(audio).astype(np.float32)
    if sr != sampling_rate:
        audio = lib.resample(audio, orig_sr=sr, target_sr=sampling_rate)
    return torch.from_numpy(audio).float().unsqueeze(0)

def normalize_peak(audio: torch.Tensor, target_peak: float = 0.5, eps: float = 1e-8):
    peak = audio.abs().max().clamp_min(eps)
    return audio / peak * target_peak, peak / target_peak

def resolve_line_to_wav(line: str, dataset_type: str, raw_wavfile_path: str):
    item = line.strip().split("|")[0]
    if not item:
        return ""
    if os.path.isabs(item) and os.path.isfile(item):
        return item
    if dataset_type.lower() == "libritts":
        if not item.endswith(".wav"):
            item = f"{item}.wav"
        return os.path.join(raw_wavfile_path, item)
    if raw_wavfile_path and not os.path.isabs(item):
        return os.path.join(raw_wavfile_path, item)
    return item

def resolve_wav_list(input_path: str, dataset_type: str = "unified", raw_wavfile_path: str = ""):
    input_path = str(input_path)
    suffix = Path(input_path).suffix.lower()

    if suffix in [".txt", ".scp"]:
        with open(input_path, "r", encoding="utf-8") as f:
            files = [resolve_line_to_wav(line, dataset_type, raw_wavfile_path) for line in f]
        return [p for p in files if p]

    p = Path(input_path)
    if p.is_dir():
        valid_suffixes = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
        return [str(x) for x in sorted(p.rglob("*")) if x.is_file() and x.suffix.lower() in valid_suffixes]
    if p.is_file():
        return [str(p)]

    raise FileNotFoundError(f"Input path not found: {input_path}")

@torch.no_grad()
def run_inference(args, h):
    device = get_device(args.device)
    print(f"Using device: {device}")

    bandwidths = list(getattr(h, "bandwidths", [1.5, 3.0, 6.0, 12.0]))
    if args.bandwidth_idx < 0 or args.bandwidth_idx >= len(bandwidths):
        raise ValueError(f"--bandwidth_idx must be in [0, {len(bandwidths)-1}], got {args.bandwidth_idx}.")
    bandwidth = bandwidths[args.bandwidth_idx]

    out_dir = Path(args.test_output_dir)
    if args.separate_bandwidth_dir:
        out_dir = out_dir / f"{bandwidth}kbps"
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = build_generator(h, device)
    load_generator_weights(generator, args.checkpoint_file_load, device)
    generator.eval()

    codec_feature_extractor = build_codec_feature_extractor(h, device)
    codec_feature_extractor.eval()

    filelist = resolve_wav_list(
        args.test_input_wavs_dir,
        dataset_type=getattr(h, "dataset_type", "unified"),
        raw_wavfile_path=getattr(h, "raw_wavfile_path", ""),
    )
    if len(filelist) == 0:
        raise RuntimeError("No input audio files found.")

    print(f"Found {len(filelist)} input files.")
    print(f"Bandwidth: {bandwidth} kbps (index={args.bandwidth_idx})")

    total_samples = 0
    start = time.time()

    for wav_path in tqdm(filelist, desc="Inferencing"):
        wav_path = str(wav_path)
        audio = load_audio(wav_path, h.sampling_rate).to(device)

        scale = None
        if args.normalize_input:
            audio, scale = normalize_peak(audio, target_peak=args.normalize_peak)

        cond = codec_feature_extractor(audio, bandwidth_id=args.bandwidth_idx)
        output = generator(cond)
        y_g = output[-1] if isinstance(output, (list, tuple)) else output
        if y_g.ndim == 3 and y_g.shape[1] == 1:
            y_g = y_g.squeeze(1)

        if scale is not None:
            y_g = y_g * scale

        y = y_g.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if args.normalize_output:
            peak = max(np.max(np.abs(y)), 1e-8)
            y = args.normalize_peak * y / peak

        out_path = out_dir / (Path(wav_path).stem + ".wav")
        sf.write(out_path, y, h.sampling_rate, subtype="PCM_16")
        total_samples += len(y)

    elapsed = time.time() - start
    audio_seconds = total_samples / float(h.sampling_rate)
    print(f"Elapsed time: {elapsed:.3f} s")
    print(f"Generated audio duration: {audio_seconds:.3f} s")
    print(f"Throughput: {audio_seconds / max(elapsed, 1e-8):.3f} audio-sec/sec")

def parse_args():
    parser = argparse.ArgumentParser("PAdec inference")
    parser.add_argument("--cfg_filename", type=str, required=True,
                        help="Path to config json.")
    parser.add_argument("--test_input_wavs_dir", type=str, required=True,
                        help="Wav file, directory, or .txt/.scp list.")
    parser.add_argument("--test_output_dir", type=str, required=True,
                        help="Output directory.")
    parser.add_argument("--checkpoint_file_load", type=str, required=True,
                        help="Generator checkpoint.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--bandwidth_idx", type=int, help="Bandwidth index from config.bandwidths.")
    parser.add_argument("--separate_bandwidth_dir", action="store_true", help="Save under <output>/<bandwidth>kbps/.")
    parser.add_argument("--normalize_input", action="store_true", help="Peak-normalize input before codec extraction, then restore scale.")
    parser.add_argument("--normalize_output", action="store_true", help="Peak-normalize output before saving.")
    parser.add_argument("--normalize_peak", type=float, default=0.5, help="Target peak for normalization.")
    return parser.parse_args()


def main():
    args = parse_args()
    h = load_config(args.cfg_filename)
    run_inference(args, h)


if __name__ == "__main__":
    main()
