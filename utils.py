import glob
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
import torch


class AttrDict(dict):
    """Dictionary with attribute-style access.

    Nested dictionaries are converted recursively by ``to_attrdict``.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def to_attrdict(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return AttrDict({k: to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_attrdict(v) for v in obj]
    return obj


def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return normalize_config(to_attrdict(cfg))


def _copy_group_to_top(h: AttrDict, group_name: str) -> None:
    if group_name not in h:
        return
    group = h[group_name]
    if not isinstance(group, Mapping):
        return
    for key, value in group.items():
        if key not in h:
            h[key] = value
            setattr(h, key, value)


def normalize_config(h: AttrDict) -> AttrDict:
    """Support both flat configs and structured configs.

    The current model code expects flat attributes such as ``h.n_fft`` and
    ``h.code_dim``. For open-source readability, the example config is grouped
    into ``data/train/model/audio/codec/loss`` sections. This function expands
    grouped fields into the flat names expected by existing modules.
    """
    for group in ["data", "train", "model", "audio", "codec", "loss"]:
        _copy_group_to_top(h, group)

    # Common aliases used by the original training code.
    if "data" in h:
        data = h.data
        if "train_list" in data and "input_training_wav_list" not in h:
            h.input_training_wav_list = data.train_list
        if "valid_list" in data and "input_validation_wav_list" not in h:
            h.input_validation_wav_list = data.valid_list

    if "loss" in h:
        loss = h.loss
        if "weights" in loss and "loss_configs" not in h:
            h.loss_configs = loss.weights

    return h


def build_env(config: str, config_name: str, path: str) -> None:
    os.makedirs(path, exist_ok=True)
    t_path = os.path.join(path, config_name)
    if os.path.abspath(config) != os.path.abspath(t_path):
        shutil.copyfile(config, t_path)


def plot_spectrogram(spectrogram):
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    fig.canvas.draw()
    plt.close()
    return fig


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def set_random_seed(seed=1234, deterministic=False, benchmark=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True


def load_checkpoint(filepath, device="cpu"):
    filepath = str(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    print(f"Loading checkpoint: {filepath}")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Checkpoint loaded.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    filepath = str(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print(f"Saving checkpoint to {filepath}")
    torch.save(obj, filepath)
    print("Checkpoint saved.")


def clean_state_dict(state_dict):
    """Remove DDP ``module.`` prefix when needed."""
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()}


def remove_older_checkpoint(filepath, pre="g", max_to_keep=5):
    par_file_dir, filename = os.path.split(filepath)
    tracker = os.path.join(par_file_dir, f"checkpoint_{pre}")
    if os.path.exists(tracker):
        with open(tracker, "r", encoding="utf-8") as f:
            ckpts = [line.split()[0] for line in f.readlines() if line.strip()]
    else:
        ckpts = []

    ckpts.append(filename)
    for item in ckpts[:-max_to_keep]:
        path = os.path.join(par_file_dir, item)
        if os.path.exists(path):
            os.remove(path)

    with open(tracker, "w", encoding="utf-8") as f:
        for item in ckpts[-max_to_keep:]:
            f.write(f"{item}\n")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + "????????")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def get_generator_state(model, distributed=False):
    return (model.module if distributed and hasattr(model, "module") else model).state_dict()
