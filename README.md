<div align="center">

# 🎧 PADec: Learning Practical High-Fidelity Neural Audio Decoder via Subband-Aware Guidance

**Andong Li, Tong Lei, Lingling Dai, Rilin Chen, Meng Yu, Cunhang Fan, Xiaodong Li, Chengshi Zheng**

🏆 Submitted to Signal Processing Letters · [Paper]() . [Appendix](https://github.com/Andong-Li-speech/PADeC/blob/master/Appendix.pdf) · [Code](https://github.com/Andong-Li-speech/PADeC)

<!-- Replace the Hugging Face path after release if needed. -->
</div>

---

## ✨ Overview

**PADec** is a practical high-fidelity neural audio decoder for latent representation-based vocoding. 

The key motivation is simple: neural audio codec latents are compact and convenient for modern generative audio systems, but they often lose fine acoustic details, especially under **low-bandwidth** conditions. PADec addresses this problem with a lightweight yet expressive time-frequency decoder built around **subband-aware guidance**.

Concretely, PADec first projects codec latents into heterogeneous subband-aware spaces, then models time-frequency context using large-kernel convolutional attention, and finally reconstructs target spectra through hierarchical subband merging.

---

## 🔥 Core Ideas

- **Latent-based neural audio decoding.**  
  PADec focuses on reconstructing high-fidelity audio from compressed codec latent representations rather than Mel-spectrograms, making it naturally compatible with neural audio codecs and generative audio/token modeling systems.

- **Subband-aware guidance.**  
  Although codec latents do not explicitly form a spectrum, PADec projects them into multiple heterogeneous subband-aware spaces, encouraging fine-grained acoustic modeling across frequency regions.

- **Low-rank embedder for efficient latent projection.**  
  The proposed **LREmbedder** replaces heavy per-subband projections with low-rank Conv1d layers, substantially reducing parameter count and computational cost while preserving useful acoustic cues.

- **Large-kernel convolutional attention.**  
  PADec adopts **LKCAM**, a convolution-style attention module with large depthwise kernels, to efficiently capture both inter-frame and inter-band dependencies without expensive self-attention.

- **Hierarchical subband merge.**  
  The **HBMM** reconstructs magnitude and phase spectra through uneven subband allocation, assigning more representational capacity to perceptually important low- and mid-frequency regions.

- **Practical quality–efficiency trade-off.**  
  With only **3.63M parameters** and **33.13 GMACs per 5-second audio**, PADec achieves strong objective and subjective performance against GAN- and diffusion-based neural audio decoders.

---

## 📈 Highlights

- **High fidelity at low bandwidth.** PADec shows clear advantages in low-bitrate scenarios, especially at **1.5 kbps**.
- **Lightweight design.** Only **3.63M** parameters.
- **Efficient inference.** One-pass GAN-style decoding without iterative diffusion or flow sampling.
- **Strong subjective preference.** AB preference tests show dominant listener preference over EnCodec, RFWave, and PeriodWave-Turbo in low-bandwidth settings.

---

## 🧠 Method Overview

PADec contains three core modules:

1. **LREmbedder**  
   Projects codec features into subband-aware spaces using low-rank Conv1d projections.

2. **LKCAM**  
   Performs efficient time-frequency contextual modeling with stacked large-kernel convolutional attention blocks.

3. **HBMM**  
   Merges hierarchical subband features and estimates magnitude/phase spectra for waveform reconstruction through iSTFT.

```text
Codec Latent Feature
        │
        ▼
  Low-Rank Embedder
        │
        ▼
 Large-Kernel Convolutional Attention
        │
        ▼
 Hierarchical Subband Merge
        │
        ▼
 Magnitude + Phase → iSTFT → Waveform
```

---

## 🗂️ Project Structure

```text
PADeC/
├── cfgs/
│   └── padec_24k_encodec.json
├── ckpts/
│   └── best_g
├── filelists/
│   ├── train.example.scp
│   └── valid.example.scp
├── Models/
│   ├── padec_24k.py
│   ├── models.py
│   ├── code_utils/
│   └── rnd_utils/
├── tool_code/
│   └── feature_extractor.py
├── dataset_code.py
├── train.py
├── inference.py
└── utils.py
```

---

## 🛠️ Installation

We recommend Python **3.9+** and a CUDA-enabled PyTorch environment.

```bash
conda create -n padec python=3.9 -y
conda activate padec
```

If you prepare a `requirements.txt`, you can install by:

```bash
pip install -r requirements.txt
```

---

## 📁 Dataset Preparation

The default configuration supports both unified audio file lists and LibriTTS-style file lists.

### Unified filelist format

Each line contains one audio path:

```text
/path/to/audio_001.wav
/path/to/audio_002.wav
```

### LibriTTS-style filelist format

Each line contains a relative LibriTTS item, optionally without the `.wav` suffix:

```text
train-clean-100/19/198/19_198_000000_000000
train-clean-100/19/198/19_198_000001_000000
```

Then update the following fields in:

```text
cfgs/padec_24k_encodec.json
```

```json
"data": {
  "train_list": "./filelists/train.example.scp",
  "valid_list": "./filelists/valid.example.scp",
  "raw_wavfile_path": "/path/to/audio/root"
}
```

---

## 🚀 Training

PADec uses pretrained EnCodec latents as conditional features. The default configuration is based on 24 kHz audio and EnCodec features.

### Single-GPU training

```bash
python train.py \
  --cfg_filename cfgs/padec_24k_encodec.json \
  --gpu 0
```

### Multi-GPU training

```bash
python train.py \
  --cfg_filename cfgs/padec_24k_encodec.json \
  --gpu 0,1
```

The script will automatically resume from the latest checkpoints under:

```text
exp/padec_24k_encodec/
```

To train from scratch even when checkpoints exist:

```bash
python train.py \
  --cfg_filename cfgs/padec_24k_encodec.json \
  --gpu 0 \
  --no_resume
```

---

## 🎙️ Inference

### Bandwidth index

The default config defines four EnCodec bandwidths:

| bandwidth_idx | Bandwidth |
|--------------|-----------|
| 0 | 1.5 kbps |
| 1 | 3.0 kbps |
| 2 | 6.0 kbps |
| 3 | 12.0 kbps |

### Inference from a wav directory

```bash
python inference.py \
  --cfg_filename cfgs/padec_24k_encodec.json \
  --test_input_wavs_dir /path/to/test_wavs \
  --test_output_dir ./generated \
  --checkpoint_file_load ./ckpts/best_g.pt \
  --bandwidth_idx 0 \
  --device auto
```

### Inference from a wav filelist

```bash
python inference.py \
  --cfg_filename cfgs/padec_24k_encodec.json \
  --test_input_wavs_dir ./filelists/test.scp \
  --test_output_dir ./generated \
  --checkpoint_file_load ./ckpts/best_g.pt \
  --bandwidth_idx 1 \
  --device auto
```

### Optional normalization

```bash
python inference.py \
  --cfg_filename cfgs/padec_24k_encodec.json \
  --test_input_wavs_dir /path/to/test_wavs \
  --test_output_dir ./generated \
  --checkpoint_file_load ./ckpts/best_g.pt \
  --bandwidth_idx 0 \
  --normalize_input \
  --device auto
```

---


## 🙏 Acknowledgement

This work was supported by the National Natural Science Foundation of China (NSFC) under Grant 62501588.

---

## 📄 License

Please refer to the `LICENSE` file for details. If you use the pretrained EnCodec feature extractor, please also follow the license terms of the original EnCodec project.
