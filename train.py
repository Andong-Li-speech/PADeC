from __future__ import annotations

import argparse
import datetime
import itertools
import os
import sys
import time

# Set CUDA_VISIBLE_DEVICES before importing torch if --gpu is provided.
# This makes `python train.py --gpu 2,3 ...` reliable.
def _preparse_gpu_env():
    if "--gpu" not in sys.argv:
        return
    idx = sys.argv.index("--gpu")
    if idx + 1 < len(sys.argv):
        gpu = sys.argv[idx + 1]
        if gpu:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu

_preparse_gpu_env()

import librosa as lib
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from pesq import pesq
except Exception:
    pesq = None

try:
    from ptflops import get_model_complexity_info
except Exception:
    get_model_complexity_info = None

from dataset_code import Dataset, amp_pha_specturm, get_dataset_filelist, spectrogram
from tool_code.feature_extractor import EncodecFeatures

from Models.padec_24k import PADeC320_24k
from Models.models import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    generator_loss,
    discriminator_loss,
    amplitude_loss,
    Weighted_OmniPhaseLoss,
    OmniRILoss,
    RILoss,
    STFT_consistency_loss,
    MultiResolutionMelLoss,
    feature_loss,
)
from utils import (
    build_env,
    clean_state_dict,
    get_generator_state,
    load_checkpoint,
    load_config,
    plot_spectrogram,
    remove_older_checkpoint,
    save_checkpoint,
    scan_checkpoint,
    set_random_seed,
)

MODEL_REGISTRY = {
    "PADeC320_24k": PADeC320_24k,
}

def is_invalid_loss(tensor):
    return isinstance(tensor, torch.Tensor) and (torch.isnan(tensor).any() or torch.isinf(tensor).any())

def build_generator(h):
    if h.model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model_name={h.model_name}. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[h.model_name](h)

def build_codec_feature_extractor(h, device):
    codec_type = getattr(h, "codec_type", "encodec").lower()
    if codec_type == "encodec":
        return EncodecFeatures(
            encodec_model=getattr(h, "encodec_model", "encodec_24khz"),
            bandwidths=getattr(h, "bandwidths", [1.5, 3.0, 6.0, 12.0]),
            train_codebooks=getattr(h, "train_codebooks", False),
        ).to(device)
    raise ValueError(f"Unsupported codec_type={codec_type}. Supported: encodec.")

def worker_init_fn(worker_id):
    # Make dataset augmentation deterministic but different across workers.
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)

def load_resume_if_available(generator, mpd, mrd, h, device, resume=True):
    steps = 0
    last_epoch = -1
    state_dict_do = None

    if not resume:
        return steps, last_epoch, state_dict_do

    cp_g = scan_checkpoint(h.checkpoint_path, "g_") if os.path.isdir(h.checkpoint_path) else None
    cp_do = scan_checkpoint(h.checkpoint_path, "do_") if os.path.isdir(h.checkpoint_path) else None

    if cp_g is None or cp_do is None:
        return steps, last_epoch, state_dict_do

    state_dict_g = load_checkpoint(cp_g, device)
    state_dict_do = load_checkpoint(cp_do, device)

    g_state = state_dict_g["generator"] if "generator" in state_dict_g else state_dict_g
    generator.load_state_dict(clean_state_dict(g_state), strict=True)
    mpd.load_state_dict(clean_state_dict(state_dict_do["mpd"]), strict=True)
    mrd.load_state_dict(clean_state_dict(state_dict_do["mrd"]), strict=True)

    steps = int(state_dict_do["steps"]) + 1
    last_epoch = int(state_dict_do["epoch"])
    print(f"Resumed from step {steps} / epoch {last_epoch}.")
    return steps, last_epoch, state_dict_do

def train(rank, args, h):
    distributed = args.num_gpus > 1
    device_flag = rank == 0

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    if distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            timeout=datetime.timedelta(seconds=3400),
            world_size=args.num_gpus,
            rank=rank,
        )

    set_random_seed(h.seed + rank, deterministic=getattr(h, "deterministic", True))

    if h.batch_size % args.num_gpus != 0 and device_flag:
        print(f"[Warning] batch_size={h.batch_size} is not divisible by num_gpus={args.num_gpus}.")
    batch_size_per_gpu = max(1, h.batch_size // args.num_gpus)

    if device_flag:
        print(f"Using {args.num_gpus} GPU(s); batch size per GPU: {batch_size_per_gpu}.")
        print(f"Checkpoint directory: {h.checkpoint_path}")

    generator = build_generator(h).to(device)
    if device_flag and get_model_complexity_info is not None:
        try:
            get_model_complexity_info(generator, (h.code_dim, h.sampling_rate // h.hop_size + 1))
        except Exception as exc:
            print(f"[Warning] ptflops failed: {exc}")

    mpd = MultiPeriodDiscriminator(h.mpd_reshapes).to(device)
    mrd = MultiResolutionDiscriminator(resolutions=h.mrd_resolutions).to(device)

    mel_loss = MultiResolutionMelLoss(
        resolutions=h.mel_resolutions,
        sampling_rate=h.sampling_rate,
    ).to(device)

    phase_loss = Weighted_OmniPhaseLoss(use_mag_weighted=getattr(h, "use_weighted_phase", True)).to(device)
    # found marginal improvement for omni-ri if omni-phase is adopted
    ri_loss = OmniRILoss().to(device) if getattr(h, "use_omni_ri_loss", True) else RILoss().to(device)

    os.makedirs(h.checkpoint_path, exist_ok=True)
    steps, last_epoch, state_dict_do = load_resume_if_available(
        generator, mpd, mrd, h, device, resume=not args.no_resume
    )

    if distributed:
        generator = DDP(generator, device_ids=[rank], find_unused_parameters=False)
        mpd = DDP(mpd, device_ids=[rank], find_unused_parameters=False)
        mrd = DDP(mrd, device_ids=[rank], find_unused_parameters=False)

    optim_g = torch.optim.AdamW(
        generator.parameters(),
        h.learning_rate,
        betas=(h.adam_b1, h.adam_b2),
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(mrd.parameters(), mpd.parameters()),
        h.learning_rate,
        betas=(h.adam_b1, h.adam_b2),
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=getattr(h, "lr_decay", 0.999), last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=getattr(h, "lr_decay", 0.999), last_epoch=last_epoch)
    if state_dict_do is not None:
        scheduler_g.load_state_dict(state_dict_do["scheduler_g"])
        scheduler_d.load_state_dict(state_dict_do["scheduler_d"])

    training_filelist, validation_filelist = get_dataset_filelist(
        h.training_dataset_type,
        h.validation_dataset_type,
        h.input_training_wav_list,
        h.input_validation_wav_list,
        getattr(h, "raw_wavfile_path", ""),
    )
    if len(training_filelist) == 0:
        raise RuntimeError("No training files found. Please check input_training_wav_list.")

    trainset = Dataset(
        "train",
        training_filelist,
        h.segment_size,
        h.sampling_rate,
        shuffle=(not distributed),
        device=device,
        n_cache_reuse=0,
        normalize=getattr(h, "normalize_audio", True),
    )
    train_sampler = DistributedSampler(trainset, shuffle=True) if distributed else None
    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        batch_size=batch_size_per_gpu,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    validation_loader = None
    if device_flag and len(validation_filelist) > 0:
        validset = Dataset(
            "val",
            validation_filelist,
            h.segment_size,
            h.sampling_rate,
            split=False,
            shuffle=False,
            device=device,
            max_segment_size=getattr(h, "max_segment_size", None),
            file_num=getattr(h, "validation_file_num", -1),
            n_cache_reuse=0,
            normalize=getattr(h, "normalize_audio", True),
        )
        validation_loader = DataLoader(
            validset,
            num_workers=getattr(h, "validation_num_workers", 1),
            shuffle=False,
            batch_size=1,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    codec_feature_extractor = build_codec_feature_extractor(h, device)
    sw = SummaryWriter(os.path.join(h.checkpoint_path, "logs")) if device_flag else None

    generator.train()
    mpd.train()
    mrd.train()
    global_pesq_score = -float("inf")

    bandwidths = list(getattr(h, "bandwidths", [1.5, 3.0, 6.0, 12.0]))

    for epoch in range(max(0, last_epoch), h.training_epochs):
        if device_flag:
            epoch_start = time.time()
            print(f"Epoch: {epoch + 1}")
        if distributed:
            train_sampler.set_epoch(epoch)

        for _, batch in enumerate(tqdm(train_loader, disable=not device_flag)):
            start_b = time.time()
            y = batch.to(device, non_blocking=True)

            logamp, pha, rea, imag = amp_pha_specturm(y, h.n_fft, h.hop_size, h.win_size)

            bandwidth_id = int(torch.randint(low=0, high=len(bandwidths), size=(1,)).item())
            cond = codec_feature_extractor(y, bandwidth_id=bandwidth_id)

            min_frame = min(cond.shape[-1], logamp.shape[-1])
            cond = cond[..., :min_frame]
            logamp = logamp[..., :min_frame]
            pha = pha[..., :min_frame]
            rea = rea[..., :min_frame]
            imag = imag[..., :min_frame]

            logamp_g, pha_g, ri_g, y_g = generator(cond)
            rea_g, imag_g = ri_g[:, 0], ri_g[:, 1]

            frame_min = min(logamp.shape[-1], logamp_g.shape[-1])
            logamp, pha, rea, imag = logamp[..., :frame_min], pha[..., :frame_min], rea[..., :frame_min], imag[..., :frame_min]
            logamp_g, pha_g, rea_g, imag_g = logamp_g[..., :frame_min], pha_g[..., :frame_min], rea_g[..., :frame_min], imag_g[..., :frame_min]

            if y_g.ndim == 3:
                y_g = y_g.squeeze(1)
            y_min = min(y_g.shape[-1], y.shape[-1])
            y_g, y = y_g[..., :y_min], y[..., :y_min]

            update_discriminator = (steps % 2 == 0)

            if update_discriminator:
                optim_d.zero_grad(set_to_none=True)

                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g.detach())
                loss_disc_f, losses_disc_f_r, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

                y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y, y_g.detach())
                loss_disc_s, losses_disc_s_r, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                loss_disc_f = loss_disc_f / max(len(losses_disc_f_r), 1)
                loss_disc_s = loss_disc_s / max(len(losses_disc_s_r), 1)

                L_D = (loss_disc_s * h.mrd_weight + loss_disc_f) * h.loss_configs["DiscriminatorLoss"]

                if is_invalid_loss(L_D):
                    if device_flag:
                        print("NaN/Inf detected in discriminator loss. Skipping this batch.")
                    optim_d.zero_grad(set_to_none=True)
                else:
                    L_D.backward()
                    optim_d.step()

            else:
                optim_g.zero_grad(set_to_none=True)

                L_A = amplitude_loss(logamp, logamp_g)
                L_P = phase_loss(pha, pha_g, torch.exp(logamp))

                _, _, rea_g_final, imag_g_final = amp_pha_specturm(y_g, h.n_fft, h.hop_size, h.win_size)
                L_C = STFT_consistency_loss(rea_g, rea_g_final, imag_g, imag_g_final)
                L_RI = ri_loss(rea, imag, rea_g, imag_g)

                _, y_df_g, fmap_f_r, fmap_f_g = mpd(y, y_g)
                _, y_ds_g, fmap_s_r, fmap_s_g = mrd(y, y_g)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)

                loss_gen_f, losses_gen_f = generator_loss(y_df_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_g)
                loss_gen_f = loss_gen_f / max(len(losses_gen_f), 1)
                loss_gen_s = loss_gen_s / max(len(losses_gen_s), 1)

                L_GAN_G = loss_gen_s * h.mrd_weight + loss_gen_f
                L_FM = loss_fm_s * h.mrd_weight + loss_fm_f
                L_Mel = mel_loss(y=y, y_hat=y_g)

                L_G = (
                    h.loss_configs["AmplitudeLoss"] * L_A
                    + h.loss_configs["PhaseLoss"] * L_P
                    + h.loss_configs["STFTConsistencyLoss"] * L_C
                    + h.loss_configs["RILoss"] * L_RI
                    + h.loss_configs["GeneratorLoss"] * L_GAN_G
                    + h.loss_configs["FeatureMatchingLoss"] * L_FM
                    + h.loss_configs["MelSpecReconstructLoss"] * L_Mel
                )

                if is_invalid_loss(L_G):
                    if device_flag:
                        print(
                            f"NaN/Inf in generator loss. "
                            f"L_A={L_A.item():.4f}, L_P={L_P.item():.4f}, L_C={L_C.item():.4f}, "
                            f"L_RI={L_RI.item():.4f}, L_GAN={L_GAN_G.item():.4f}, "
                            f"L_FM={L_FM.item():.4f}, L_Mel={L_Mel.item():.4f}"
                        )
                    optim_g.zero_grad(set_to_none=True)
                else:
                    L_G.backward()
                    if getattr(h, "g_gradient", -1) > 0:
                        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=h.g_gradient)
                    optim_g.step()

                    if device_flag and steps % h.stdout_interval == 0 and steps != 0:
                        print(
                            f"Steps: {steps}, "
                            f"Loss_G: {L_G.item():.4f}, "
                            f"Amp: {L_A.item():.4f}, Phase: {L_P.item():.4f}, "
                            f"Cons: {L_C.item():.4f}, RI: {L_RI.item():.4f}, "
                            f"Mel: {L_Mel.item():.4f}, "
                            f"s/b: {time.time() - start_b:.3f}"
                        )

                    if device_flag and steps % h.summary_interval == 0 and steps != 0:
                        sw.add_scalar("Training/Generator_Total_Loss", L_G.item(), steps)
                        sw.add_scalar("Training/Mel_Spectrogram_Loss", L_Mel.item(), steps)

            if device_flag and steps % h.checkpoint_interval == 0 and steps != 0:
                g_path = f"{h.checkpoint_path}/g_{steps:08d}"
                save_checkpoint(g_path, {"generator": get_generator_state(generator, distributed)})
                remove_older_checkpoint(g_path, pre="g", max_to_keep=h.max_to_keep)

                do_path = f"{h.checkpoint_path}/do_{steps:08d}"
                save_checkpoint(
                    do_path,
                    {
                        "mpd": (mpd.module if distributed else mpd).state_dict(),
                        "mrd": (mrd.module if distributed else mrd).state_dict(),
                        "optim_g": optim_g.state_dict(),
                        "optim_d": optim_d.state_dict(),
                        "scheduler_d": scheduler_d.state_dict(),
                        "scheduler_g": scheduler_g.state_dict(),
                        "steps": steps,
                        "epoch": epoch,
                    },
                )
                remove_older_checkpoint(do_path, pre="d", max_to_keep=h.max_to_keep)

            if device_flag and validation_loader is not None and steps % h.validation_interval == 0 and steps != 0:
                val_pesq_score = validate(
                    generator,
                    codec_feature_extractor,
                    validation_loader,
                    h,
                    device,
                    sw,
                    steps,
                    mel_loss,
                    phase_loss,
                    ri_loss,
                    bandwidth_id=getattr(h, "validation_bandwidth_id", 0),
                    distributed=distributed,
                )
                if getattr(h, "save_best", True) and val_pesq_score > global_pesq_score:
                    best_path = f"{h.checkpoint_path}/best_g"
                    save_checkpoint(best_path, {"generator": get_generator_state(generator, distributed)})
                    global_pesq_score = val_pesq_score

            steps += 1
            if steps >= h.training_steps:
                if device_flag:
                    print(f"Reached training_steps={h.training_steps}.")
                if sw is not None:
                    sw.close()
                if distributed:
                    dist.destroy_process_group()
                return

        scheduler_g.step()
        scheduler_d.step()
        if device_flag:
            print(f"Time taken for epoch {epoch + 1}: {int(time.time() - epoch_start)} sec\n")

    if sw is not None:
        sw.close()
    if distributed:
        dist.destroy_process_group()


@torch.no_grad()
def validate(
    generator,
    codec_feature_extractor,
    validation_loader,
    h,
    device,
    sw,
    steps,
    mel_loss,
    phase_loss,
    ri_loss,
    bandwidth_id=0,  # fix in the validation 
    distributed=False,
):
    generator.eval()

    val_A_err_tot = 0.0
    val_P_err_tot = 0.0
    val_C_err_tot = 0.0
    val_RI_err_tot = 0.0
    val_Mel_err_tot = 0.0
    pesq_tot = 0.0
    pesq_cnt = 0
    num_batches = 0

    model = generator.module if distributed and hasattr(generator, "module") else generator

    for j, batch in enumerate(tqdm(validation_loader, desc="Validation")):
        y = batch.to(device, non_blocking=True)
        logamp, pha, rea, imag = amp_pha_specturm(y, h.n_fft, h.hop_size, h.win_size)

        cond = codec_feature_extractor(y, bandwidth_id=bandwidth_id)

        min_frame = min(cond.shape[-1], logamp.shape[-1])
        cond = cond[..., :min_frame]
        logamp, pha, rea, imag = logamp[..., :min_frame], pha[..., :min_frame], rea[..., :min_frame], imag[..., :min_frame]

        logamp_g, pha_g, ri_g, y_g = model(cond)
        rea_g, imag_g = ri_g[:, 0], ri_g[:, 1]

        frame_min = min(pha.shape[-1], pha_g.shape[-1])
        logamp_g, logamp = logamp_g[..., :frame_min], logamp[..., :frame_min]
        pha_g, pha = pha_g[..., :frame_min], pha[..., :frame_min]
        rea_g, rea = rea_g[..., :frame_min], rea[..., :frame_min]
        imag_g, imag = imag_g[..., :frame_min], imag[..., :frame_min]

        if y_g.ndim == 3:
            y_g = y_g.squeeze(1)
        y_min = min(y_g.shape[-1], y.shape[-1])
        y_g, y = y_g[..., :y_min], y[..., :y_min]

        _, _, rea_g_final, imag_g_final = amp_pha_specturm(y_g, h.n_fft, h.hop_size, h.win_size)

        val_A_err_tot += amplitude_loss(logamp, logamp_g).item()
        val_P_err_tot += phase_loss(pha, pha_g, torch.exp(logamp)).item()
        val_C_err_tot += STFT_consistency_loss(rea_g, rea_g_final, imag_g, imag_g_final).item()
        val_RI_err_tot += ri_loss(rea, imag, rea_g, imag_g).item()
        val_Mel_err_tot += mel_loss(y=y, y_hat=y_g).item()
        num_batches += 1

        if pesq is not None:
            try:
                y_g_np = y_g.detach().cpu().squeeze().numpy()
                y_np = y.detach().cpu().squeeze().numpy()
                if h.sampling_rate != 16000:
                    y_g_np = lib.resample(y_g_np, orig_sr=h.sampling_rate, target_sr=16000)
                    y_np = lib.resample(y_np, orig_sr=h.sampling_rate, target_sr=16000)
                pesq_tot += pesq(16000, y_np, y_g_np, mode="wb")
                pesq_cnt += 1
            except Exception:
                pass

        if j < getattr(h, "visualize_num", 10):
            sw.add_audio(f"gt/y_{j}", y[0], steps, h.sampling_rate)
            y_spec = spectrogram(y, h.n_fft, 100, h.sampling_rate, h.hop_size, h.win_size, 0, h.sampling_rate / 2)
            sw.add_figure(f"gt/y_spec_{j}", plot_spectrogram(y_spec.squeeze(0).cpu()), steps)

            sw.add_audio(f"generated/y_g_{j}", y_g[0], steps, h.sampling_rate)
            y_g_spec = spectrogram(y_g, h.n_fft, 100, h.sampling_rate, h.hop_size, h.win_size, 0, h.sampling_rate / 2)
            sw.add_figure(f"generated/y_g_spec_{j}", plot_spectrogram(y_g_spec.squeeze(0).cpu().numpy()), steps)

    denom = max(num_batches, 1)
    val_A_err = val_A_err_tot / denom
    val_P_err = val_P_err_tot / denom
    val_C_err = val_C_err_tot / denom
    val_RI_err = val_RI_err_tot / denom
    val_Mel_err = val_Mel_err_tot / denom
    val_pesq_score = pesq_tot / pesq_cnt if pesq_cnt > 0 else -float("inf")

    sw.add_scalar("Validation/Amplitude_Loss", val_A_err, steps)
    sw.add_scalar("Validation/Phase_Loss", val_P_err, steps)
    sw.add_scalar("Validation/STFT_Consistency_Loss", val_C_err, steps)
    sw.add_scalar("Validation/Real-Imaginary_Part_Loss", val_RI_err, steps)
    sw.add_scalar("Validation/Mel_Spectrogram_Loss", val_Mel_err, steps)
    if pesq_cnt > 0:
        sw.add_scalar("Validation/PESQ_score", val_pesq_score, steps)

    print(
        f"Validation at step {steps}: "
        f"Amp={val_A_err:.4f}, Phase={val_P_err:.4f}, Cons={val_C_err:.4f}, "
        f"RI={val_RI_err:.4f}, Mel={val_Mel_err:.4f}, PESQ={val_pesq_score:.4f}"
    )

    generator.train()
    return val_pesq_score


def parse_args():
    parser = argparse.ArgumentParser("PAdec training")
    parser.add_argument("--cfg_filename", type=str, default="./cfgs/padec_24k_encodec.json", help="Path to config json.")
    parser.add_argument("--gpu", type=str, default="2", help="GPU id(s), e.g., '0' or '0,1'. Must be parsed before torch import.")
    parser.add_argument("--no_resume", action="store_true", help="Start from scratch even if checkpoints exist.")
    parser.add_argument("--dist_backend", type=str, default="nccl", help="DDP backend.")
    parser.add_argument("--dist_url", type=str, default="tcp://127.0.0.1:54321", help="DDP init URL for single-node training.")
    return parser.parse_args()


def main():
    print("Initializing Training Process...")
    args = parse_args()

    h = load_config(args.cfg_filename)
    if args.dist_backend:
        h.dist_backend = args.dist_backend
    if args.dist_url:
        h.dist_url = args.dist_url

    config_filename = os.path.basename(args.cfg_filename)
    build_env(args.cfg_filename, config_filename, h.checkpoint_path)

    if torch.cuda.is_available():
        args.num_gpus = torch.cuda.device_count()
    else:
        args.num_gpus = 1
        print("[Warning] CUDA is not available. Training on CPU will be very slow.")

    print(f"Visible GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Detected num_gpus: {args.num_gpus}")

    if args.num_gpus > 1:
        mp.spawn(train, nprocs=args.num_gpus, args=(args, h))
    else:
        train(0, args, h)


if __name__ == "__main__":
    main()
