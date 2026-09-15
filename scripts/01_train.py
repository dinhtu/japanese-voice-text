"""Fine-tune wav2vec2 (base or large) with dual CTC for Japanese kana + phoneme ASR.

Usage:
    uv run python scripts/01_train.py --data-split small --epochs 20
    uv run python scripts/01_train.py --pretrained reazon-research/japanese-wav2vec2-large \
        --data-split small --epochs 15 --batch-size 16 --grad-accum 2 \
        --lr 1e-4 --warmup-steps 1000 --grad-clip 1.0 --bf16
"""

# ruff: noqa: E402

import argparse
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Sampler
from transformers import Wav2Vec2FeatureExtractor

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.asr.dataset import (
    ASRDataset,
    PreprocessedASRDataset,
    collate_fn,
    load_prepared_reazonspeech,
)
from src.asr.kana_converter import JapaneseKanaConverter
from src.asr.kana_vocab import KanaVocab
from src.asr.losses import DualCTCLoss
from src.asr.model import create_model
from src.asr.phoneme_converter import JapanesePhonemeConverter
from src.asr.phoneme_vocab import PhonemeVocab

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train wav2vec2 + dual CTC for Japanese ASR")
    p.add_argument("--pretrained", default="reazon-research/japanese-wav2vec2-base")
    p.add_argument("--data-split", default="small", choices=["tiny", "small", "medium"])
    p.add_argument("--dataset-dir", type=Path, default=Path("data/datasets/reazonspeech"))
    p.add_argument("--output-dir", default="models/checkpoints", type=Path)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--cr-weight", type=float, default=0.1, help="CR-CTC regularization weight")
    p.add_argument("--inter-weight", type=float, default=0.3, help="InterCTC loss weight")
    p.add_argument("--inter-ctc-layer", type=int, default=None,
                   help="Encoder layer for InterCTC (default: auto = num_layers // 2)")
    p.add_argument("--mask-time-prob", type=float, default=0.05, help="SpecAugment time masking")
    p.add_argument("--max-duration", type=float, default=15.0, help="Max audio duration in seconds")
    p.add_argument("--speed-perturb-prob", type=float, default=0.2)
    p.add_argument("--noise-prob", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (default: auto)")
    p.add_argument("--eval-steps", type=int, default=5000)
    p.add_argument("--save-steps", type=int, default=5000)
    p.add_argument(
        "--eval-at-epoch-end",
        action="store_true",
        default=True,
        help="Run validation at each epoch end.",
    )
    p.add_argument("--no-eval-at-epoch-end", dest="eval_at_epoch_end", action="store_false")
    p.add_argument(
        "--save-at-epoch-end",
        action="store_true",
        default=True,
        help="Save a checkpoint at each epoch end.",
    )
    p.add_argument("--no-save-at-epoch-end", dest="save_at_epoch_end", action="store_false")
    p.add_argument(
        "--bucket-batching",
        action="store_true",
        default=True,
        help="Use length-based bucketed batching (preprocessed datasets only).",
    )
    p.add_argument("--no-bucket-batching", dest="bucket_batching", action="store_false")
    p.add_argument("--bucket-size", type=int, default=2048, help="Bucket size for length sorting")
    p.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Checkpoint path for resuming training state.",
    )
    p.add_argument("--fp16", action="store_true", default=False,
                   help="Enable FP16 (may NaN on wav2vec2 CNN; needs bf16-capable GPU)")
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--bf16", action="store_true", default=False,
                   help="Enable BF16 (recommended for A100/H100; no overflow like FP16)")
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--wandb-project", default="asr-test")
    return p.parse_args()


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int
):
    """Linear warmup then cosine decay to zero."""
    total_steps = max(total_steps, 1)
    warmup_steps = min(max(warmup_steps, 0), total_steps - 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        progress = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def resolve_num_workers(requested: int | None) -> int:
    if requested is not None:
        return max(requested, 0)
    n_cpu = os.cpu_count() or 4
    if n_cpu <= 4:
        return max(n_cpu - 1, 1)
    return min(8, max(2, n_cpu - 2))


def extract_waveform_lengths(hf_dataset) -> list[int]:
    """Extract waveform lengths from preprocessed dataset for bucket batching."""
    if "waveform_len" in hf_dataset.column_names:
        return [int(x) for x in hf_dataset["waveform_len"]]
    if "waveform" in hf_dataset.column_names:
        return [int(len(w)) for w in hf_dataset["waveform"]]
    raise ValueError(
        "Bucket batching requires 'waveform_len' or 'waveform' column in preprocessed dataset."
    )


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Batch sampler that reduces padding by grouping similar lengths."""

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        bucket_size: int = 2048,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_size = max(bucket_size, batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)

        batches: list[list[int]] = []
        for i in range(0, len(indices), self.bucket_size):
            bucket = indices[i:i + self.bucket_size]
            bucket.sort(key=lambda x: self.lengths[x])
            for j in range(0, len(bucket), self.batch_size):
                batch = bucket[j:j + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        self.epoch += 1
        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def save_checkpoint(
    path: Path,
    model,
    pretrained: str,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    use_fp16: bool,
    global_step: int,
    epoch: int,
    best_val_loss: float,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "pretrained": pretrained,
            "inter_ctc_layer": model.inter_ctc_layer,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_fp16 else None,
            "global_step": global_step,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    use_fp16: bool,
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # Backward compatibility for model-only checkpoints.
        model.load_state_dict(ckpt)
        ckpt = {}

    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if use_fp16 and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    return ckpt


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_fp16 = args.fp16 and device.type == "cuda"
    use_bf16 = args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if use_bf16:
        use_fp16 = False  # BF16 takes priority
        log.info("Using BF16 (no GradScaler needed)")

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    log.info(f"Device: {device}")
    log.info(f"Loading model: {args.pretrained}")

    # Model
    model = create_model(
        pretrained=args.pretrained,
        mask_time_prob=args.mask_time_prob,
        inter_ctc_layer=args.inter_ctc_layer,
    )
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {n_params/1e6:.1f}M params ({n_trainable/1e6:.1f}M trainable)")
    log.info(f"InterCTC layer: {model.inter_ctc_layer}")

    # Feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.pretrained)

    # Dataset — prefer preprocessed data if available
    proc_dir = args.dataset_dir / f"{args.data_split}_proc"
    use_preprocessed = proc_dir.exists()

    if use_preprocessed:
        from datasets import load_from_disk

        log.info(f"Using PREPROCESSED data from {proc_dir}")
        hf_dataset = load_from_disk(str(proc_dir))
    else:
        split_dir = args.dataset_dir / args.data_split
        log.info(f"Loading raw ReazonSpeech from {split_dir} ...")
        hf_dataset = load_prepared_reazonspeech(args.data_split, args.dataset_dir)

    kana_vocab = KanaVocab()
    phoneme_vocab = PhonemeVocab()

    n = len(hf_dataset)
    n_val = max(int(n * 0.05), 1)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    train_hf = hf_dataset.select(train_indices)
    val_hf = hf_dataset.select(val_indices)

    train_batch_sampler = None

    if use_preprocessed:
        train_set = PreprocessedASRDataset(
            train_hf, feature_extractor,
            speed_perturb_prob=args.speed_perturb_prob,
            noise_prob=args.noise_prob,
        )
        val_set = PreprocessedASRDataset(
            val_hf, feature_extractor,
            speed_perturb_prob=0.0,
            noise_prob=0.0,
        )
        if args.bucket_batching:
            log.info("Building length buckets from preprocessed dataset...")
            train_lengths = extract_waveform_lengths(train_hf)
            train_batch_sampler = LengthBucketBatchSampler(
                lengths=train_lengths,
                batch_size=args.batch_size,
                bucket_size=args.bucket_size,
                shuffle=True,
                drop_last=False,
            )
    else:
        kana_converter = JapaneseKanaConverter()
        phoneme_converter = JapanesePhonemeConverter()
        train_set = ASRDataset(
            train_hf, feature_extractor,
            kana_converter=kana_converter,
            phoneme_converter=phoneme_converter,
            kana_vocab=kana_vocab,
            phoneme_vocab=phoneme_vocab,
            max_duration_s=args.max_duration,
            speed_perturb_prob=args.speed_perturb_prob,
            noise_prob=args.noise_prob,
        )
        val_set = ASRDataset(
            val_hf, feature_extractor,
            kana_converter=kana_converter,
            phoneme_converter=phoneme_converter,
            kana_vocab=kana_vocab,
            phoneme_vocab=phoneme_vocab,
            max_duration_s=args.max_duration,
            speed_perturb_prob=0.0,
            noise_prob=0.0,
        )

    num_workers = resolve_num_workers(args.num_workers)
    pin = num_workers > 0
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    if train_batch_sampler is not None:
        train_loader = DataLoader(
            train_set,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_fn,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            **loader_kwargs,
        )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    log.info(f"Train: {len(train_set)}, Val: {len(val_set)}")
    log.info(f"Kana vocab: {kana_vocab.size}, Phoneme vocab: {phoneme_vocab.size}")
    log.info(
        f"DataLoader workers={num_workers}, pin_memory={pin}, "
        f"bucket_batching={train_batch_sampler is not None}"
    )

    # Loss
    criterion = DualCTCLoss(
        blank=0, cr_weight=args.cr_weight, inter_weight=args.inter_weight,
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum, 1))
    total_steps = max(steps_per_epoch * args.epochs, 1)
    scheduler = build_warmup_cosine_scheduler(
        optimizer, total_steps=total_steps, warmup_steps=args.warmup_steps,
    )

    use_amp = use_fp16 or use_bf16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    # GradScaler only for FP16; BF16 doesn't need it (same exponent range as FP32)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 0

    if args.resume_from is not None:
        if not args.resume_from.exists():
            raise FileNotFoundError(f"--resume-from not found: {args.resume_from}")
        ckpt = load_checkpoint(
            path=args.resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            use_fp16=use_fp16,
        )
        global_step = int(ckpt.get("global_step", 0))
        start_epoch = int(ckpt.get("epoch", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        log.info(
            f"Resumed from {args.resume_from}: "
            f"epoch={start_epoch}, step={global_step}, best_val={best_val_loss:.4f}"
        )

    if start_epoch >= args.epochs:
        log.info(
            f"Nothing to train: resume epoch {start_epoch} >= target epochs {args.epochs}. "
            "Increase --epochs to continue."
        )
        return

    optimizer.zero_grad(set_to_none=True)
    n_train_batches = len(train_loader)
    last_eval_step = -1
    last_save_step = -1

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        import time as _time
        _batch_t0 = _time.time()
        for batch_idx, batch in enumerate(train_loader):
            _data_t = _time.time() - _batch_t0
            if batch_idx < 5 or batch_idx % 50 == 0:
                log.info(
                    f"  Epoch {epoch+1} batch {batch_idx}/{n_train_batches} "
                    f"step={global_step} data={_data_t:.2f}s"
                )
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            kana_labels = batch["kana_labels"].to(device)
            phoneme_labels = batch["phoneme_labels"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            kana_target_lengths = batch["kana_target_lengths"].to(device)
            phoneme_target_lengths = batch["phoneme_target_lengths"].to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_values, attention_mask=attention_mask)

            # CTC loss requires float32 — compute outside autocast
            kana_logits = outputs["kana_logits"].float()
            phoneme_logits = outputs["phoneme_logits"].float()

            # CTC expects (T, B, V)
            kana_log_probs = kana_logits.log_softmax(dim=-1).permute(1, 0, 2)
            phoneme_log_probs = phoneme_logits.log_softmax(dim=-1).permute(1, 0, 2)

            # Compute output lengths (wav2vec2 downsamples by ~320x)
            out_lengths = model.get_feat_extract_output_lengths(input_lengths)
            out_lengths = out_lengths.clamp(max=kana_log_probs.shape[0])

            loss_dict = criterion(
                kana_log_probs, kana_labels,
                phoneme_log_probs, phoneme_labels,
                out_lengths, kana_target_lengths, phoneme_target_lengths,
            )
            loss = loss_dict["loss"] / args.grad_accum

            scaler.scale(loss).backward()

            should_step = (
                (batch_idx + 1) % args.grad_accum == 0
                or (batch_idx + 1) == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if args.wandb:
                    import wandb
                    wandb.log({
                        "train/loss": loss_dict["loss"].item(),
                        "train/kana_loss": loss_dict["kana_loss"].item(),
                        "train/phoneme_loss": loss_dict["phoneme_loss"].item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/step": global_step,
                    })

                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    if global_step != last_eval_step:
                        val_loss = evaluate(
                            model, val_loader, criterion, device, use_amp, amp_dtype
                        )
                        last_eval_step = global_step
                        log.info(f"  Step {global_step}: val_loss={val_loss:.4f}")

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            save_path = args.output_dir / "best.pt"
                            save_checkpoint(
                                path=save_path,
                                model=model,
                                pretrained=args.pretrained,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scaler=scaler,
                                use_fp16=use_fp16,
                                global_step=global_step,
                                epoch=epoch,
                                best_val_loss=best_val_loss,
                            )
                            log.info(f"  Saved best model (val_loss={val_loss:.4f})")

                        if args.wandb:
                            import wandb

                            wandb.log({"val/loss": val_loss, "val/step": global_step})

                        model.train()

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    if global_step != last_save_step:
                        save_path = args.output_dir / f"checkpoint-{global_step}.pt"
                        save_checkpoint(
                            path=save_path,
                            model=model,
                            pretrained=args.pretrained,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            use_fp16=use_fp16,
                            global_step=global_step,
                            epoch=epoch,
                            best_val_loss=best_val_loss,
                        )
                        last_save_step = global_step

            epoch_loss += loss_dict["loss"].item()
            n_batches += 1
            if batch_idx < 5:
                _total_t = _time.time() - _batch_t0
                log.info(
                    f"    batch {batch_idx} done: loss={loss_dict['loss'].item():.2f} "
                    f"total={_total_t:.2f}s seq={input_values.shape[1]}"
                )
            _batch_t0 = _time.time()

        avg_loss = epoch_loss / max(n_batches, 1)
        log.info(f"Epoch {epoch + 1}/{args.epochs}: avg_loss={avg_loss:.4f}")

        if args.eval_at_epoch_end:
            val_loss = evaluate(model, val_loader, criterion, device, use_amp, amp_dtype)
            log.info(f"  Epoch {epoch + 1} end: val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = args.output_dir / "best.pt"
                save_checkpoint(
                    path=save_path,
                    model=model,
                    pretrained=args.pretrained,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    use_fp16=use_fp16,
                    global_step=global_step,
                    epoch=epoch + 1,
                    best_val_loss=best_val_loss,
                )
                log.info(f"  Saved best model (val_loss={val_loss:.4f})")
            if args.wandb:
                import wandb

                wandb.log(
                    {
                        "val/loss_epoch_end": val_loss,
                        "train/epoch": epoch + 1,
                        "train/step": global_step,
                    }
                )
            model.train()

        if args.save_at_epoch_end:
            save_path = args.output_dir / f"epoch-{epoch + 1}.pt"
            save_checkpoint(
                path=save_path,
                model=model,
                pretrained=args.pretrained,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                use_fp16=use_fp16,
                global_step=global_step,
                epoch=epoch + 1,
                best_val_loss=best_val_loss,
            )

    # Final save
    save_checkpoint(
        path=args.output_dir / "final.pt",
        model=model,
        pretrained=args.pretrained,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        use_fp16=use_fp16,
        global_step=global_step,
        epoch=args.epochs,
        best_val_loss=best_val_loss,
    )
    log.info("Training complete.")


@torch.no_grad()
def evaluate(model, val_loader, criterion, device, use_amp, amp_dtype):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in val_loader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        kana_labels = batch["kana_labels"].to(device)
        phoneme_labels = batch["phoneme_labels"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        kana_target_lengths = batch["kana_target_lengths"].to(device)
        phoneme_target_lengths = batch["phoneme_target_lengths"].to(device)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            outputs = model(input_values, attention_mask=attention_mask)

        kana_log_probs = outputs["kana_logits"].float().log_softmax(dim=-1).permute(1, 0, 2)
        phoneme_log_probs = outputs["phoneme_logits"].float().log_softmax(dim=-1).permute(1, 0, 2)
        out_lengths = model.get_feat_extract_output_lengths(input_lengths)
        out_lengths = out_lengths.clamp(max=kana_log_probs.shape[0])

        loss_dict = criterion(
            kana_log_probs, kana_labels,
            phoneme_log_probs, phoneme_labels,
            out_lengths, kana_target_lengths, phoneme_target_lengths,
        )

        total_loss += loss_dict["loss"].item()
        n += 1
    return total_loss / max(n, 1)


if __name__ == "__main__":
    main()
