from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .checkpoints import load_checkpoint, save_checkpoint
from .ema import ModelEMA
from .evaluate import evaluate_and_save, predict
from .losses import MixupCriterion, mixup_batch
from .metrics import compute_metrics
from .models import build_model


def build_scheduler(optimizer, config: dict[str, Any]):
    epochs = int(config["training"]["epochs"])
    warmup = int(config["training"]["warmup_epochs"])
    min_lr = float(config["training"]["min_learning_rate"])
    max_lr = float(config["training"]["learning_rate"])
    min_factor = min_lr / max_lr

    def lr_lambda(epoch: int):
        if epoch < warmup:
            return min_factor + (1.0 - min_factor) * float(epoch + 1) / float(max(1, warmup))
        progress = float(epoch - warmup) / float(max(1, epochs - warmup))
        return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def append_history(path: str | Path, row: dict[str, Any]) -> None:
    out = Path(path)
    exists = out.exists()
    with out.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, ema, device, config):
    model.train()
    total_loss = 0.0
    total_items = 0
    mixup_alpha = float(config["training"]["mixup_alpha"])
    use_amp = bool(config["training"]["amp"]) and device.type == "cuda"
    grad_clip = float(config["training"]["gradient_clip_norm"])

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        images, y_a, y_b, lam = mixup_batch(images, labels, mixup_alpha)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, y_a, y_b, lam)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if ema:
            ema.update(model)

        batch_size = images.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size

    scheduler.step()
    return total_loss / max(1, total_items)


@torch.no_grad()
def validate_loss(model, loader, criterion, device, config):
    model.eval()
    total_loss = 0.0
    total_items = 0
    use_amp = bool(config["training"]["amp"]) and device.type == "cuda"
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += float(loss.detach().cpu()) * images.size(0)
        total_items += images.size(0)
    return total_loss / max(1, total_items)


def train_hybrid(config, train_dataset, val_dataset, test_dataset, exp_dir: str | Path, logger):
    exp_dir = Path(exp_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = [k for k, _ in sorted(config["classes"].items(), key=lambda item: item[1])]

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["dataset"]["num_workers"]),
        pin_memory=bool(config["dataset"]["pin_memory"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["dataset"]["num_workers"]),
        pin_memory=bool(config["dataset"]["pin_memory"]),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["dataset"]["num_workers"]),
        pin_memory=bool(config["dataset"]["pin_memory"]),
    )

    model = build_model(config).to(device)
    optimizer = AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["training"]["amp"]) and device.type == "cuda")
    criterion = MixupCriterion(float(config["training"]["label_smoothing"]))
    ema = ModelEMA(model, float(config["training"]["ema_decay"]))

    start_epoch = 0
    best_metric = float("inf")
    latest = exp_dir / "latest.pt"
    if bool(config["training"].get("resume", True)) and latest.exists():
        ckpt = load_checkpoint(latest, model, optimizer, scheduler, scaler, ema, device)
        start_epoch = int(ckpt["epoch"]) + 1
        best_metric = float(ckpt["best_metric"])
        logger.info("Resumed from epoch %s", start_epoch)

    patience = int(config["training"]["early_stopping_patience"])
    stale_epochs = 0
    start_time = time.time()
    logger.info("Starting training on device=%s", device)

    for epoch in range(start_epoch, int(config["training"]["epochs"])):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, ema, device, config)
        eval_model = ema.ema if ema else model
        val_loss = validate_loss(eval_model, val_loader, criterion, device, config)
        y_true, y_pred, y_prob, _ = predict(eval_model, val_loader, device)
        val_metrics = compute_metrics(y_true, y_pred, y_prob, class_names)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        append_history(exp_dir / "history.csv", row)
        logger.info("epoch=%s train_loss=%.6f val_loss=%.6f val_bal_acc=%.6f", epoch, train_loss, val_loss, val_metrics["balanced_accuracy"])

        improved = val_loss < best_metric
        if improved:
            best_metric = val_loss
            stale_epochs = 0
            save_checkpoint(exp_dir / "best.pt", model, optimizer, scheduler, scaler, ema, epoch, best_metric, config)
        else:
            stale_epochs += 1

        save_checkpoint(latest, model, optimizer, scheduler, scaler, ema, epoch, best_metric, config)
        if stale_epochs >= patience:
            logger.info("Early stopping at epoch %s", epoch)
            break

    logger.info("Training duration seconds=%.2f", time.time() - start_time)
    if (exp_dir / "best.pt").exists():
        load_checkpoint(exp_dir / "best.pt", model, ema=ema, device=device)
    eval_model = ema.ema if ema else model
    return evaluate_and_save(eval_model, test_loader, device, class_names, exp_dir)
