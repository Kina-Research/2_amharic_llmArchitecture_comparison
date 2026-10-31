"""Minimal, explicit training loop shared by both tasks.

A hand-written loop (rather than HF Trainer) keeps three things transparent:
example-weighted losses (loss-level mitigation), exact energy/VRAM boundaries,
and the absence of any checkpoint selection on the evaluation split
(we always report the model after the final epoch).
"""
from __future__ import annotations

import math
import time
from typing import Callable

import torch
from torch.utils.data import DataLoader


def train_loop(model, loader: DataLoader, loss_fn: Callable, *, lr: float, epochs: int,
               warmup_ratio: float, weight_decay: float, accum: int, fp16: bool,
               max_grad_norm: float = 1.0, log_every: int = 20) -> list[dict]:
    from transformers import get_linear_schedule_with_warmup

    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    total = math.ceil(len(loader) / accum) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(warmup_ratio * total), total)
    use_amp = fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    log, step, t0 = [], 0, time.perf_counter()
    model.train()
    for epoch in range(epochs):
        running, n = 0.0, 0
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch} batch {i}")
            scaler.scale(loss / accum).backward()
            running, n = running + loss.item(), n + 1
            if (i + 1) % accum == 0 or i + 1 == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % log_every == 0:
                    log.append({"epoch": epoch, "step": step, "loss": running / n,
                                "lr": sched.get_last_lr()[0], "elapsed_s": time.perf_counter() - t0})
        log.append({"epoch": epoch, "step": step, "loss": running / max(n, 1),
                    "lr": sched.get_last_lr()[0], "elapsed_s": time.perf_counter() - t0, "epoch_end": True})
        print(f"  epoch {epoch + 1}/{epochs}  mean loss {running / max(n, 1):.4f}", flush=True)
    model.eval()
    return log

