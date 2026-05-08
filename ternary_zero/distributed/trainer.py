from __future__ import annotations

import os
from typing import Optional

from .._backend import has_torch, is_cuda_available

if has_torch():
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed(
    backend: Optional[str] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    init_method: str = "env://",
):
    if not has_torch():
        raise RuntimeError("PyTorch is required for distributed training")

    if backend is None:
        backend = "nccl" if is_cuda_available() else "gloo"

    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )

    if is_cuda_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())

    return rank, world_size


def cleanup_distributed():
    if has_torch() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor, world_size: int, op: str = "mean"):
    if not has_torch() or not dist.is_initialized():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt.data if hasattr(rt, 'data') else rt, op=dist.ReduceOp.SUM)
    if op == "mean":
        rt = rt / world_size
    return rt


class DistributedTrainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn=None,
        device: Optional[str] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = False,
    ):
        if not has_torch():
            raise RuntimeError("PyTorch is required for DistributedTrainer")

        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if device is None:
            if is_cuda_available():
                self.device = torch.device(f"cuda:{self.local_rank}")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = model
        if hasattr(model, 'data'):
            pass
        elif isinstance(model, torch.nn.Module):
            self.model = model.to(self.device)
            if dist.is_initialized():
                self.model = DDP(self.model, device_ids=[self.local_rank] if is_cuda_available() else None)

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and is_cuda_available()

        self.scaler = None
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self._step_count = 0

    def train_step(self, batch, targets=None):
        self.model.train()

        if isinstance(batch, (list, tuple)) and targets is None:
            inputs, targets = batch[0], batch[1] if len(batch) > 1 else None
        else:
            inputs = batch

        if is_cuda_available() and isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
            if targets is not None and isinstance(targets, torch.Tensor):
                targets = targets.to(self.device)

        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                if self.loss_fn is not None and targets is not None:
                    loss = self.loss_fn(outputs, targets)
                else:
                    loss = outputs
            self.scaler.scale(loss / self.gradient_accumulation_steps).backward()
        else:
            outputs = self.model(inputs)
            if self.loss_fn is not None and targets is not None:
                loss = self.loss_fn(outputs, targets)
            else:
                loss = outputs
            (loss / self.gradient_accumulation_steps).backward()

        self._step_count += 1

        if self._step_count % self.gradient_accumulation_steps == 0:
            if self.max_grad_norm > 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

        return loss.detach()

    def is_main_process(self) -> bool:
        return self.rank == 0

    def barrier(self):
        if dist.is_initialized():
            dist.barrier()

    def save_checkpoint(self, path: str, **extra_state):
        if not self.is_main_process():
            return
        state = {
            "model": self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._step_count,
        }
        state.update(extra_state)
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        if is_cuda_available():
            checkpoint = torch.load(path, map_location=self.device)
        else:
            checkpoint = torch.load(path, map_location="cpu")
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._step_count = checkpoint.get("step", 0)
        return checkpoint
