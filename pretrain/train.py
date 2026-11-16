"""Continued multimodal pre-training of Qwen-VL models with PyTorch FSDP2.

Tested with Qwen3-VL-8B-Instruct and Qwen3.8-27B. Launch with torchrun, see run_pretrain.sh.

* Data is tokenized on the fly and packed into rows of at most `--seq_len` tokens (see data.py).
* Samples in a row cannot see each other. Full attention uses flash-attn varlen, the linear
  attention of Qwen3.5 / 3.8 restarts at every sample, and positions restart from 0.
* Every optimizer step trains `--global_batch_size` rows, i.e. a fixed token budget,
  while the number of samples per step varies.
* The loss is a token-level mean over the whole global batch.
* Training length is given in epochs over the weighted mixture. Every checkpoint keeps HF weights;
  only the latest one also keeps the resumable `dcp/` state (weights + optimizer).
"""

import argparse
import copy
import glob
import json
import math
import os
import shutil
import time
from datetime import timedelta
from itertools import islice

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import (StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
                                                     set_model_state_dict, set_optimizer_state_dict)
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy, fully_shard
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from data import IGNORE, PackedDataset, RecordIndex, estimate_tokens, parse_data_specs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data", nargs="+", required=True, help="'path[:weight]', path = file or directory, weight = passes")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seq_len", type=int, default=32768, help="tokens of one packed row on one GPU")
    p.add_argument("--max_len", type=int, default=4096, help="max tokens of one sample")
    p.add_argument("--global_batch_size", type=int, default=16, help="packed rows per optimizer step")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=0, help="if > 0, overrides warmup_ratio")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=0, help="0 = estimate from data (num_epochs passes of the mixture)")
    p.add_argument("--num_epochs", type=float, default=1.0)
    p.add_argument("--max_pixels", type=int, default=2097152,
                   help="max total pixels per image (about 2048 visual tokens); larger images are downscaled")
    p.add_argument("--min_pixels", type=int, default=None)
    p.add_argument("--freeze_vision", action="store_true")
    p.add_argument("--cpu_offload", action="store_true", help="offload FSDP params / grads / optimizer to CPU")
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--no_ckpt_layers", type=int, default=0,
                   help="skip activation recomputation in the last N decoder layers (faster, uses spare memory)")
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--prefetch_factor", type=int, default=2, help="windows prefetched per worker")
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--save_every_epochs", type=float, default=0, help="if > 0, overrides save_steps")
    p.add_argument("--save_total_limit", type=int, default=0, help="HF checkpoints kept, 0 = all")
    p.add_argument("--no_save_optimizer", action="store_true", help="save HF weights only (cannot resume)")
    p.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in output_dir")
    p.add_argument("--logger", choices=["none", "swanlab", "wandb"], default="none")
    p.add_argument("--project", default="HuatuoGPT-Vision-2")
    p.add_argument("--run_name", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def log0(*a):
    if dist.get_rank() == 0:
        print(*a, flush=True)


# ----------------------------------------------------------------------------- model


def build_model(args, device):
    config = AutoConfig.from_pretrained(args.model_path)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.float()  # fp32 master weights; FSDP computes in bf16
    model.config.use_cache = False
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for layer in model.model.language_model.layers[len(model.model.language_model.layers) - args.no_ckpt_layers:]:
            layer.gradient_checkpointing = False
    if args.freeze_vision:
        model.model.visual.requires_grad_(False)

    # Shard the embeddings, every decoder layer and lm_head. The vision tower stays in the root unit
    # so that ranks with and without images issue identical collectives (see dummy_vision_loss).
    kw = dict(mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
              offload_policy=CPUOffloadPolicy() if args.cpu_offload else OffloadPolicy())
    lm = model.model.language_model
    for module in [lm.embed_tokens, *lm.layers, model.lm_head]:
        fully_shard(module, **kw)
    fully_shard(model, **kw)
    model.to_empty(device="cpu" if args.cpu_offload else device)

    # Rank 0 reads the checkpoint, every rank receives only its own shard.
    hf = AutoModelForImageTextToText.from_pretrained(args.model_path, dtype=torch.bfloat16) if dist.get_rank() == 0 else None
    set_model_state_dict(model, hf.state_dict() if hf else {},
                         options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
    for name, buf in list(model.named_buffers()):  # non-persistent buffers, e.g. rotary inv_freq
        t = hf.get_buffer(name).to(device, buf.dtype) if hf else torch.empty(buf.shape, dtype=buf.dtype, device=device)
        dist.broadcast(t, 0)
        mod, _, attr = name.rpartition(".")
        model.get_submodule(mod)._buffers[attr] = t
    del hf
    patch_linear_attention(model)
    return model


PACKED = {}  # sample boundaries of the current row, read by the patched linear attention


def patch_linear_attention(model):
    """Make Qwen3.5 GatedDeltaNet restart its conv and recurrent state at every sample boundary."""
    for m in model.modules():
        if hasattr(m, "chunk_gated_delta_rule") and hasattr(m, "causal_conv1d_fn"):
            assert m.causal_conv1d_fn is not None and m.chunk_gated_delta_rule.__module__.startswith("fla"), \
                "packed training of linear attention needs causal-conv1d and flash-linear-attention"
            conv, chunk = m.causal_conv1d_fn, m.chunk_gated_delta_rule
            m.causal_conv1d_fn = lambda conv=conv, **kw: conv(**{**kw, "seq_idx": PACKED["seq_idx"]})
            m.chunk_gated_delta_rule = lambda *a, chunk=chunk, **kw: chunk(*a, **kw, cu_seqlens=PACKED["cu_seqlens"])


def packed_position_ids(model, ids, cu, images_per_sample, grid, image_pad_id):
    """RoPE positions that restart at every sample: (4, 1, L) = text positions + 3D (t, h, w) positions."""
    text, mrope, k = [], [], 0
    for start, end, n in zip(cu[:-1].tolist(), cu[1:].tolist(), images_per_sample.tolist()):
        seg = ids[None, start:end]
        pos, _ = model.model.get_rope_index(seg, mm_token_type_ids=(seg == image_pad_id).int(),
                                            image_grid_thw=grid[k:k + n] if n else None)
        k += n
        mrope.append(pos)
        text.append(torch.arange(end - start, device=ids.device))
    return torch.cat([torch.cat(text)[None, None], torch.cat(mrope, -1)])


def dummy_vision_loss(model, device):
    """0 * vision(dummy image): keeps vision params in the graph on ranks whose batch has no image."""
    vc = model.config.vision_config
    pixels = torch.zeros(4, vc.in_channels * vc.temporal_patch_size * vc.patch_size**2, device=device)
    out = model.model.visual(pixels, grid_thw=torch.tensor([[1, 2, 2]], device=device))
    tensors = [t for t in out.values() if torch.is_tensor(t)] + [t for v in out.values() if isinstance(v, (list, tuple)) for t in v]
    return sum(t.float().sum() for t in tensors) * 0.0


def _ce_sum(logits, targets):
    return F.cross_entropy(logits.float(), targets, ignore_index=IGNORE, reduction="sum")


def chunked_cross_entropy(logits, targets, chunk=4096):
    """Summed CE. fp32 logits only exist for one chunk at a time, which keeps long windows in memory."""
    chunks = zip(logits.split(chunk), targets.split(chunk))  # split: one gradient buffer in backward
    return sum(checkpoint(_ce_sum, x, t, use_reentrant=False) for x, t in chunks)


def forward_loss(model, batch, image_pad_id, device):
    """Summed next-token CE of one packed row. Logits are only computed where a token is trained."""
    ids, labels, cu = batch["input_ids"], batch["labels"], batch["cu_seqlens"]
    grid = batch.get("image_grid_thw")
    lengths = (cu[1:] - cu[:-1]).long()
    PACKED["cu_seqlens"] = cu.long()
    PACKED["seq_idx"] = torch.repeat_interleave(torch.arange(len(lengths), device=device, dtype=torch.int32), lengths)[None]
    keep = (labels[1:] != IGNORE).nonzero().squeeze(-1)
    # positions restart at every sample, so flash-attn runs varlen attention over the samples
    inputs = dict(input_ids=ids[None], logits_to_keep=keep, use_cache=False,
                  position_ids=packed_position_ids(model, ids, cu, batch["images_per_sample"], grid, image_pad_id))
    if grid is not None:
        inputs.update(pixel_values=batch["pixel_values"], image_grid_thw=grid)
    logits = model(**inputs).logits[0]
    loss = chunked_cross_entropy(logits, labels[keep + 1])
    if grid is None:
        loss = loss + dummy_vision_loss(model, device)
    return loss


# ----------------------------------------------------------------------------- checkpoint


def copy_unloaded_weights(src, out):
    """Copy tensors that transformers does not load (e.g. Qwen3.5 `mtp.*` used by vLLM) unchanged."""
    if not os.path.isdir(src):  # Hub model id: from_pretrained has already cached the files
        from huggingface_hub import snapshot_download
        src = snapshot_download(src, allow_patterns=["*.json", "*.safetensors"])
    src_index, index_path = os.path.join(src, "model.safetensors.index.json"), os.path.join(out, "model.safetensors.index.json")
    if not (os.path.exists(src_index) and os.path.exists(index_path)):
        return  # single-file checkpoints have no index and no extra tensors to copy
    src_map = json.load(open(src_index))["weight_map"]
    index = json.load(open(index_path))
    missing = [k for k in src_map if k not in index["weight_map"]]
    if not missing:
        return
    tensors = {}
    for file in sorted({src_map[k] for k in missing}):
        with safe_open(os.path.join(src, file), "pt") as f:
            tensors.update({k: f.get_tensor(k) for k in missing if src_map[k] == file})
    save_file(tensors, os.path.join(out, "model-extra.safetensors"), metadata={"format": "pt"})
    index["weight_map"].update({k: "model-extra.safetensors" for k in missing})
    json.dump(index, open(index_path, "w"), indent=2)


def save_checkpoint(args, model, optimizer, scheduler, processor, step, total_steps):
    out = os.path.join(args.output_dir, f"checkpoint-{step}")
    full = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if dist.get_rank() == 0:
        model.save_pretrained(out, state_dict={k: v.to(torch.bfloat16) for k, v in full.items()}, max_shard_size="5GB")
        config = copy.deepcopy(model.config)  # save_pretrained records the fp32 training dtype
        config.dtype, config.use_cache = torch.bfloat16, True
        config.save_pretrained(out)
        processor.save_pretrained(out)
        copy_unloaded_weights(args.model_path, out)
    del full
    if not args.no_save_optimizer:
        state = {"model": get_model_state_dict(model), "optim": get_optimizer_state_dict(model, optimizer)}
        dcp.save(state, checkpoint_id=os.path.join(out, "dcp"))
    if dist.get_rank() == 0:
        with open(os.path.join(out, "trainer_state.json"), "w") as f:
            json.dump({"step": step, "total_steps": total_steps, "scheduler": scheduler.state_dict()}, f)
        ckpts = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")), key=checkpoint_step)
        for old in ckpts[:-args.save_total_limit] if args.save_total_limit > 0 else []:
            shutil.rmtree(old, ignore_errors=True)
        if not args.no_save_optimizer:  # resume state is only needed for the latest checkpoint
            for old in ckpts[:-1]:
                shutil.rmtree(os.path.join(old, "dcp"), ignore_errors=True)
    dist.barrier()
    log0(f"saved {out}")


def checkpoint_step(path):
    """Step number of `.../checkpoint-N` or of a file inside it."""
    name = os.path.basename(path.rstrip("/")) if os.path.isdir(path) else os.path.basename(os.path.dirname(path))
    return int(name.rsplit("-", 1)[1])


def load_checkpoint(args, model, optimizer, scheduler):
    ckpts = [p for p in glob.glob(os.path.join(args.output_dir, "checkpoint-*", "trainer_state.json"))
             if os.path.isdir(os.path.join(os.path.dirname(p), "dcp"))]
    if not ckpts:
        return 0, None
    path = os.path.dirname(max(ckpts, key=checkpoint_step))
    state = {"model": get_model_state_dict(model), "optim": get_optimizer_state_dict(model, optimizer)}
    dcp.load(state, checkpoint_id=os.path.join(path, "dcp"))
    set_model_state_dict(model, state["model"])
    set_optimizer_state_dict(model, optimizer, state["optim"])
    meta = json.load(open(os.path.join(path, "trainer_state.json")))
    scheduler.load_state_dict(meta["scheduler"])
    log0(f"resumed from {path}")
    return meta["step"], meta["total_steps"]


# ----------------------------------------------------------------------------- main


def main():
    args = parse_args()
    dist.init_process_group("cpu:gloo,cuda:nccl", timeout=timedelta(minutes=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    assert args.global_batch_size % world == 0, "global_batch_size must be divisible by the number of GPUs"
    grad_accum = args.global_batch_size // world
    processor = AutoProcessor.from_pretrained(args.model_path)
    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    data_specs = parse_data_specs(args.data)
    if rank == 0:  # one-off scan of record locations, cached in output_dir
        os.makedirs(args.output_dir, exist_ok=True)
        RecordIndex(data_specs, args.output_dir, build=True)
    dist.barrier()
    index = RecordIndex(data_specs, args.output_dir)

    model = build_model(args, device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True)

    # steps of one pass over the weighted mixture, assuming ~98% packing efficiency (cached for resume)
    est_file = os.path.join(args.output_dir, "data_estimate.json")
    est = [None]
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        key = {"data": args.data, "max_len": args.max_len or args.seq_len, "min_pixels": args.min_pixels, "max_pixels": args.max_pixels}
        if os.path.exists(est_file) and json.load(open(est_file))["key"] == key:
            est = [json.load(open(est_file))["tokens"]]
        else:
            est = [estimate_tokens(index, processor, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
                                   max_len=args.max_len or args.seq_len)]
            json.dump({"key": key, "tokens": est[0]}, open(est_file, "w"))
    dist.broadcast_object_list(est, 0)
    steps_per_epoch = est[0] / (0.98 * args.seq_len * args.global_batch_size)
    total_steps = args.max_steps or max(1, round(args.num_epochs * steps_per_epoch))
    save_steps = max(1, round(args.save_every_epochs * steps_per_epoch)) if args.save_every_epochs > 0 else args.save_steps
    warmup = args.warmup_steps or max(1, int(args.warmup_ratio * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_step = 0
    if args.resume:
        start_step, saved_total = load_checkpoint(args, model, optimizer, scheduler)
        total_steps = saved_total or total_steps

    # Resume: the DataLoader reads workers round-robin starting from worker 0, so every worker
    # skips the same number of full rounds and the remaining batches are dropped below.
    consumed, workers = start_step * grad_accum, max(1, args.num_workers)
    # one extra epoch of data so that an over-estimated step count never runs out of data
    dataset = PackedDataset(index, processor, args.seq_len, rank, world, seed=args.seed,
                            epochs=math.ceil(args.num_epochs) + 1, max_len=args.max_len,
                            skip_windows=consumed // workers,
                            min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    loader = DataLoader(dataset, batch_size=None, num_workers=args.num_workers, pin_memory=True,
                        prefetch_factor=args.prefetch_factor if args.num_workers else None)
    log0(f"world={world} grad_accum={grad_accum} max tokens/step={args.global_batch_size * args.seq_len:,} "
         f"steps/epoch={steps_per_epoch:.1f} total_steps={total_steps} save_steps={save_steps} "
         f"warmup={warmup} start_step={start_step}")
    if rank == 0:
        log_file = open(os.path.join(args.output_dir, "train_log.jsonl"), "a")
        tracker = None
        if args.logger != "none":
            tracker = __import__(args.logger)  # swanlab and wandb share init(...) and log(dict, step)
            name_key = "experiment_name" if args.logger == "swanlab" else "name"
            tracker.init(project=args.project, config={**vars(args), "steps_per_epoch": steps_per_epoch, "total_steps": total_steps},
                         **{name_key: args.run_name or os.path.basename(args.output_dir.rstrip("/"))})

    model.train()
    data_iter, step, t0 = iter(loader), start_step, time.time()
    for _ in range(consumed % workers):
        next(data_iter)
    while step < total_steps:
        t_data = time.time()
        micro_batches = list(islice(data_iter, grad_accum))
        t_data = time.time() - t_data
        # [trained tokens, ranks with a full step, all tokens]; stop as soon as one rank runs out of data
        stats = torch.tensor([sum(int((mb["labels"] != IGNORE).sum()) for mb in micro_batches),
                              float(len(micro_batches) == grad_accum),
                              sum(len(mb["input_ids"]) for mb in micro_batches)], device=device)
        dist.all_reduce(stats)
        data_wait = torch.tensor(t_data, device=device)
        dist.all_reduce(data_wait, op=dist.ReduceOp.MAX)
        if stats[1] < world:
            log0("data exhausted")
            break
        n_tokens = stats[0].clamp(min=1)

        loss_sum = torch.zeros((), device=device)
        for mb in micro_batches:
            mb = {k: v.to(device, non_blocking=True) for k, v in mb.items()}
            loss = forward_loss(model, mb, image_pad_id, device)
            (loss * world / n_tokens).backward()  # FSDP averages gradients over ranks
            loss_sum += loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        grad_norm = grad_norm.full_tensor() if hasattr(grad_norm, "full_tensor") else grad_norm
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        dist.all_reduce(loss_sum)
        elapsed, t0 = time.time() - t0, time.time()
        rec = {"step": step, "epoch": round(step / steps_per_epoch, 4),
               "loss": round((loss_sum / n_tokens).item(), 4), "grad_norm": round(grad_norm.item(), 4),
               "lr": scheduler.get_last_lr()[0], "trained_tokens": int(n_tokens),
               "fill": round(stats[2].item() / (args.global_batch_size * args.seq_len), 4),
               "step_time": round(elapsed, 2), "data_wait": round(data_wait.item(), 2),
               "tokens_per_sec_per_gpu": round(stats[2].item() / elapsed / world),
               "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        if rank == 0:
            print(json.dumps(rec), flush=True)
            log_file.write(json.dumps(rec) + "\n")
            log_file.flush()
            if tracker:
                tracker.log(rec, step=step)
        if step % save_steps == 0:
            save_checkpoint(args, model, optimizer, scheduler, processor, step, total_steps)

    if step % save_steps:
        save_checkpoint(args, model, optimizer, scheduler, processor, step, total_steps)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
