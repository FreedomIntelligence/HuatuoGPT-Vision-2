"""Streaming data pipeline for multimodal pre-training.

records (parquet / jsonl / json)  ->  tokenized samples  ->  packed windows of the training length

A one-off scan records where every record starts (jsonl byte offset / parquet row), cached next to
the outputs. Each epoch shuffles all records of all sources globally and deals them out round-robin
to the (rank, worker) shards, so every shard sees the same dataset mix and the same record count.

Everything happens on the fly inside DataLoader workers, so no offline tokenization is needed.
Supported record formats (one record per row / line):

  caption       {"image": "images/0001.jpg", "caption": "..."}              loss on caption
  text          {"text": "..."}                                             loss on all text
  conversation  {"messages": [{"role": "user", "content": "<image>\\n..."},  loss on assistant turns
                              {"role": "assistant", "content": "..."}],
                 "images": ["a.jpg"]}

Relative image paths are resolved against the directory of the data file.
"image_path" is accepted as an alias of "image".

Two lengths are involved: `max_len` caps a single sample and `seq_len` is the packed window.
Images keep their own resolution up to `max_pixels`. When the images of a sample would push it
past `max_len`, they are shrunk together just enough to fit, so image samples are not dropped.
"""

import bisect
import functools
import glob
import io
import json
import os
import random
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

IGNORE = -100
IMAGE_PLACEHOLDER = "<image>"
Image.MAX_IMAGE_PIXELS = None


# ----------------------------------------------------------------------------- reading


def parse_data_specs(specs):
    """'path[:weight]' -> [(path, weight)]. weight = number of passes over the dataset (may be fractional)."""
    out = []
    for spec in specs:
        path, _, weight = spec.rpartition(":") if ":" in spec else (spec, "", "1")
        out.append((path, float(weight)))
    return out


def list_files(path):
    if not os.path.isdir(path):
        return [path]
    # top level only: dataset folders often contain millions of images in sub-directories
    return sorted(f for ext in ("parquet", "jsonl", "json") for f in glob.glob(os.path.join(path, f"*.{ext}")))


# ----------------------------------------------------------------------------- record index


def _scan_file(path):
    """Record locations of one file: byte offsets of jsonl lines, row numbers otherwise."""
    if path.endswith(".jsonl"):
        starts, pos = [np.zeros(1, np.int64)], 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(64 << 20), b""):
                starts.append(np.flatnonzero(np.frombuffer(chunk, np.uint8) == 10).astype(np.int64) + pos + 1)
                pos += len(chunk)
        starts = np.concatenate(starts)
        return starts[starts < pos]  # drop the position after the final newline
    if path.endswith(".parquet"):
        return np.arange(pq.ParquetFile(path).metadata.num_rows, dtype=np.int64)
    return np.arange(len(json.load(open(path))), dtype=np.int64)


class RecordIndex:
    """Location of every record: files, file_id[i], loc[i]; source s owns rows bounds[s]:bounds[s + 1]."""

    def __init__(self, data_specs, cache_dir, build=False, workers=32):
        self.files = [f for path, _ in data_specs for f in list_files(path)]
        self.weights = [w for _, w in data_specs]
        # file sizes + how files are grouped into sources, so a different DATA grouping rebuilds the index
        key = [[f, os.path.getsize(f)] for f in self.files] + [len(list_files(path)) for path, _ in data_specs]
        meta_path = os.path.join(cache_dir, "record_index.json")
        cached = os.path.exists(meta_path) and json.load(open(meta_path))["key"] == key
        if build and not cached:
            with Pool(min(workers, len(self.files))) as pool:
                locs = pool.map(_scan_file, self.files)
            np.save(os.path.join(cache_dir, "record_index_file.npy"),
                    np.concatenate([np.full(len(l), i, np.int32) for i, l in enumerate(locs)]))
            np.save(os.path.join(cache_dir, "record_index_loc.npy"), np.concatenate(locs))
            counts, k = [], 0
            for path, _ in data_specs:
                n = len(list_files(path))
                counts.append(sum(len(l) for l in locs[k:k + n]))
                k += n
            json.dump({"key": key, "counts": counts}, open(meta_path, "w"))
        counts = json.load(open(meta_path))["counts"]
        self.bounds = np.concatenate([[0], np.cumsum(counts)]).tolist()
        self.file_id = np.load(os.path.join(cache_dir, "record_index_file.npy"), mmap_mode="r")
        self.loc = np.load(os.path.join(cache_dir, "record_index_loc.npy"), mmap_mode="r")

    def epoch_order(self, seed, epoch):
        """All record ids of one epoch in random order.

        Each source is consumed as an endless stream of random permutations of its records, and
        epoch e takes the next weight x n records of that stream. A weight of 0.5 thus covers every
        record once in two epochs, and 1.2 uses every record once per epoch plus 20% more that
        rotate through the source, so no record is skipped while others repeat."""
        ids = []
        for s, w in enumerate(self.weights):
            lo, n = self.bounds[s], self.bounds[s + 1] - self.bounds[s]
            start, end = round(epoch * w * n), round((epoch + 1) * w * n)
            for cycle in range(start // n, -(-end // n)):
                perm = np.random.default_rng([seed, s, cycle]).permutation(n)
                ids.append(lo + perm[max(start - cycle * n, 0):min(end - cycle * n, n)])
        ids = np.concatenate(ids)
        return ids[np.random.default_rng([seed, epoch]).permutation(len(ids))]


class RecordReader:
    """Random access to single records, with open files and the last parquet row groups cached."""

    def __init__(self, index):
        self.index, self.handles = index, {}

    def __call__(self, i):
        path = self.index.files[self.index.file_id[i]]
        loc = int(self.index.loc[i])
        if path.endswith(".jsonl"):
            if path not in self.handles:
                self.handles[path] = open(path, "rb")
            f = self.handles[path]
            f.seek(loc)
            line = f.readline()
            return json.loads(line) if line.strip() else None
        if path.endswith(".parquet"):
            starts = self._row_group_starts(path)
            g = bisect.bisect_right(starts, loc) - 1
            return self._row_group(path, g).slice(loc - starts[g], 1).to_pylist()[0]
        return self._json(path)[loc]

    @functools.lru_cache(maxsize=None)
    def _row_group_starts(self, path):
        md = pq.ParquetFile(path).metadata
        return np.concatenate([[0], np.cumsum([md.row_group(g).num_rows for g in range(md.num_row_groups)])]).tolist()

    @functools.lru_cache(maxsize=4)
    def _row_group(self, path, g):
        return pq.ParquetFile(path).read_row_group(g)

    @functools.lru_cache(maxsize=2)
    def _json(self, path):
        return json.load(open(path))


def open_image(image, root):
    if isinstance(image, dict):  # HF datasets Image feature: {"bytes": ..., "path": ...}
        image = io.BytesIO(image["bytes"]) if image.get("bytes") else image["path"]
    if isinstance(image, str) and not os.path.isabs(image):
        image = os.path.join(root, image)
    return Image.open(image)


# ----------------------------------------------------------------------------- tokenization


class SampleBuilder:
    """Turns one record into {"input_ids", "labels", "images", "image_hw", "splittable"}."""

    def __init__(self, processor, min_pixels=None, max_pixels=None, max_len=4096):
        self.max_len = max_len
        self.tok = processor.tokenizer
        self.image_processor = processor.image_processor
        size = self.image_processor.size
        self.min_pixels = min_pixels or size["shortest_edge"]
        self.max_pixels = max_pixels or size["longest_edge"]
        self.patch, self.merge = self.image_processor.patch_size, self.image_processor.merge_size

        tid = self.tok.convert_tokens_to_ids
        self.eot, self.im_end = tid("<|endoftext|>"), tid("<|im_end|>")
        self.vision_start, self.vision_end, self.image_pad = tid("<|vision_start|>"), tid("<|vision_end|>"), tid("<|image_pad|>")
        self.image_str = "<|vision_start|><|image_pad|><|vision_end|>"
        self.assistant_prefix = self.encode("<|im_start|>assistant\n")
        self.empty_think = self.encode("<think>\n\n</think>\n\n")

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def fit_images(self, images, root, n_text):
        """Resized (h, w) of every image. Images larger than `max_pixels` are scaled down, and if the
        sample would be longer than `max_len`, all its images share the remaining token budget."""
        sizes = []
        for image in images:
            with open_image(image, root) as img:  # reads the header only
                sizes.append(img.size)
        factor = self.patch * self.merge
        max_pixels = self.max_pixels
        while True:
            hw = [smart_resize(h, w, factor=factor, min_pixels=self.min_pixels, max_pixels=max_pixels) for w, h in sizes]
            budget = self.max_len - n_text
            if n_text + sum(self.n_tokens(x) for x in hw) <= self.max_len or max_pixels <= self.min_pixels:
                return hw  # fits, or the text alone is too long and pack() drops the sample
            max_pixels = max(self.min_pixels, min(max_pixels - factor**2, budget // len(sizes) * factor**2))

    def n_tokens(self, hw):
        return hw[0] * hw[1] // (self.patch * self.merge) ** 2

    def __call__(self, rec, root):
        try:
            if rec.get("messages") or rec.get("conversations"):
                return self.conversation(rec, root)
            if rec.get("caption"):
                return self.caption(rec, root)
            if rec.get("text"):
                return self.text(rec)
        except Exception as e:  # broken image / malformed record: drop it
            print(f"[data] skip record: {type(e).__name__}: {e}", flush=True)
        return None

    def text(self, rec):
        ids = self.encode(rec["text"])
        return dict(input_ids=ids + [self.eot], labels=ids + [IGNORE], images=[], image_hw=[], splittable=True)

    def caption(self, rec, root):
        image = rec.get("image") or rec["image_path"]
        cap = self.encode("\n" + rec["caption"].strip())
        hw = self.fit_images([image], root, len(cap) + 3)
        prefix = [self.vision_start] + [self.image_pad] * self.n_tokens(hw[0]) + [self.vision_end]
        return dict(
            input_ids=prefix + cap + [self.eot],
            labels=[IGNORE] * len(prefix) + cap + [IGNORE],
            images=[image], image_hw=hw, splittable=False,
        )

    def conversation(self, rec, root):
        images = rec.get("images") or rec.get("image") or rec.get("image_path") or []
        images = [images] if isinstance(images, (str, dict)) else list(images)
        messages = []
        for m in rec.get("messages") or rec["conversations"]:
            role = m.get("role") or {"human": "user", "gpt": "assistant"}.get(m["from"], m["from"])
            messages.append({"role": role, "content": m.get("content", m.get("value"))})
        n_placeholders = sum(m["content"].count(IMAGE_PLACEHOLDER) for m in messages)
        if n_placeholders == 0 and images:  # no placeholder: put all images before the first user turn
            first = next(m for m in messages if m["role"] == "user")
            first["content"] = IMAGE_PLACEHOLDER * len(images) + "\n" + first["content"]
        elif n_placeholders != len(images):
            raise ValueError(f"{n_placeholders} <image> placeholders but {len(images)} images")
        for m in messages:
            m["content"] = m["content"].replace(IMAGE_PLACEHOLDER, self.image_str)

        ids = self.encode(self.tok.apply_chat_template(messages, tokenize=False))
        hw = self.fit_images(images, root, len(ids) - len(images) + 1)
        expanded, k = [], 0
        for t in ids:  # expand each <|image_pad|> to the real number of visual tokens
            if t == self.image_pad:
                expanded += [t] * self.n_tokens(hw[k])
                k += 1
            else:
                expanded.append(t)
        labels = self.assistant_labels(expanded)
        return dict(input_ids=expanded + [self.eot], labels=labels + [IGNORE],
                    images=images, image_hw=hw, splittable=False)

    def assistant_labels(self, ids):
        """Train on assistant replies (including <|im_end|>), skipping an empty <think></think> block."""
        labels, p, n = [IGNORE] * len(ids), self.assistant_prefix, len(self.assistant_prefix)
        for i in range(len(ids) - n + 1):
            if ids[i:i + n] != p:
                continue
            start = i + n
            if ids[start:start + len(self.empty_think)] == self.empty_think:
                start += len(self.empty_think)
            end = next((j for j in range(start, len(ids)) if ids[j] == self.im_end), len(ids) - 1)
            labels[start:end + 1] = ids[start:end + 1]
        return labels

    def pixels(self, images, roots, image_hw):
        """Decode the images of one window into (pixel_values, image_grid_thw), resized to `image_hw`."""
        imgs = []
        for img, root, (h, w) in zip(images, roots, image_hw):
            img = open_image(img, root)
            img.draft("RGB", (w, h))  # JPEG: decode directly at a reduced scale that is still >= (w, h)
            imgs.append(img.convert("RGB").resize((w, h), Image.BICUBIC))
        out = self.image_processor(images=imgs, do_resize=False, return_tensors="pt")
        grid = out["image_grid_thw"]
        if grid[:, 1:].mul(self.patch).tolist() != [list(hw) for hw in image_hw]:
            raise ValueError("image size mismatch")
        return out["pixel_values"].to(torch.bfloat16), grid


# ----------------------------------------------------------------------------- packing


def pack(samples, seq_len, max_len, rng, stats, min_fill=0.98, max_open=64):
    """Online best-fit packing into windows of at most `seq_len` tokens.

    Open windows are kept sorted by length. Each sample goes into the fullest window it still
    fits in; a window is emitted once it reaches `min_fill`, or when more than `max_open` windows
    are open (the fullest one is emitted). Text longer than `max_len` is cut into `max_len`
    chunks; other overlong samples (conversations with very long text) are dropped.
    """
    open_windows, min_len = [], int(seq_len * min_fill)  # [(used, [samples])], ascending by used

    def chunks():
        for s in samples:
            n = len(s["input_ids"])
            if n <= max_len:
                yield s
            elif s["splittable"]:
                for i in range(0, n, max_len):
                    yield {**s, "input_ids": s["input_ids"][i:i + max_len], "labels": s["labels"][i:i + max_len]}
            else:
                stats["overlong"] += 1

    def emit(items):
        rng.shuffle(items)
        return items

    for s in chunks():
        n = len(s["input_ids"])
        i = bisect.bisect_right(open_windows, seq_len - n, key=lambda w: w[0])
        used, items = open_windows.pop(i - 1) if i else (0, [])
        used += n
        items.append(s)
        if used >= min_len:
            yield emit(items)
        else:
            bisect.insort(open_windows, (used, items), key=lambda w: w[0])
        if len(open_windows) > max_open:
            yield emit(open_windows.pop()[1])
    while open_windows:
        yield emit(open_windows.pop()[1])


# ----------------------------------------------------------------------------- dataset


class PackedDataset(IterableDataset):
    """Each item is one packed row of at most `seq_len` tokens: input_ids, labels, cu_seqlens (+ images).

    Every epoch is one global shuffle of all records, dealt round-robin to the (rank, worker)
    shards, so all shards read disjoint data with the same mix and the pipeline is deterministic
    for a given seed, world size and number of workers.
    """

    def __init__(self, index, processor, seq_len, rank, world_size, seed=42, epochs=1, max_len=None,
                 skip_windows=0, min_pixels=None, max_pixels=None):
        """`seq_len` is the training window: samples of at most `max_len` tokens are packed on the fly
        to this length. `skip_windows` is skipped by every worker (used to resume)."""
        self.index = index
        self.max_len = max_len or seq_len
        self.builder = SampleBuilder(processor, min_pixels, max_pixels, self.max_len)
        self.seq_len, self.rank, self.world_size, self.seed, self.epochs = seq_len, rank, world_size, seed, epochs
        self.skip_windows = skip_windows

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        torch.set_num_threads(1)  # many workers per GPU: one thread each for the image processor
        shard, n_shards = self.rank * nw + wid, self.world_size * nw
        rng = random.Random(self.seed * 1000 + shard)
        stats = {"overlong": 0}

        read = RecordReader(self.index)
        roots = [os.path.dirname(os.path.abspath(f)) for f in self.index.files]

        def samples():
            for epoch in range(self.epochs):
                for i in self.index.epoch_order(self.seed, epoch)[shard::n_shards]:
                    rec, root = read(i), roots[self.index.file_id[i]]
                    s = self.builder(rec, root) if rec else None
                    if s is not None:
                        s["roots"] = [root] * len(s["images"])
                        yield s

        for i, window in enumerate(pack(samples(), self.seq_len, self.max_len, rng, stats)):
            if i < self.skip_windows:  # resuming: windows are rebuilt from image headers only, pixels are not decoded
                continue
            item = self.to_tensors(window)
            if item is not None:
                yield item
        if stats["overlong"]:
            print(f"[data] shard {shard}: dropped {stats['overlong']} samples longer than {self.max_len}", flush=True)

    def to_tensors(self, window):
        """One packed row without padding. `cu_seqlens` marks sample boundaries for varlen attention.
        A sample whose images fail to decode is dropped on its own; the rest of the row is kept."""
        kept, pixels, grids = [], [], []
        for s in window:
            if s["images"]:
                try:
                    pv, grid = self.builder.pixels(s["images"], s["roots"], s["image_hw"])
                except Exception as e:
                    print(f"[data] skip sample: {type(e).__name__}: {e}", flush=True)
                    continue
                pixels.append(pv)
                grids.append(grid)
            kept.append(s)
        if not kept:
            return None
        lengths = [len(s["input_ids"]) for s in kept]
        item = {
            "input_ids": torch.tensor([t for s in kept for t in s["input_ids"]]),
            # the first token of a sample cannot be predicted from inside the sample
            "labels": torch.tensor([t for s in kept for t in [IGNORE] + s["labels"][1:]]),
            "cu_seqlens": torch.tensor([0] + lengths).cumsum(0).int(),
            "images_per_sample": torch.tensor([len(s["images"]) for s in kept]),
        }
        if pixels:
            item["pixel_values"], item["image_grid_thw"] = torch.cat(pixels), torch.cat(grids)
        return item


_BUILDER, _READER = None, None


def _probe_init(processor, kw, index):
    global _BUILDER, _READER
    _BUILDER, _READER = SampleBuilder(processor, **kw), RecordReader(index)


def _probe(i):
    """Token length of record i. Samples longer than max_len: text is split, the rest is dropped."""
    rec = _READER(i)
    s = _BUILDER(rec, os.path.dirname(os.path.abspath(_READER.index.files[_READER.index.file_id[i]]))) if rec else None
    if s is None:
        return None
    n = len(s["input_ids"])
    return n if n <= _BUILDER.max_len or s["splittable"] else 0


def estimate_tokens(index, processor, n_probe=4000, workers=64, seed=0, **kw):
    """Tokens (text + visual) of one epoch: record count x weight x mean length of `n_probe` random records."""
    rng, total = np.random.default_rng(seed), 0
    with Pool(workers, initializer=_probe_init, initargs=(processor, kw, index)) as pool:
        for s, w in enumerate(index.weights):
            lo, hi = index.bounds[s], index.bounds[s + 1]
            ids = (lo + rng.choice(hi - lo, min(n_probe, hi - lo), replace=False)).tolist()
            lengths = [n for n in pool.map(_probe, ids, chunksize=16) if n is not None]
            avg = sum(lengths) / max(1, len(lengths))
            total += w * (hi - lo) * avg
            name = os.path.dirname(index.files[index.file_id[lo]])
            print(f"[data] {name}: {hi - lo:,} records x {avg:.0f} tokens x weight {w} = {w * (hi - lo) * avg / 1e9:.3f}B tokens", flush=True)
    return total
