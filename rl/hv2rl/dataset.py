"""Medical RL dataset for verl: multiple-choice and rubric-based open-ended tasks, with or without images.

Records (json / jsonl / parquet, or a folder of them):

  multiple choice  {"question": ..., "options": {"A": ...}, "answer_idx": "A", "images": [...]}
  open ended       {"question": ..., "rubrics": [{"criterion": ..., "points": 3}], "images": [...]}

`images` is optional. Relative image paths are resolved against the data file's folder or one of
its parents, so PubMedVision-RL (`images/..` relative to the repository root) works as downloaded.
Instead of `question`, a record may give `prompt` as a list of chat messages.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any

import datasets
import torch
from verl.utils.dataset.rl_dataset import RLHFDataset

MC_INSTRUCTION = (
    "Answer the multiple-choice question and place the final selection within the <answer> tag. "
    "Example: <answer>A</answer>."
)


def expand_files(paths) -> list[str]:
    files = []
    for path in [paths] if isinstance(paths, str) else paths:
        if os.path.isdir(path):
            files += sorted(f for ext in ("jsonl", "json", "parquet")
                            for f in glob.glob(os.path.join(path, "**", f"*.{ext}"), recursive=True))
        else:
            files.append(path)
    return files


def read_records(path: str) -> list[dict]:
    if path.endswith(".parquet"):
        return list(datasets.load_dataset("parquet", data_files=path)["train"])
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def resolve_image(path: str, data_file: str) -> str:
    if os.path.isabs(path):
        return path
    base = os.path.dirname(os.path.abspath(data_file))
    for _ in range(4):  # the file's folder and up to three parents
        candidate = os.path.join(base, path)
        if os.path.exists(candidate):
            return candidate
        base = os.path.dirname(base)
    raise FileNotFoundError(f"image {path} not found near {data_file}")


def user_messages(row: dict) -> list[dict]:
    """Text-only chat messages of the task. Images are added separately."""
    if row.get("prompt"):
        prompt = row["prompt"]
        return [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
    question = row["question"].strip()
    if row.get("options"):
        options = "\n".join(f"{k}. {v}" for k, v in sorted(row["options"].items()) if v is not None)
        question = f"{MC_INSTRUCTION}\n{question}\n{options}"
    return [{"role": "user", "content": question}]


class MedicalRLDataset(RLHFDataset):
    """Normalizes mixed medical RL records into verl's RL dataset contract."""

    def _normalize(self, row: dict, data_file: str, index: int) -> dict:
        images = [resolve_image(p, data_file) for p in (row.get("images") or [])]
        messages = user_messages(row)
        if row.get("options"):
            ground_truth = {"type": "multiple_choice", "answer_idx": str(row["answer_idx"]).strip().upper()}
        elif row.get("rubrics"):
            ground_truth = {"type": "open_ended", "rubrics": row["rubrics"], "prompt": messages, "images": images}
        else:
            raise ValueError(f"record {row.get('id', index)} has neither options nor rubrics")
        return {
            "messages_json": json.dumps(messages, ensure_ascii=False),
            "images": images,
            "ground_truth_json": json.dumps(ground_truth, ensure_ascii=False),
            "data_source": str(row.get("data_source", ground_truth["type"])),
            "index": index,
        }

    def _read_files_and_tokenize(self) -> None:
        records = []
        for data_file in expand_files(self.data_files):
            records += [self._normalize(row, data_file, len(records) + i) for i, row in enumerate(read_records(data_file))]
        if 0 < self.max_samples < len(records):
            generator = torch.Generator().manual_seed(self.seed or 0)
            keep = torch.randperm(len(records), generator=generator)[: self.max_samples] if self.shuffle else range(self.max_samples)
            records = [records[int(i)] for i in keep]
        self.dataframe = datasets.Dataset.from_list(records)
        tokenizer, limit, cap = self.tokenizer, self.max_prompt_length, int(self.config.get("max_image_tokens", 0))

        def image_budget(row) -> dict:
            """Images share whatever the prompt length leaves after the text (optionally capped per image)."""
            messages = json.loads(row["messages_json"])
            text = len(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"])
            n = len(row["images"])
            tokens = (limit - text - 8 * n) // n if n else 0  # 8: vision start / end tokens and a small margin
            return {"text_tokens": text, "image_tokens": min(tokens, cap) if n and cap > 0 else tokens}

        self.dataframe = self.dataframe.map(image_budget, num_proc=self.num_workers, desc="Image token budget")
        if self.filter_overlong_prompts:
            self.dataframe = self.dataframe.filter(
                lambda row: row["text_tokens"] <= limit and (not row["images"] or row["image_tokens"] >= 64),
                num_proc=self.num_workers, desc="Filtering overlong prompts")
        print(f"MedicalRLDataset: {len(self.dataframe)} samples from {self.data_files}")

    def _image_item(self, path: str, max_tokens: int) -> dict:
        """One token covers (patch_size * merge_size)^2 pixels. Larger images are downscaled to fit."""
        image_processor = self.processor.image_processor
        pixels = (image_processor.patch_size * image_processor.merge_size) ** 2
        return {"type": "image", "image": path, "min_pixels": min(64, max_tokens) * pixels, "max_pixels": max_tokens * pixels}

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.dataframe[item]
        messages = json.loads(row["messages_json"])
        if row["images"]:  # images go before the text of the first user turn
            first = next(m for m in messages if m["role"] == "user")
            first["content"] = [self._image_item(p, row["image_tokens"]) for p in row["images"]] + [{"type": "text", "text": first["content"]}]
        ground_truth = json.loads(row["ground_truth_json"])
        return {
            "raw_prompt": messages,
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "data_source": row["data_source"],
            "reward_model": {"ground_truth": ground_truth, "style": ground_truth["type"]},
            "index": row["index"],
            "extra_info": {"index": row["index"]},
            "tools_kwargs": {},
            "interaction_kwargs": {},
        }
