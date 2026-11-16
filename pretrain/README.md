<div align="center">
<h1>Medical Multimodal Pre-training</h1>
<p><b>Stage 1 of HuatuoGPT-Vision-2</b></p>
</div>

<p align="center">
📃 <a href="https://openreview.net/pdf?id=0BXKsjKlFJ" target="_blank">Paper</a> ｜
📚 <a href="https://huggingface.co/datasets/FreedomIntelligence/PubMedVision-Plus" target="_blank">PubMedVision-Plus</a> ｜
🏠 <a href="../README.md">HuatuoGPT-Vision-2</a>
</p>

## ⚡ Introduction

This code continues pre-training multimodal LLMs on **PubMedVision-Plus**. It supports the **Qwen3-VL** and **Qwen3.8** model families.

- **Zero preprocessing.** Point the script at raw parquet or jsonl files. Tokenization, image processing and packing all happen on the fly during training.
- **Flexible data mixing.** Image captions, plain text and QA / VQA data can be trained together, each with its own sampling weight.
- **Efficient sequence packing.** Samples of up to 4096 tokens are packed into rows of 32768 tokens with best-fit packing. Over 99% of the tokens are real data and no padding is computed.
- **Isolated attention.** Packed samples never attend to each other. Compute grows with the length of each sample instead of the whole row.
- **Exact resume.** Weights, optimizer state and data position are all restored after an interruption.

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 📚 Data

### PubMedVision-Plus

```bash
huggingface-cli download FreedomIntelligence/PubMedVision-Plus --repo-type dataset --local-dir PubMedVision-Plus
cd PubMedVision-Plus
find images -maxdepth 1 -type f -name '*.tar' -print0 | xargs -0 -P 4 -I {} tar -xf '{}' -C .
```

After extraction, the `core-*.parquet` files can be used for training as they are.

### Data Format

Each data source is a `.parquet`, `.jsonl` or `.json` file, or a folder of them. Three kinds of records are supported:

| Type | Fields | Loss on |
| --- | --- | --- |
| Image caption | `image`, `caption` | Caption |
| Plain text | `text` | All text |
| QA / VQA | `messages`, `images` (optional) | Assistant answers |

```jsonc
// Image caption (the PubMedVision-Plus format)
{"image": "images/000000/0001.jpg", "caption": "Axial CT shows a mass in the left kidney."}

// Plain text (e.g. FineWeb)
{"text": "Hypertension is a common chronic condition ..."}

// QA / VQA
{"messages": [{"role": "user", "content": "<image>\nWhat does this image show?"},
              {"role": "assistant", "content": "A chest X-ray with ..."}],
 "images": ["xray.jpg"]}
```

Image paths may be absolute or relative to the data file. The ShareGPT format (`conversations` with `from` / `value`) is also accepted, and extra fields are ignored.

### Data Mixing

Each source is passed as `path:weight`, where the weight is the number of passes over that source per epoch. A weight of `2` uses it twice. A weight of `0.5` uses half of it per epoch, so every sample is seen once in two epochs. The token count of every source is printed at startup.

```bash
DATA="/path/to/PubMedVision-Plus:1 /path/to/fineweb:0.15"
```

## 🚀 Training

```bash
# Qwen3-VL
MODEL=Qwen/Qwen3-VL-8B-Instruct OUTPUT=outputs/qwen3vl_8b DATA="..." bash run_pretrain.sh

# Qwen3.8 (27B rows of 8192 tokens fit 8 x 80GB GPUs)
MODEL=/path/to/Qwen3.8-27B OUTPUT=outputs/qwen38_27b SEQ_LEN=8192 DATA="..." bash run_pretrain.sh

# Multiple nodes (run on every node)
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MODEL=... DATA="..." bash run_pretrain.sh
```

`--global_batch_size` must be divisible by the total number of GPUs. With more than 16 GPUs, lower `SEQ_LEN` (for example `SEQ_LEN=16384` gives 32 rows per step) so that the tokens per step stay the same. The output folder must be on storage shared by all nodes.

The defaults follow the paper. Any option can be appended to the command, for example `bash run_pretrain.sh --lr 1e-5`. An interrupted run resumes from its latest checkpoint.

| Option | Default | Description |
| --- | --- | --- |
| `--seq_len` | 32768 | Tokens of one packed row on one GPU. Samples in a row never see each other. Use 8192 for 27B models |
| `--max_len` | 4096 | Max tokens of one sample. Longer text is split, image samples are shrunk to fit |
| `--global_batch_size` | 16 | Packed rows per optimizer step (16 x 32768 = 128 x 4096 tokens). `run_pretrain.sh` sets it from `SEQ_LEN` |
| `--lr` | 2e-5 | Peak learning rate with cosine decay |
| `--num_epochs` | 1 | Passes over the weighted data mix |
| `--max_pixels` | 2097152 | Max total pixels per image (about 2048 visual tokens). Larger images are downscaled, keeping the aspect ratio |
| `--freeze_vision` | off | Freeze the vision encoder |
| `--cpu_offload` | off | Offload weights and optimizer states to CPU |
| `--save_steps` | 1000 | Checkpoint interval (or `--save_every_epochs`) |
| `--logger` | none | `swanlab` or `wandb` |

Each `checkpoint-N/` is a standard Hugging Face model that can be loaded with `transformers` or served with vLLM and SGLang. Only the latest checkpoint keeps the `dcp/` folder used for resuming.

## 🔍 Design

| Component | Approach |
| --- | --- |
| Data index | A one-off scan records where every sample starts (a byte offset or a row), cached in the output folder. Each epoch shuffles all samples of all sources together and deals them evenly to every GPU and loader worker |
| Images | Each image keeps its resolution up to `--max_pixels`. If a sample would exceed `--max_len`, its images are shrunk just enough to fit, so no image sample is dropped |
| Packing | Each sample goes into the fullest row that still has room, and a row is used once it is 98% full |
| Attention | Full attention runs flash-attn varlen. The linear attention of Qwen3.8 restarts its state at every sample. Positions restart from 0 in each sample |
| Loss | Every trained token in a step has equal weight. Logits are computed only where a loss is needed |
| Parallelism | PyTorch FSDP2 with fp32 master weights and bf16 compute |
