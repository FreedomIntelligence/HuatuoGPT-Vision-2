"""Reward for medical RL: verifiable multiple choice + rubric grading with a (multimodal) grader.

Multiple choice: 1 if the option in the last <answer>X</answer> matches the label, else 0.
Open ended: the grader judges every rubric item of the final answer in one generation. When the
task has images, they are sent to the grader together with the grading prompt, at their original
resolution unless GRADER_MAX_PIXELS is set (the grader's own processor resizes them). The score is the
matched points divided by the total positive points, clipped to [0, 1].
Thinking models (REQUIRE_THINKING=true): a response without </think> gets 0.
Answer length: the final answer (after </think>, or the whole response without thinking) loses
ANSWER_PENALTY_FACTOR x (tokens - ANSWER_LEN_START) / (ANSWER_LEN_END - ANSWER_LEN_START), clipped to
[0, ANSWER_PENALTY_FACTOR]. The penalty on the whole response is DAPO's overlong shaping (see run_rl.sh).
The grader is any OpenAI-compatible chat endpoint, e.g. HuatuoGPT-Vision-2-Grader served by SGLang.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from functools import lru_cache
from typing import Any

import requests
from PIL import Image

GRADING_TEMPLATE = """Score the assistant's response against each rubric item.

## Conversation
<Conversation>
{conversation}
</Conversation>

## Rubrics
<Rubric_items>
{rubrics}
</Rubric_items>

## Output
Return only a JSON list of booleans, one for each rubric in order.
- true: the criterion is met
- false: the criterion is not met
- For a negative criterion (bad behavior with negative points), return true only when the response shows that bad behavior."""

ANSWER_RE = re.compile(r"<answer>\s*([A-Za-z])\s*</answer>", flags=re.IGNORECASE)
KEYS = ("score", "acc", "success", "error", "predicted_answer", "rubric_hits", "answer_penalty")  # same keys for every sample


def final_answer(text: str) -> str:
    """The user-visible answer, without the thinking part."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()


def extract_choice(text: str) -> str | None:
    matches = ANSWER_RE.findall(final_answer(text))
    return matches[-1].upper() if matches else None


def conversation_text(messages: list[dict], answer: str) -> str:
    lines = [f"{m['role']}: {m['content']}" for m in messages]
    return "\n".join(lines + [f"assistant: {answer}"])


def rubrics_text(rubrics: list[dict]) -> str:
    return "\n".join(f"{i}. ({float(r['points']):+g}pts) {r['criterion']}" for i, r in enumerate(rubrics, 1))


def grading_prompt(messages: list[dict], answer: str, rubrics: list[dict]) -> str:
    return GRADING_TEMPLATE.format(conversation=conversation_text(messages, answer[:10000]), rubrics=rubrics_text(rubrics))


def parse_grades(text: str, expected: int) -> list[bool] | None:
    for candidate in re.findall(r"\[[\s\S]*?\]", text) + [text]:
        try:
            values = json.loads(candidate.lower())
        except json.JSONDecodeError:
            continue
        if isinstance(values, list) and len(values) == expected and all(isinstance(v, bool) for v in values):
            return values
    return None


def rubric_score(rubrics: list[dict], decisions: list[bool]) -> float:
    earned = sum(float(r["points"]) for r, hit in zip(rubrics, decisions) if hit)
    positive = sum(max(float(r["points"]), 0.0) for r in rubrics)
    return max(0.0, min(1.0, earned / positive)) if positive else 0.0


@lru_cache(maxsize=256)
def image_url(path: str) -> str:
    """Data URL of an image file. Only images above GRADER_MAX_PIXELS (0 = no limit) are downscaled."""
    max_pixels = int(os.environ.get("GRADER_MAX_PIXELS", "0"))
    with Image.open(path) as img:
        scale = (max_pixels / (img.width * img.height)) ** 0.5 if max_pixels > 0 else 1.0
        if scale >= 1 and img.format in ("JPEG", "PNG", "WEBP"):  # send the original file
            with open(path, "rb") as f:
                return f"data:image/{img.format.lower()};base64," + base64.b64encode(f.read()).decode()
        img = img.convert("RGB")
        if scale < 1:
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BICUBIC)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def request_grader(prompt: str, images: list[str], temperature: float) -> str:
    content = [{"type": "image_url", "image_url": {"url": image_url(p)}} for p in images]
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": os.environ.get("REWARD_MODEL_NAME", "default"),
        "messages": [{"role": "user", "content": content if images else prompt}],
        "max_tokens": int(os.environ.get("REWARD_MODEL_MAX_TOKENS", "512")),
        "temperature": temperature,
    }
    url = os.environ.get("REWARD_MODEL_URL", "http://127.0.0.1:30001/v1/chat/completions")
    response = requests.post(url, json=payload, timeout=float(os.environ.get("REWARD_MODEL_TIMEOUT", "180")))
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"] or ""


def grade(solution: str, ground_truth: dict) -> dict:
    rubrics = ground_truth["rubrics"]
    prompt = grading_prompt(ground_truth["prompt"], final_answer(solution), rubrics)
    error = "invalid_grader_output"
    for attempt, temperature in enumerate((0.0, 0.2, 0.4)):  # retry with a little sampling on bad output
        try:
            decisions = parse_grades(request_grader(prompt, ground_truth.get("images") or [], temperature), len(rubrics))
            if decisions is not None:
                score = rubric_score(rubrics, decisions)
                return {"score": score, "acc": score, "success": True, "rubric_hits": sum(decisions)}
        except (requests.RequestException, KeyError, IndexError, ValueError, OSError) as e:
            error = type(e).__name__
        time.sleep(2**attempt)
    print(f"[reward] grading failed ({error}), reward 0", flush=True)
    return {"score": 0.0, "acc": 0.0, "success": False, "error": error}


@lru_cache(maxsize=1)
def policy_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(os.environ["POLICY_TOKENIZER"])


def answer_penalty(solution: str) -> float:
    """Linear penalty on the final answer: 0 up to ANSWER_LEN_START tokens, -factor from ANSWER_LEN_END on."""
    start, end = int(os.environ.get("ANSWER_LEN_START", "200")), int(os.environ.get("ANSWER_LEN_END", "4096"))
    factor = float(os.environ.get("ANSWER_PENALTY_FACTOR", "0.4"))
    tokens = len(policy_tokenizer().encode(final_answer(solution), add_special_tokens=False))
    return -factor * min(max(tokens - start, 0) / max(end - start, 1), 1.0)


def compute_score(data_source: str, solution_str: str, ground_truth: dict, extra_info: dict | None = None,
                  **kwargs: Any) -> dict:
    thinking = os.environ.get("REQUIRE_THINKING", "false").lower() == "true"
    if thinking and "</think>" not in solution_str:
        result = {"score": 0.0, "acc": 0.0, "success": True, "error": "missing_thinking"}
    elif ground_truth["type"] == "open_ended":
        result = grade(solution_str, ground_truth)
    else:
        predicted = extract_choice(solution_str)
        correct = float(predicted is not None and predicted == ground_truth["answer_idx"])
        result = {"score": correct, "acc": correct, "success": True, "predicted_answer": predicted or ""}
    if float(os.environ.get("ANSWER_PENALTY_FACTOR", "0.4")) > 0 and result.get("success") and not result.get("error"):
        result["answer_penalty"] = answer_penalty(solution_str)
        result["score"] += result["answer_penalty"]
    defaults = {"score": 0.0, "acc": 0.0, "success": False, "error": "", "predicted_answer": "", "rubric_hits": 0,
                "answer_penalty": 0.0}
    return {key: result.get(key, defaults[key]) for key in KEYS}
