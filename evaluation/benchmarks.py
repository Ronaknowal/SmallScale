"""Evaluation suite: perplexity, HellaSwag, ARC-Easy (few-shot).

These are standard small-scale benchmarks. At 150M params, don't expect
great absolute numbers — the value is in comparing across ablations.
"""

import torch
import torch.nn.functional as F
import json
import math
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm


@torch.no_grad()
def eval_perplexity(model, dataloader, device: str, max_batches: int = 100) -> float:
    """Compute perplexity on a dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        out = model(x, targets=y)
        total_loss += out["loss"].item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


@torch.no_grad()
def eval_hellaswag(
    model,
    tokenizer,
    device: str,
    data_path: str = "./data/hellaswag_val.jsonl",
    max_examples: int = 200,
) -> float:
    """Evaluate on HellaSwag (sentence completion) via log-likelihood scoring.

    For each example, we compute the log-likelihood of each candidate ending
    given the context, and pick the one with highest likelihood.
    """
    model.eval()

    if not Path(data_path).exists():
        print(f"HellaSwag data not found at {data_path}. Skipping.")
        print("Download from: https://github.com/rowanz/hellaswag")
        return 0.0

    with open(data_path) as f:
        examples = [json.loads(line) for line in f][:max_examples]

    correct = 0
    total = 0

    for ex in tqdm(examples, desc="HellaSwag"):
        context = ex["ctx"]
        endings = ex["endings"]
        label = int(ex["label"])

        scores = []
        for ending in endings:
            full_text = context + " " + ending
            tokens = tokenizer.encode(full_text)
            ctx_len = len(tokenizer.encode(context))

            if len(tokens) > model.cfg.max_seq_len:
                tokens = tokens[:model.cfg.max_seq_len]

            input_ids = torch.tensor([tokens[:-1]], device=device)
            target_ids = torch.tensor([tokens[1:]], device=device)

            out = model(input_ids)
            logits = out["logits"]

            # Only score the ending tokens (after context)
            ending_logits = logits[0, ctx_len - 1:]
            ending_targets = target_ids[0, ctx_len - 1:]

            log_probs = F.log_softmax(ending_logits, dim=-1)
            token_log_probs = log_probs.gather(1, ending_targets.unsqueeze(-1)).squeeze(-1)
            score = token_log_probs.mean().item()  # length-normalized
            scores.append(score)

        pred = max(range(len(scores)), key=lambda i: scores[i])
        if pred == label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"HellaSwag accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


@torch.no_grad()
def eval_text_generation(model, tokenizer, prompts: List[str], device: str,
                         max_tokens: int = 100) -> List[str]:
    """Generate text from prompts for qualitative evaluation."""
    model.eval()
    generations = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=device)
        output_ids = model.generate(input_ids, max_new_tokens=max_tokens)
        generated = tokenizer.decode(output_ids[0].tolist())
        generations.append(generated)
    return generations


def run_all_evals(model, tokenizer, val_loader, device: str) -> dict:
    """Run full evaluation suite and return metrics dict."""
    results = {}

    # Perplexity
    ppl = eval_perplexity(model, val_loader, device)
    results["perplexity"] = round(ppl, 2)
    print(f"Perplexity: {ppl:.2f}")

    # HellaSwag
    acc = eval_hellaswag(model, tokenizer, device)
    results["hellaswag_acc"] = round(acc, 4)

    # Qualitative generation
    test_prompts = [
        "The meaning of life is",
        "In a groundbreaking study, researchers found that",
        "Once upon a time in a land far away,",
    ]
    generations = eval_text_generation(model, tokenizer, test_prompts, device)
    for prompt, gen in zip(test_prompts, generations):
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {gen[:200]}...")

    return results
