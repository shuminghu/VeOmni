# Initial baseline: A0 pilot (Qwen3-S, full attention, T=4096, 1B tokens)

First result of the long-context ablation (design and full log: `~/diary/2026-10-18-long-context-synthetic-design.md`).
Selective-copy (SC) numbers here use **training templates only**: fresh instances rendered with the prompt templates
seen in training. That is the primary metric; held-out templates only measure prompt transfer.

**Takeaways**
- SC is learned abruptly: train SC loss stays at ln(2048) ≈ 7.6 (guessing among the content words) until step
  ~310, then falls to 0.9 by step 360. By step 500 SC is solved up to L=2k.
- Within the training length SC keeps improving with training: exact match at L=4k goes 0.35–0.48 (step 500) →
  0.84–0.93 (step 1908), and is ≥0.95 for L ≤ 2k at the end.
- **No length generalization.** Exact match is 0 at L=8k for every checkpoint, first-item accuracy ≤0.57 at 8k and
  ≈0 from 16k. It does not improve with training.
- FineWeb-Edu NLL (positions < T) falls with position and with training: mean 3.80 → 3.30.

## Model
| | |
|---|---|
| Architecture | Qwen3 decoder (`Qwen3ForCausalLM`), all 12 layers full causal attention (no sliding window) |
| Size | hidden 768, 12 layers, 12 query / 4 KV heads (GQA), head_dim 64, SwiGLU FFN 2048, RMSNorm (eps 1e-6) |
| Params | 308.9M total = 75.5M non-embedding + 233.4M embedding and untied lm_head (vocab 151,936) |
| Position encoding | RoPE, θ = 1e6; `max_position_embeddings` 131072 (so evaluation past T runs) |
| Tokenizer | Qwen3 (`~/models/Qwen3-0.6B-Base`) |
| Config | `configs/model_configs/qwen/Qwen3-S-longctx.json`; trained from scratch |

## Training setup
| | |
|---|---|
| Code | worktree `~/VeOmni-longctx`, branch `longctx-synth`, commit `3ea48b45` |
| Launch | `~/runs/longctx/a0_pilot_1B/launch.sh` → `tasks/train_text.py configs/text/longctx/qwen3_s_T4096.yaml --data.train_size 1000000000` (resolved config: `~/runs/longctx/a0_pilot_1B/veomni_cli.yaml`) |
| Hardware | 8× H200, FSDP2 (dp_shard 8), bf16 params / fp32 reduce, flash-attention 2, full activation checkpointing |
| Context | T = 4096 (`max_seq_len`); documents are chunked at T, so no position ≥ T is ever trained |
| Batch | 128 × 4096 tokens = 524,288 tokens/step (packed, dynamic batching), micro-batch 4 |
| Duration | 1,908 steps ≈ 1.0B tokens, 36 min (~1.12 s/step) |
| Optimizer | Muon for hidden matrices (lr 0.02, momentum 0.95 Nesterov, 5-step quintic Newton–Schulz schedule, spectral √(d_out/d_in) scaling, wd 0); AdamW for embeddings/lm_head/norms (lr 3e-3, β 0.9/0.95, wd 0.1); grad clip 1.0 |
| LR schedule | 1% linear warmup, cosine decay to 0.1× |
| Init | muP (embedding N(0,1), hidden spectral 1/√fan_in·min(1,√(fan_out/fan_in)), lm_head N(0,(1/fan_in)²)) |
| Seed | 42 |

### Training data (90/10 by tokens, `configs/text/longctx/mix_fineweb_sc10.yaml` at `3ea48b45`)
- **FineWeb-Edu** sample-10BT train split (`~/datasets/fineweb-edu-longctx/train`, 9.57M docs). Held out:
  md5(id) % 100 < 1 for docs < 16k GPT-2 tokens, < 25 for longer docs.
- **Selective copy**, 10% of tokens (`~/datasets/synthetic/selective_copy/train_T4096`, 4M recipes, about 1.2k
  tokens per sample). Each sample is prompt + body + question + answer. The body is L single-token items: filler
  `" ."` plus n content words drawn from a 2,048-word lexicon. The answer is the n words in order, followed by eos,
  and the loss is on the answer only. n ~ U{1..64}, L log-uniform in [max(64, 4n), T − overhead], words uniformly
  placed, prompt from 8 hand-written `nl` templates (`veomni/data/synthetic/harness.py`).
- In practice SC is 8.5–11.7% of tokens per step but only ~0.35% of loss tokens.
- Caveat: this run used the pre-fix lexicon (`30eac6a39fdd`). The eval lexicon (`12bacc2c1a25`) swaps 5 of the
  2,048 words (contraction stems for 5 other words), which is negligible.

## Evaluation setup
Script: `scripts/longctx/eval_longctx.py` (one teacher-forced forward per sequence, bf16, single sequences with no
packing, sdpa attention), run by `~/runs/longctx/a0_pilot_1B/eval_all.sh`. Outputs:
`~/runs/longctx/a0_pilot_1B/eval/step{500,1000,1500,1908}/results.json`.

**Selective-copy test set** `~/datasets/synthetic/selective_copy/test_nl` (`template_split == train` rows):
- Grid: L ∈ {256, 512, 1k, 2k, 4k, 8k, 16k, 32k} × n ∈ {8, 32, 64} × placement {`uniform` (anywhere in the
  body), `early` (first 25% of the body, i.e. maximum distance)}, 128 instances per cell, filler `" ."`, the same
  8 `nl` train templates.
- Instances are new: seeds ≥ 1e9 (training uses 0..4M), with positions and content pinned in the parquet file.
- At L=4k the full sequence is already slightly longer than T (prompt + question + answer ≈ 30 + n tokens).

Metrics:
- **exact**: all n words correct.
- **first-item acc**: the first answer word. This is pure retrieval; later words get help from teacher forcing,
  because the previous correct word is in the input.
- **token acc**: mean over the n answer words.

**Positional NLL**: held-out FineWeb-Edu docs, 256 docs truncated to exactly 4096 Qwen3 tokens
(`~/datasets/fineweb-edu-longctx/nll_eval_qwen3/L4096.parquet`), mean NLL in log2 position bins.

## Results

#### Exact match (all n items + order correct), placement `uniform`

| step | n_copy | 256 | 512 | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|---|---|---|
| 500 | 8 | 0.98 | 0.96 | 0.97 | 0.86 | 0.46 | 0.00 | 0.00 | 0.00 |
| 500 | 32 | 0.87 | 0.85 | 0.91 | 0.84 | 0.48 | 0.00 | 0.00 | 0.00 |
| 500 | 64 | 0.66 | 0.66 | 0.78 | 0.64 | 0.35 | 0.00 | 0.00 | 0.00 |
| 1000 | 8 | 1.00 | 0.99 | 1.00 | 0.93 | 0.67 | 0.00 | 0.00 | 0.00 |
| 1000 | 32 | 0.98 | 0.99 | 0.98 | 0.94 | 0.80 | 0.00 | 0.00 | 0.00 |
| 1000 | 64 | 0.95 | 0.87 | 0.89 | 0.91 | 0.59 | 0.00 | 0.00 | 0.00 |
| 1500 | 8 | 0.99 | 0.99 | 1.00 | 0.98 | 0.84 | 0.00 | 0.00 | 0.00 |
| 1500 | 32 | 0.98 | 0.95 | 0.99 | 0.98 | 0.85 | 0.00 | 0.00 | 0.00 |
| 1500 | 64 | 0.96 | 0.94 | 0.98 | 0.92 | 0.77 | 0.00 | 0.00 | 0.00 |
| 1908 | 8 | 0.99 | 1.00 | 1.00 | 0.98 | 0.91 | 0.00 | 0.00 | 0.00 |
| 1908 | 32 | 1.00 | 0.98 | 1.00 | 0.99 | 0.93 | 0.00 | 0.00 | 0.00 |
| 1908 | 64 | 0.98 | 0.95 | 0.98 | 0.95 | 0.84 | 0.00 | 0.00 | 0.00 |

#### Exact match (all n items + order correct), placement `early`

| step | n_copy | 256 | 512 | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|---|---|---|
| 500 | 8 | 0.96 | 0.92 | 0.95 | 0.95 | 0.62 | 0.00 | 0.00 | 0.00 |
| 500 | 32 | 0.62 | 0.81 | 0.87 | 0.80 | 0.48 | 0.00 | 0.00 | 0.00 |
| 500 | 64 | 0.00 | 0.41 | 0.50 | 0.45 | 0.04 | 0.00 | 0.00 | 0.00 |
| 1000 | 8 | 0.99 | 0.98 | 0.97 | 0.95 | 0.51 | 0.00 | 0.00 | 0.00 |
| 1000 | 32 | 0.90 | 0.94 | 0.95 | 0.95 | 0.59 | 0.00 | 0.00 | 0.00 |
| 1000 | 64 | 0.02 | 0.75 | 0.88 | 0.88 | 0.21 | 0.00 | 0.00 | 0.00 |
| 1500 | 8 | 0.98 | 1.00 | 0.99 | 0.99 | 0.61 | 0.00 | 0.00 | 0.00 |
| 1500 | 32 | 0.95 | 0.94 | 0.96 | 0.97 | 0.79 | 0.00 | 0.00 | 0.00 |
| 1500 | 64 | 0.05 | 0.71 | 0.93 | 0.91 | 0.52 | 0.00 | 0.00 | 0.00 |
| 1908 | 8 | 0.99 | 1.00 | 0.98 | 0.99 | 0.79 | 0.00 | 0.00 | 0.00 |
| 1908 | 32 | 0.97 | 1.00 | 1.00 | 0.98 | 0.86 | 0.00 | 0.00 | 0.00 |
| 1908 | 64 | 0.09 | 0.84 | 0.97 | 0.95 | 0.69 | 0.00 | 0.00 | 0.00 |

The `early`, L=256, n=64 cell is degenerate: all 64 words fill the first 64 body slots back to back, which never
happens in training (n ≤ L/4).

#### First-item accuracy (pure retrieval, no teacher-forced induction help), placement `uniform`

| step | n_copy | 256 | 512 | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|---|---|---|
| 500 | 8 | 0.99 | 1.00 | 0.99 | 0.98 | 1.00 | 0.08 | 0.00 | 0.00 |
| 500 | 32 | 0.99 | 1.00 | 1.00 | 0.99 | 1.00 | 0.34 | 0.00 | 0.00 |
| 500 | 64 | 1.00 | 0.99 | 1.00 | 1.00 | 0.98 | 0.57 | 0.00 | 0.00 |
| 1000 | 8 | 1.00 | 1.00 | 1.00 | 0.99 | 0.96 | 0.01 | 0.01 | 0.00 |
| 1000 | 32 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.11 | 0.00 | 0.00 |
| 1000 | 64 | 1.00 | 1.00 | 0.99 | 1.00 | 0.99 | 0.16 | 0.00 | 0.00 |
| 1500 | 8 | 0.99 | 1.00 | 1.00 | 1.00 | 0.99 | 0.08 | 0.00 | 0.00 |
| 1500 | 32 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.44 | 0.00 | 0.00 |
| 1500 | 64 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | 0.55 | 0.01 | 0.00 |
| 1908 | 8 | 0.99 | 1.00 | 1.00 | 1.00 | 0.98 | 0.05 | 0.00 | 0.00 |
| 1908 | 32 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.20 | 0.00 | 0.00 |
| 1908 | 64 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.31 | 0.00 | 0.00 |

#### Token accuracy (teacher-forced, per answer item), placement `uniform`

| step | n_copy | 256 | 512 | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|---|---|---|
| 500 | 8 | 1.00 | 1.00 | 1.00 | 0.98 | 0.90 | 0.30 | 0.00 | 0.00 |
| 500 | 32 | 1.00 | 0.99 | 1.00 | 0.99 | 0.98 | 0.54 | 0.03 | 0.00 |
| 500 | 64 | 0.99 | 0.99 | 1.00 | 0.99 | 0.98 | 0.63 | 0.05 | 0.00 |
| 1000 | 8 | 1.00 | 1.00 | 1.00 | 0.99 | 0.94 | 0.36 | 0.02 | 0.00 |
| 1000 | 32 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.55 | 0.05 | 0.00 |
| 1000 | 64 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.65 | 0.05 | 0.00 |
| 1500 | 8 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 0.41 | 0.00 | 0.00 |
| 1500 | 32 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.62 | 0.02 | 0.00 |
| 1500 | 64 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.70 | 0.03 | 0.00 |
| 1908 | 8 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.38 | 0.00 | 0.00 |
| 1908 | 32 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.61 | 0.00 | 0.00 |
| 1908 | 64 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.70 | 0.01 | 0.00 |

#### FineWeb-Edu held-out NLL by position (L=4096 eval set, 256 docs, positions < T)

| step | mean NLL | [16,32) | [32,64) | [64,128) | [128,256) | [256,512) | [512,1024) | [1024,2048) | [2048,4096) |
|---|---|---|---|---|---|---|---|---|---|
| 500 | 3.796 | 4.25 | 3.97 | 3.82 | 3.83 | 3.73 | 3.77 | 3.80 | 3.79 |
| 1000 | 3.483 | 3.91 | 3.64 | 3.51 | 3.51 | 3.42 | 3.46 | 3.49 | 3.48 |
| 1500 | 3.346 | 3.76 | 3.48 | 3.37 | 3.38 | 3.29 | 3.33 | 3.35 | 3.34 |
| 1908 | 3.298 | 3.69 | 3.43 | 3.32 | 3.32 | 3.24 | 3.28 | 3.31 | 3.29 |

### Plots
Training templates, k=1 filler. The dashed line marks T=4096, and the right panel is FineWeb-Edu NLL vs position
(< T).

Exact match, `uniform` placement:
![exact uniform](initial_baseline_plots/sc_exact_uniform.png)

Exact match, `early` placement:
![exact early](initial_baseline_plots/sc_exact_early.png)

First-item accuracy, `uniform`:
![first item uniform](initial_baseline_plots/sc_first_item_acc_uniform.png)

Token accuracy, `uniform`:
![token acc uniform](initial_baseline_plots/sc_token_acc_uniform.png)

Regenerate:
```bash
E=~/runs/longctx/a0_pilot_1B/eval
python scripts/longctx/plot_longctx.py --results $E/step{500,1000,1500,1908}/results.json \
  --labels "step 500" "step 1000" "step 1500" "step 1908" --train_len 4096 \
  --template_split train --metric exact --placement uniform --out initial_baseline_plots/sc_exact_uniform.png
```

## Notes
- Held-out templates (not shown) give lower first-item accuracy that falls with training. Cause: every `nl` train
  prefix has exactly one colon, right before the body, and the model learned "the answer starts after the first
  colon". Fixed for future runs by the composed `nl2` templates (`6d5741f5`).
- The next baseline should be trained on `train_T4096_nl2` and evaluated on `test_nl2` (train split primary).
