# Attention vs KDA hybrid, bf16 vs ternary (FineWeb-Edu + selective copy, 1B tokens)

Four runs at equal parameters, data, tokens and recipe: **Qwen3-S** (12 full-attention layers, RoPE) and
**Qwen3-S-Hybrid** (9 Kimi Delta Attention layers + 3 full-attention layers without positional encoding), each in
**bf16** and with **ternary (TWN) weights** trained by QAT. The Qwen3-S pair is the one in
[`initial_baseline.md` → Ternary vs bf16](initial_baseline.md#ternary-vs-bf16-cap512-data-1b-tokens); the hybrid pair
is new. Full log: `~/diary/2026-10-19-hybrid-kda.md`.

Selective-copy (SC) numbers use **training templates, uniform placement** unless stated, as in the baseline write-up.

**Takeaways**
- **The hybrid is better in-distribution.** FineWeb NLL at L = 4k: 3.262 vs 3.295 (bf16) and 3.463 vs 3.529 (ternary),
  and the gap holds at every eval checkpoint. Within T it copies almost perfectly: SC exact ≥ 0.98 for every n_copy
  up to 512 in bf16, vs 0.73–0.91 for Qwen3-S at L = 4k.
- **The hybrid generalizes in length; Qwen3-S does not.** Past T = 4k, Qwen3-S's SC exact match is 0 at every
  length. The hybrid keeps 0.99 / 0.97 (bf16 / ternary) at 8k, 0.94 / 0.88 at 16k and 0.76 / 0.45 at 32k. Its
  FineWeb NLL stays nearly flat to 64k (3.26 → 3.60), while Qwen3-S rises to 4.39.
- **Long-range failures are about spacing, not retrieval.** The hybrid's first-item accuracy is ≥ 0.97 in every
  cell up to 32k (mean ≥ 0.99), in both precisions. What fails is the next item when consecutive words are far apart. Exact match falls with the
  gap L / n_copy: ≈ 0.95 at 256-token spacing, 0.7–0.8 at 1–2k and 0.16 at 4k (n = 8, L = 32k, bf16).
- **Ternary costs the hybrid about as much NLL as Qwen3-S** (+0.20 vs +0.23 nats at 4k), but more at long range:
  SC exact at 32k drops 0.76 → 0.45, and the n = 8 / 32 cells collapse first. Within T, ternary barely matters for
  the hybrid (exact ≥ 0.92 for n up to 512 at 4k, vs 0.52 for ternary Qwen3-S at n = 512).
- **Held-out prompt templates do not follow suit.** At L ≤ 2k, Qwen3-S ternary (0.82–0.90) beats both hybrids
  (0.67–0.71) and Qwen3-S bf16 (0.72–0.77). The hybrid is flat across lengths (0.65–0.71 up to 8k), so its gap is
  prompt transfer, not copying. One seed; not diagnosed.
- **The hybrid is slower per token at this size,** despite 21% fewer nominal FLOPs per token: 55.3k vs 62.6k
  tokens/s per GPU (bf16) and MFU 6.8% vs 8.7%. Likely kernel overhead at this small width (448-dim KDA
  projections, 7 heads); not profiled.

## Runs

| run | dir (`~/runs/longctx/…`) | code | wandb |
|---|---|---|---|
| Qwen3-S bf16 | `qat_pair_1B/bf16` | `longctx-synth` @ `6bd6eeaf` | offline (`wandb/offline-run-20261019_000831-lf1dkn0g`) |
| Qwen3-S ternary | `qat_pair_1B/ternary` | `longctx-synth` @ `6bd6eeaf` | offline (`wandb/offline-run-20261018_231630-mvbg772y`) |
| Hybrid bf16 | `hybrid_pair_1B/bf16` | `hybrid-cache` @ `746aab53` | [longctx/d1qe9avq](https://forge.coreweave.com/wandb/shumingh/longctx/runs/d1qe9avq) |
| Hybrid ternary | `hybrid_pair_1B/ternary` | `hybrid-cache` @ `fd8aaa07` | [longctx/cpbyonao](https://forge.coreweave.com/wandb/shumingh/longctx/runs/cpbyonao) |

The two hybrid commits differ only in eval/metrics code (standalone kernel default, per-GPU tokens/s, MFU), not in
the training path. All are merged into `longctx-synth` (`219b9a15`).

## Model: Qwen3-S-Hybrid

| | |
|---|---|
| Config | `configs/model_configs/qwen/Qwen3-S-hybrid-kda.json`, model type `qwen3_hybrid` (`veomni/models/transformers/qwen3_hybrid/`) |
| Shared with Qwen3-S | hidden 768, 12 layers, SwiGLU FFN 2048, RMSNorm 1e-6, untied embeddings, vocab 151,936, Qwen3 tokenizer, muP init |
| Layer pattern | `[KDA, KDA, KDA, attention] × 3` (`full_attention_interval` 4): layers 3, 7, 11 are full attention |
| Full-attention layers | the Qwen3 attention module (12 q / 4 kv heads, head_dim 64, q/k-norm), **no RoPE (NoPE)** |
| KDA layers | Kimi Delta Attention (Kimi Linear): 7 heads, d_k = d_v = 64. q, k, v = SiLU(depthwise causal conv, width 4); q, k L2-normalized; per-channel forget gate g = −exp(A_log)·softplus(f_proj(x) + dt_bias), with f_proj low-rank (rank 64); β = sigmoid(b_proj(x)); gated delta rule over a [64 × 64] state per head; output o_proj(RMSNorm(o) · sigmoid(g_proj(x))), with g_proj low-rank. Parameter names follow `fla.layers.KimiDeltaAttention`. |
| Parameter match | KDA mixer 1,543,623 params vs attention mixer 1,572,992 (−1.9%; 8 heads would be +11%). Model: **308.6M total / 75.25M non-embedding** vs 308.9M / 75.52M for Qwen3-S |
| KDA init | A_log = log U(1, 16), dt_bias = 0, short convs U(±1/√4); muP for every Linear |
| Kernels | flash-linear-attention **0.5.2** Triton: `chunk_kda` (training, forward + backward, varlen `cu_seqlens` so packed documents never share state), `fused_recurrent_kda` (inference inputs ≤ 64 tokens, forward only), FLA causal conv1d and gated RMSNorm. VeOmni ops `chunk_kda` / `fused_recurrent_kda` (`veomni/ops/kernels/kda/`), with an FP32 token-recurrence `eager` reference for tests. |
| Generation | HF `DynamicCache` built from `layer_types`: a KV cache on the 3 attention layers; on the KDA layers, the recurrent state plus the last 4 inputs of each q/k/v conv. Padded positions have their KDA inputs zeroed, so left padding leaves the state at exactly zero (needed by RULER's `generate`). |

**Ternary (TWN QAT)** is the same recipe as the Qwen3-S ternary run (`veomni/ops/qat/ternary.py`). It applies to
every hidden `nn.Linear`, here **129 matrices, 75.18M parameters**: q/k/v/o, b_proj and the low-rank f/g gate
projections of each KDA layer, plus the attention and MLP projections. Embeddings, lm_head, norms, convs, A_log and
dt_bias stay bf16.

**FLA version.** `uv.lock` pins fla 0.4.1, whose `chunk_kda` backward crashes with an illegal memory access during
autotune (H200, Triton 3.6). The hybrid runs use 0.5.2 from an overlay, with the Triton backend forced (0.5.2's
default TileLang backend does not compile on this host):
`PYTHONPATH=<worktree>:~/envs/fla-0.5.2 FLA_TILELANG=0`. The KDA adapter refuses fla < 0.5.

## Training setup

Identical to the Qwen3-S pair (see [`initial_baseline.md` → Training setup](initial_baseline.md#training-setup) and
the cap512 data of the ternary section) except for the model. Summary:

| | |
|---|---|
| Config | `configs/text/longctx/qwen3_s_hybrid_T4096.yaml` (= `qwen3_s_T4096.yaml` + hybrid model, `kda_implementation: fla`, `no_decay_params: [A_log, dt_bias]`) |
| Data | `mix_fineweb_sc10.yaml`: FineWeb-Edu 90% + selective copy 10% of tokens (`train_T4096_cap512`: n_copy log-uniform 1..512, composed `nl2` templates), T = 4096, documents chunked at T |
| Budget | 1,908 steps × 128 × 4096 = 1.0B tokens; 8× H200, FSDP2, bf16, full activation checkpointing, seed 42 |
| Optimizer | Muon (lr 0.02, spectral √(d_out/d_in), 5-step quintic NS, wd 0) on hidden matrices; AdamW (lr 3e-3, wd 0.1) on embeddings / lm_head / norms / convs / A_log / dt_bias (the last two without decay); 1% warmup, cosine to 0.1× |
| Precision arms | `--model.qat.weight none` (bf16) or `ternary_twn` |
| Logging | wandb online (project `longctx`). Per-layer `grad_norm/layer_XX`, `act_rms/layer_XX`, `act_absmax/layer_XX` every 10 steps (`train.layer_stats`); `tokens_per_second_per_gpu`; `mfu` (hybrid FLOP model added in `b12d5ebd`) |
| Runs | `~/runs/longctx/hybrid_pair_1B/run.sh` (ternary, 02:38–03:27) and `run2.sh` (bf16 05:58–06:41, then all merges and evals). bf16 ran with `FLA_CACHE_RESULTS=0` (fresh in-process autotune); ternary used the default on-disk autotune cache |

### Throughput (median over steps > 50; 8× H200, 0.5M tokens / step)

| run | s / step (wall, incl. ckpt) | tokens/s per GPU | MFU |
|---|---|---|---|
| Qwen3-S bf16 | 1.26 | 62,583 | 0.087 |
| Qwen3-S ternary | 1.26 | 59,198 | 0.082 |
| Hybrid bf16 | 1.36 | 55,284 | 0.068 |
| Hybrid ternary | 1.53 | 47,428 | ≈ 0.058 (not logged; scaled from bf16 by tokens/s) |

Nominal FLOPs per token (fwd + bwd, 4k documents): Qwen3-S 1.61 G, hybrid 1.27 G (quadratic attention term on 3
layers instead of 12; KDA recurrence counted as 15·d_k·d_v·H per token, as for GatedDeltaNet). MFU uses each model's
own FLOP count, so lower MFU at equal tokens/s means fewer FLOPs, not idle hardware. The hybrid ternary run's lower
throughput may partly be the stale autotune cache (bf16 → `FLA_CACHE_RESULTS=0`, not a controlled comparison).

## Evaluation setup

Same as the Qwen3-S ternary section of `initial_baseline.md`. `scripts/longctx/eval_longctx.py` runs one
teacher-forced forward per sequence in bf16, with sdpa for the attention layers and FLA kernels for KDA. Ternary
checkpoints are ternarized at load (`--qat_weight ternary_twn`). Checkpoints are evaluated at steps 500, 1000, 1500
and 1908.

- **Selective copy** `~/datasets/synthetic/selective_copy/test_cap`: L ∈ {256 … 32k} × n_copy ∈ {8, 32, 64, 128,
  256, 512} × placement {uniform, early} × template split {train, test (held-out)}, with 128 fresh instances per
  cell; cells with L < 4n are skipped.
  - Metrics: **exact** (all n words in order), **first-item acc** (pure retrieval of word 1) and **token acc** (per
    word, teacher forced).
- **Positional NLL**: held-out FineWeb-Edu documents truncated to L ∈ {4k, 8k, 16k, 32k, 64k} (256 docs each; 128
  at 64k), mean NLL per position bin and over the whole sequence.

## Results

### FineWeb-Edu held-out NLL

| eval length L | Qwen3-S bf16 | Qwen3-S ternary | Hybrid bf16 | Hybrid ternary |
|---|---|---|---|---|
| 4k (= T) | 3.295 | 3.529 | **3.262** | 3.463 |
| 8k | 3.386 | 3.607 | **3.321** | 3.517 |
| 16k | 3.459 | 3.641 | **3.314** | 3.481 |
| 32k | 3.791 | 3.891 | **3.418** | 3.545 |
| 64k | 4.388 | 4.377 | **3.604** | 3.707 |

Mean over the whole sequence, step 1908. Over training (L = 4k):

| step | Qwen3-S bf16 | Qwen3-S ternary | Hybrid bf16 | Hybrid ternary |
|---|---|---|---|---|
| 500 | 3.787 | 4.081 | 3.753 | 3.977 |
| 1000 | 3.479 | 3.728 | 3.449 | 3.661 |
| 1500 | 3.342 | 3.583 | 3.310 | 3.517 |
| 1908 | 3.295 | 3.529 | 3.262 | 3.463 |

![NLL](fineweb_1B_syn_select_copy_attn_kda_tern_plots/nll.png)

Per position (left), Qwen3-S's NLL turns up right after T and reaches ≈ 4.8 in the last bin (positions 32k–64k).
The hybrid's rises by only ≈ 0.25 over the same range.

### Selective copy vs length

Exact match, mean over n_copy, step 1908:

| | 256 | 512 | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|---|---|
| Qwen3-S bf16 | 0.99 | 0.99 | 0.99 | 0.98 | 0.86 | 0.00 | 0.00 | 0.00 |
| Qwen3-S ternary | 0.98 | 0.98 | 0.97 | 0.91 | 0.65 | 0.00 | 0.00 | 0.00 |
| Hybrid bf16 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | **0.99** | **0.94** | **0.76** |
| Hybrid ternary | 0.99 | 1.00 | 1.00 | 0.98 | 0.98 | **0.97** | **0.88** | **0.45** |

![SC exact by length](fineweb_1B_syn_select_copy_attn_kda_tern_plots/sc_exact_by_len.png)

Hybrid, exact match per cell (bf16 / ternary):

| n_copy | 256 | 512 | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|---|---|
| 8 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 0.99 | 0.98 / 0.96 | 0.79 / 0.73 | 0.16 / 0.05 |
| 32 | 0.99 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 1.00 | 0.99 / 1.00 | 0.98 / 0.90 | 0.70 / 0.13 |
| 64 | 1.00 / 0.97 | 1.00 / 0.99 | 1.00 / 1.00 | 1.00 / 0.99 | 1.00 / 1.00 | 1.00 / 0.99 | 0.98 / 0.97 | 0.89 / 0.39 |
| 128 | – | 1.00 / 0.99 | 1.00 / 1.00 | 1.00 / 0.99 | 0.99 / 0.99 | 0.99 / 0.97 | 0.98 / 0.96 | 0.95 / 0.75 |
| 256 | – | – | 0.98 / 0.98 | 1.00 / 0.98 | 0.99 / 0.98 | 0.98 / 0.98 | 0.99 / 0.91 | 0.95 / 0.79 |
| 512 | – | – | – | 0.99 / 0.94 | 0.98 / 0.92 | 0.98 / 0.89 | 0.95 / 0.84 | 0.93 / 0.56 |

Qwen3-S, same cells within T (bf16 / ternary; 0.00 / 0.00 for every cell at L ≥ 8k):

| n_copy | 256 | 512 | 1k | 2k | 4k |
|---|---|---|---|---|---|
| 8 | 1.00 / 0.99 | 1.00 / 0.98 | 0.99 / 0.98 | 0.98 / 0.94 | 0.89 / 0.73 |
| 32 | 1.00 / 0.98 | 1.00 / 0.98 | 0.98 / 0.98 | 0.98 / 0.96 | 0.91 / 0.66 |
| 64 | 0.98 / 0.98 | 1.00 / 0.98 | 0.99 / 0.99 | 0.99 / 0.90 | 0.88 / 0.74 |
| 128 | – | 0.98 / 0.98 | 0.98 / 0.96 | 0.98 / 0.95 | 0.89 / 0.60 |
| 256 | – | – | 0.99 / 0.95 | 0.98 / 0.90 | 0.84 / 0.67 |
| 512 | – | – | – | 0.95 / 0.80 | 0.73 / 0.52 |

**Where the hybrid fails at long L.** First-item accuracy (pure retrieval) stays ≥ 0.97 in every cell for both
hybrids up to 32k (mean over n ≥ 0.99).
Qwen3-S's is 0.45 / 0.73 (bf16 / ternary) at 8k and ≈ 0 from 16k. The hybrid's errors are in later items, and
they track the spacing between consecutive copied words, L / n_copy. At 32k, bf16 token accuracy is 0.84 at n = 8
(4k tokens apart) and 0.99 at n = 32 (1k apart). Exact match is 0.93–0.95 at n ≥ 128 (≤ 256 tokens apart). At the same
spacing the ternary model drops further, so ternary hurts holding item order across long gaps, not finding items.

![SC first-item accuracy](fineweb_1B_syn_select_copy_attn_kda_tern_plots/sc_first_item_acc_by_len.png)

### Copy capacity

![SC capacity](fineweb_1B_syn_select_copy_attn_kda_tern_plots/sc_capacity.png)

At L = 4k the hybrids keep exact ≥ 0.92 up to n = 512, while Qwen3-S ternary falls to 0.52. At 8k and 16k only the
hybrids copy at all.

### Seen vs held-out prompt templates

| exact, mean over n_copy | 256 | 512 | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|---|---|
| Qwen3-S bf16 | 0.73 | 0.77 | 0.76 | 0.72 | 0.58 | 0.00 | 0.00 | 0.00 |
| Qwen3-S ternary | **0.90** | **0.90** | **0.87** | **0.82** | 0.56 | 0.00 | 0.00 | 0.00 |
| Hybrid bf16 | 0.70 | 0.71 | 0.69 | 0.67 | 0.65 | 0.68 | 0.63 | 0.50 |
| Hybrid ternary | 0.68 | 0.67 | 0.67 | 0.67 | 0.67 | 0.68 | 0.60 | 0.28 |

![SC templates](fineweb_1B_syn_select_copy_attn_kda_tern_plots/sc_templates.png)

On held-out templates every model loses 0.2–0.3 exact match within T. The hybrid's held-out curve is flat in L, so
the loss is prompt transfer, not copying. Qwen3-S ternary is the outlier: it transfers best at short L (the same
effect as in the baseline write-up), not yet diagnosed.

### Over training and training loss

![Learning curves](fineweb_1B_syn_select_copy_attn_kda_tern_plots/learning_curves.png)

![Training loss](fineweb_1B_syn_select_copy_attn_kda_tern_plots/train_loss.png)

The hybrid learns the copy task earlier. Its train SC loss first drops below 1 nat at step 330 (bf16) and 390
(ternary), vs 420 and 590 for Qwen3-S; the curves above are smoothed, so they lag. Past T, hybrid bf16 is already at
0.82 exact by step 1000 (ternary 0.73). Final SC channel loss (mean of the last 10 logs) is 0.0032 / 0.0099
(bf16 / ternary) vs 0.0068 / 0.0244 for Qwen3-S. Final FineWeb channel loss is 3.241 / 3.437 vs 3.273 / 3.503: the
bf16 runs are close, and the ternary gap is smaller for the hybrid.

## Caveats
- One seed per arm and a single 1B-token budget at 75M non-embedding parameters.
- The Qwen3-S pair ran at commit `6bd6eeaf` (offline wandb, no per-layer stats); the hybrid pair at `fd8aaa07` /
  `746aab53`. The training recipe is identical; only the logging differs.
- Throughput: the hybrid ternary run used FLA's on-disk autotune cache (possibly tuned while GPUs were shared); the
  hybrid bf16 run used `FLA_CACHE_RESULTS=0`. The Qwen3-S runs used no FLA kernels.
- Evaluation is teacher-forced: SC exact match asks whether every answer token is the argmax given the correct prefix.
  Generation-based long-context results (RULER) for the hybrids are not run yet (the generation cache is ready).

## Reproduce

```bash
cd ~/VeOmni-longctx    # longctx-synth >= 219b9a15
export PYTHONPATH=$PWD:~/envs/fla-0.5.2 FLA_TILELANG=0 FLA_CACHE_RESULTS=0
# train one hybrid arm (ternary: --model.qat.weight ternary_twn)
torchrun --nproc_per_node 8 tasks/train_text.py configs/text/longctx/qwen3_s_hybrid_T4096.yaml \
  --model.qat.weight none --data.train_size 1000000000 --train.checkpoint.save_steps 500 \
  --train.checkpoint.output_dir ~/runs/longctx/hybrid_pair_1B/bf16 --train.wandb.name hybrid_pair_1B_bf16
# DCP -> HF for intermediate steps, then eval (same for every arm / step)
python scripts/merge_dcp_to_hf.py --load-dir <run>/checkpoints/global_step_<N> --model-assets-dir <run>/model_assets
torchrun --nproc_per_node 8 scripts/longctx/eval_longctx.py --model_path <run>/checkpoints/global_step_<N>/hf_ckpt \
  --tokenizer_path ~/models/Qwen3-0.6B-Base --qat_weight none \
  --sc_test_dir ~/datasets/synthetic/selective_copy/test_cap --nll_dir ~/datasets/fineweb-edu-longctx/nll_eval_qwen3 \
  --out_dir <run>/eval/step<N>
# figures + throughput table
python scripts/longctx/plot_compare_runs.py --out_dir fineweb_1B_syn_select_copy_attn_kda_tern_plots \
  --run "Qwen3-S bf16" ~/runs/longctx/qat_pair_1B/bf16 "#2a78d6" solid o \
  --run "Qwen3-S ternary" ~/runs/longctx/qat_pair_1B/ternary "#2a78d6" dashed o \
  --run "Hybrid bf16" ~/runs/longctx/hybrid_pair_1B/bf16 "#eb6834" solid s \
  --run "Hybrid ternary" ~/runs/longctx/hybrid_pair_1B/ternary "#eb6834" dashed s
```
The whole pair (both arms, merges, evals) is `~/runs/longctx/hybrid_pair_1B/run.sh` followed by `run2.sh`.
