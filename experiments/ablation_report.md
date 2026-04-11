# Ablation Study Results

## Overall Ranking (by final train loss)

| Rank | Experiment | Final Train Loss | Train PPL | Throughput |
|------|-----------|--------------|---------|------------|
| 1 | base | 3.0870 | 21.91 | 56,855 |
| 2 | lr_1e3 | 3.1565 | 23.49 | 430,307 |
| 3 | lr_6e4 | 3.1879 | 24.24 | 425,904 |
| 4 | pos_alibi | 3.2380 | 25.48 | 306,989 |
| 5 | warmup_1000 | 3.2645 | 26.17 | 430,293 |
| 6 | norm_pre_layernorm | 3.2659 | 26.20 | 423,116 |
| 7 | pos_rope | 3.2660 | 26.21 | 428,814 |
| 8 | norm_pre_rmsnorm | 3.2668 | 26.23 | 428,509 |
| 9 | attn_mha | 3.2670 | 26.23 | 431,290 |
| 10 | lr_3e4 | 3.2670 | 26.23 | 430,106 |
| 11 | act_swiglu | 3.2671 | 26.24 | 429,781 |
| 12 | warmup_2000 | 3.2681 | 26.26 | 431,477 |
| 13 | warmup_500 | 3.2682 | 26.26 | 430,864 |
| 14 | warmup_100 | 3.2725 | 26.38 | 430,505 |
| 15 | attn_gqa4 | 3.2985 | 27.07 | 439,999 |
| 16 | pos_learned | 3.3001 | 27.12 | 457,796 |
| 17 | norm_post_layernorm | 3.3043 | 27.23 | 422,560 |
| 18 | norm_post_rmsnorm | 3.3045 | 27.23 | 423,272 |
| 19 | act_gelu | 3.3068 | 27.30 | 476,751 |
| 20 | attn_gqa2 | 3.3109 | 27.41 | 447,290 |
| 21 | attn_mqa | 3.3113 | 27.42 | 441,266 |
| 22 | lr_1e4 | 3.4964 | 33.00 | 429,854 |
| 23 | test_run | 9.1398 | 9318.90 | 780 |

## Suite Winners

- **Positional Encoding**: `ALiBi` (loss: 3.2380)
- **Attention Mechanism**: `MHA (12 KV heads)` (loss: 3.2670)
- **Activation Function**: `SwiGLU` (loss: 3.2671)
- **Normalization**: `Pre-norm + LayerNorm` (loss: 3.2659)
- **Warmup Steps**: `1000 steps` (loss: 3.2645)
- **Learning Rate**: `1e-3` (loss: 3.1565)

## Recommended Best Stack

Combining the winner from each independent ablation:

- Positional Encoding: **ALiBi**
- Attention Mechanism: **MHA (12 KV heads)**
- Activation Function: **SwiGLU**
- Normalization: **Pre-norm + LayerNorm**
- Warmup Steps: **1000 steps**
- Learning Rate: **1e-3**