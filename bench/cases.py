"""MiniMax-H3 实际使用的张量形状定义。

来源：MiniMax-H3/transformer/config.json
  hidden_size = 5376, num_attention_heads = 56, attention_head_dim = 128
  ffn_dim = 14336, rope_freq_dim = 16, rope_theta = 10000
"""

HIDDEN = 5376
HEADS = 56
HEAD_DIM = 128
FFN = 14336
ROPE_DIM = 16
ROPE_THETA = 10000.0
FFN_INNER = FFN * 2  # SwiGLU 的 gate+up 拼接宽度

# 720p / 124 帧的 latent token 数约为 31 * 45 * 80 = 111,600。
# 单卡基准先用可控规模，逐步逼近真实值。
SEQ_LENS = (2048, 8192, 32768)

# 视频 VAE 解码器（36 层 ViT，dim = 2048）
VAE_DIM = 2048
VAE_HEADS = 32
VAE_HEAD_DIM = 64
VAE_SEQ = 4096


def h3_summary():
    return {
        "hidden": HIDDEN,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "ffn": FFN,
        "rope_dim": ROPE_DIM,
        "rope_theta": ROPE_THETA,
        "seq_lens": list(SEQ_LENS),
        "estimated_tokens_720p_124f": 31 * 45 * 80,
    }
