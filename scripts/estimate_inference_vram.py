"""
估算模型推理部署时需要的显存

根据模型配置计算：
1. 模型参数显存
2. KV Cache显存
3. 激活值和其他开销
"""

import argparse


def estimate_model_vram(depth=20, batch_size=1, max_seq_len=1024, dtype_bytes=2):
    """
    估算模型推理时的显存需求

    Args:
        depth: 模型深度（层数）
        batch_size: 推理时的batch size
        max_seq_len: 最大序列长度
        dtype_bytes: 数据类型字节数（2=bfloat16/fp16, 4=fp32）
    """
    # 根据depth计算模型配置
    # 参考 scripts/base_train.py:83-84
    num_layers = depth
    model_dim = depth * 64  # aspect ratio 64
    num_heads = 6  # 默认值
    num_kv_heads = 6  # 默认值，使用MQA时可能不同
    vocab_size = 50304  # 默认值

    print(f"\n{'='*60}")
    print(f"模型配置 (depth={depth}):")
    print(f"{'='*60}")
    print(f"  层数 (n_layer):        {num_layers}")
    print(f"  嵌入维度 (n_embd):     {model_dim}")
    print(f"  注意力头数 (n_head):   {num_heads}")
    print(f"  KV头数 (n_kv_head):    {num_kv_heads}")
    print(f"  词表大小 (vocab_size): {vocab_size}")
    print(f"  序列长度:              {max_seq_len}")

    # 计算参数量
    # 1. Token嵌入: vocab_size * n_embd
    params_embedding = vocab_size * model_dim

    # 2. 每个Transformer块的参数
    # - Attention: Q, K, V, O投影
    #   Q: n_embd * (n_head * head_dim) = n_embd * n_embd
    #   K: n_embd * (n_kv_head * head_dim)
    #   V: n_embd * (n_kv_head * head_dim)
    #   O: n_embd * n_embd
    head_dim = model_dim // num_heads
    params_per_attn = (
        model_dim * model_dim +  # Q
        model_dim * num_kv_heads * head_dim +  # K
        model_dim * num_kv_heads * head_dim +  # V
        model_dim * model_dim  # O
    )

    # - MLP: 两个线性层
    #   fc: n_embd * (4 * n_embd)
    #   proj: (4 * n_embd) * n_embd
    params_per_mlp = model_dim * (4 * model_dim) + (4 * model_dim) * model_dim

    # 每个块的总参数
    params_per_block = params_per_attn + params_per_mlp
    params_blocks = num_layers * params_per_block

    # 3. LM head: n_embd * vocab_size
    params_lm_head = model_dim * vocab_size

    # 总参数量
    total_params = params_embedding + params_blocks + params_lm_head

    print(f"\n{'='*60}")
    print(f"参数量统计:")
    print(f"{'='*60}")
    print(f"  Token嵌入:      {params_embedding:>15,} ({params_embedding/1e6:.1f}M)")
    print(f"  Transformer块:  {params_blocks:>15,} ({params_blocks/1e6:.1f}M)")
    print(f"    - 每块:       {params_per_block:>15,} ({params_per_block/1e6:.1f}M)")
    print(f"    - 注意力:     {params_per_attn:>15,} ({params_per_attn/1e6:.1f}M)")
    print(f"    - MLP:        {params_per_mlp:>15,} ({params_per_mlp/1e6:.1f}M)")
    print(f"  LM Head:        {params_lm_head:>15,} ({params_lm_head/1e6:.1f}M)")
    print(f"  {'─'*60}")
    print(f"  总参数量:       {total_params:>15,} ({total_params/1e6:.1f}M)")

    # 计算显存需求
    print(f"\n{'='*60}")
    print(f"显存需求估算 (数据类型: {'bfloat16' if dtype_bytes == 2 else 'fp32'}):")
    print(f"{'='*60}")

    # 1. 模型参数
    model_params_mem = total_params * dtype_bytes
    print(f"  1. 模型参数:           {model_params_mem / 1e9:.3f} GB")

    # 2. KV Cache
    # KV cache shape: (num_layers, 2, batch_size, num_kv_heads, seq_len, head_dim)
    # 2 代表 K 和 V
    kv_cache_elements = num_layers * 2 * batch_size * num_kv_heads * max_seq_len * head_dim
    kv_cache_mem = kv_cache_elements * dtype_bytes
    print(f"  2. KV Cache:           {kv_cache_mem / 1e9:.3f} GB")
    print(f"     (batch_size={batch_size}, seq_len={max_seq_len})")

    # 3. 激活值和临时缓冲区（估算）
    # 推理时主要是logits和一些中间激活
    # logits: batch_size * seq_len * vocab_size * 4 bytes (通常用fp32)
    logits_mem = batch_size * max_seq_len * vocab_size * 4
    # 其他激活值（估算为模型参数的10%）
    other_activations = model_params_mem * 0.1
    activations_mem = logits_mem + other_activations
    print(f"  3. 激活值/临时缓冲:    {activations_mem / 1e9:.3f} GB")

    # 4. PyTorch开销（估算约500MB）
    pytorch_overhead = 500 * 1024 * 1024
    print(f"  4. PyTorch框架开销:    {pytorch_overhead / 1e9:.3f} GB")

    # 总显存
    total_vram = model_params_mem + kv_cache_mem + activations_mem + pytorch_overhead

    print(f"  {'─'*60}")
    print(f"  总显存需求:            {total_vram / 1e9:.3f} GB")

    # 推荐配置
    print(f"\n{'='*60}")
    print(f"推荐配置:")
    print(f"{'='*60}")
    recommended_vram = total_vram * 1.2  # 留20%余量
    print(f"  推荐显存（含20%余量）: {recommended_vram / 1e9:.1f} GB")

    # 常见GPU建议
    if recommended_vram <= 8 * 1e9:
        print(f"  可用GPU: RTX 3070, RTX 4060 Ti (8GB), RTX 3060 (12GB)")
    elif recommended_vram <= 12 * 1e9:
        print(f"  可用GPU: RTX 3060 (12GB), RTX 4070 (12GB)")
    elif recommended_vram <= 16 * 1e9:
        print(f"  可用GPU: RTX 4080 (16GB), RTX 3090 (24GB)")
    elif recommended_vram <= 24 * 1e9:
        print(f"  可用GPU: RTX 3090 (24GB), RTX 4090 (24GB), A5000 (24GB)")
    elif recommended_vram <= 48 * 1e9:
        print(f"  可用GPU: A6000 (48GB), RTX 6000 Ada (48GB)")
    else:
        print(f"  可用GPU: A100 (40/80GB), H100 (80GB)")

    print(f"\n{'='*60}")
    print(f"不同序列长度下的显存需求:")
    print(f"{'='*60}")
    for seq_len in [512, 1024, 2048, 4096]:
        kv_cache_mem_var = num_layers * 2 * batch_size * num_kv_heads * seq_len * head_dim * dtype_bytes
        total_var = model_params_mem + kv_cache_mem_var + activations_mem + pytorch_overhead
        print(f"  seq_len={seq_len:4d}:  {total_var / 1e9:.2f} GB")

    print(f"\n{'='*60}\n")

    return {
        "total_params": total_params,
        "total_vram_bytes": total_vram,
        "total_vram_gb": total_vram / 1e9,
        "recommended_vram_gb": recommended_vram / 1e9,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="估算nanochat模型推理显存需求")
    parser.add_argument("--depth", type=int, default=20,
                        help="模型深度（默认20，speedrun.sh训练的模型）")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="推理batch size（默认1）")
    parser.add_argument("--max-seq-len", type=int, default=1024,
                        help="最大序列长度（默认1024）")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "fp16", "fp32"],
                        help="推理数据类型（默认bfloat16）")

    args = parser.parse_args()

    dtype_bytes = 2 if args.dtype in ["bfloat16", "fp16"] else 4

    result = estimate_model_vram(
        depth=args.depth,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        dtype_bytes=dtype_bytes
    )
