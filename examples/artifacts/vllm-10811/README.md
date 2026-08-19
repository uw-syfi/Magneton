# vLLM Issue 9471

## Notice

To reproduce the issue, you should modify the `vllm/attention/backends/flash_attn.py`:
```python
output = flash_attn_varlen_func( # line 817
    q=query,
    k=key,
    v=value,
    cu_seqlens_q=q_seq_start_loc,
    cu_seqlens_k=k_seq_start_loc,
    max_seqlen_q=q_seq_len,
    max_seqlen_k=k_seq_len,
    softmax_scale=softmax_scale,
    causal=_get_causal_option(attn_type),
    window_size=window_size,
    alibi_slopes=alibi_slopes,
    softcap=logits_soft_cap,
    out=None,
    fa_version=self.vllm_flash_attn_version,
    q_descale=layer._q_scale.expand(descale_shape),
    k_descale=layer._k_scale.expand(descale_shape),
    v_descale=layer._v_scale.expand(descale_shape),
)
prefill_output.copy_(output)
```

## Evaluation

The original version has operators in a layer:
1. `rms_norm`: latency 7.3us, power 149W
2. `kqv_proj`: latency 13.5us, power 682W
3. `flash_attn_prefill`: latency 7.8us + 1.8us + 25.2us + 3.1us + 2.2us = 40.1us, power 327W
4. `o_proj`: latency 6.5us, power 241W
5. `rms_norm`: latency 7.3us, power 149W
6. `up_proj`: latency 17us, power 686W
7. `gelu`: latency 22us, power 381W
8. `down_proj`: latency 16.2us, power 686W
The total latency is 129.9us, and the energy consumption is 57.2mJ.

In the fixed version, the memory copy kernel in `flash_attn_prefill` is removed. Therefore, the total latency becomes 127.7us, and the energy consumption is 56.4mJ.
