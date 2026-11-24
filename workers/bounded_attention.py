"""Query-chunked SDPA retaining full keys and exact causal positions."""
from contextlib import contextmanager


@contextmanager
def bounded_sdpa(query_chunk_size=128):
    import torch
    import torch.nn.functional as functional
    original = functional.scaled_dot_product_attention

    def forward(query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=None, enable_gqa=False):
        if query.shape[-2] <= query_chunk_size:
            return original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        if dropout_p:
            raise ValueError('bounded attention supports inference dropout=0 only')
        output = torch.empty_like(query)
        for start in range(0, query.shape[-2], query_chunk_size):
            end = min(start + query_chunk_size, query.shape[-2])
            mask = attn_mask
            if mask is not None and mask.shape[-2] != 1:
                mask = mask[..., start:end, :]
            if is_causal:
                allowed = torch.arange(key.shape[-2], device=query.device)[None, :] <= torch.arange(start, end, device=query.device)[:, None]
                if mask is None:
                    mask = allowed
                elif mask.dtype == torch.bool:
                    mask = mask & allowed
                else:
                    mask = mask.masked_fill(~allowed, float('-inf'))
            output[..., start:end, :] = original(query[..., start:end, :], key, value,
                attn_mask=mask, dropout_p=0.0, is_causal=False, scale=scale, enable_gqa=enable_gqa)
        return output

    functional.scaled_dot_product_attention = forward
    try:
        yield
    finally:
        functional.scaled_dot_product_attention = original
