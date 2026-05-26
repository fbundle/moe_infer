# Reasonix project memory

Notes the user pinned via the `#` prompt prefix. The whole file is
loaded into the immutable system prefix every session — keep it terse.

- Splice vision features at IMAGE_PAD positions
    pad_idx = np.where(input_ids == IMAGE_PAD)[0]
    assert len(pad_idx) == n_merged, f"{len(pad_idx)} IMAGE_PAD tokens, expected {n_merged}"
    embeds[pad_idx] = vis_feats this means tokenizer.encode will put IMAGE_PAD into the correct place, it won't get affected by engine.embed_lookup?
- Splice vision features at IMAGE_PAD positions
    pad_idx = np.where(input_ids == IMAGE_PAD)[0]
    assert len(pad_idx) == n_merged, f"{len(pad_idx)} IMAGE_PAD tokens, expected {n_merged}"
    embeds[pad_idx] = vis_feats this means tokenizer.encode will put IMAGE_PAD into the correct place, it won't get affected by engine.embed_lookup?
