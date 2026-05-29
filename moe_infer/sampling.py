"""Token sampling strategies — softmax, top-k, top-p, min-p."""

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax.

    Parameters
    ----------
    x : np.ndarray
        Input logits, 1-D.

    Returns
    -------
    np.ndarray
        Probability distribution, same shape.
    """
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def sample(
    logits: np.ndarray,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
) -> int:
    """Sample a single token id from logits.

    The caller's array is not modified — a working copy is taken for any
    temperature/filter step that would scale or zero entries.
    Zero-temperature returns ``argmax`` (greedy) without copying.

    Parameters
    ----------
    logits : np.ndarray
        1-D float array of unnormalised scores, shape ``[vocab_size]``.
    temperature : float
        Scaling factor.  0.0 = greedy.  Default 0.0.
    top_k : int
        Keep only the *k* highest-probability tokens.  0 = disabled.
    top_p : float
        Nucleus sampling: keep the smallest set of tokens whose
        cumulative probability ≥ *top_p*.  1.0 = disabled.
    min_p : float
        Drop tokens with probability < ``max_prob * min_p``.
        0.0 = disabled.

    Returns
    -------
    int
        Sampled token id.
    """
    if temperature < 0.01:
        return int(np.argmax(logits))
    n = len(logits)
    work = logits.astype(np.float32, copy=True)
    if abs(temperature - 1.0) > 1e-7:
        work /= max(temperature, 1e-8)
    probs = softmax(work)

    if top_k > 0 and top_k < n:
        indices = np.argpartition(probs, -top_k)[-top_k:]
        mask = np.ones(n, dtype=bool)
        mask[indices] = False
        probs[mask] = 0.0

    if top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_idx])
        cutoff_idx = np.searchsorted(cumsum, top_p)
        if cutoff_idx < n:
            probs[sorted_idx[cutoff_idx + 1 :]] = 0.0

    if min_p > 0.0:
        threshold = probs.max() * min_p
        probs[probs < threshold] = 0.0

    total = probs.sum()
    if total <= 0:
        return int(np.argmax(work))
    probs /= total
    return int(np.random.choice(n, p=probs))
