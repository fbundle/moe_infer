"""Classical bandit baselines on the same K-armed Bernoulli envs the LLM bench uses.

Same 200 envs from data/bench_subsets/bandit.jsonl, same T=30 rounds. Each
(algo, env) pair is replayed for M seeds and we report the mean regret over
all (env, seed) pairs.
"""

import json
import math
import random
import statistics
from pathlib import Path

T = 30
M = 20  # seeds per (algo, env)


def play(probs, algo, seed):
    K = len(probs)
    rng = random.Random(seed)
    state = algo.init(K)
    best = max(probs)
    total_regret = 0.0
    for t in range(T):
        a = algo.pick(state, rng, t)
        r = 1 if rng.random() < probs[a] else 0
        algo.update(state, a, r)
        total_regret += best - probs[a]
    return total_regret / T


class Uniform:
    def init(self, K): return {"K": K}
    def pick(self, s, rng, t): return rng.randrange(s["K"])
    def update(self, s, a, r): pass


class EpsilonGreedy:
    def __init__(self, eps): self.eps = eps
    def init(self, K): return {"K": K, "n": [0]*K, "s": [0]*K}
    def pick(self, s, rng, t):
        if rng.random() < self.eps:
            return rng.randrange(s["K"])
        means = [s["s"][i]/s["n"][i] if s["n"][i] else 0.0 for i in range(s["K"])]
        m = max(means)
        return rng.choice([i for i in range(s["K"]) if means[i] == m])
    def update(self, s, a, r):
        s["n"][a] += 1
        s["s"][a] += r


class UCB1:
    def init(self, K): return {"K": K, "n": [0]*K, "s": [0]*K}
    def pick(self, s, rng, t):
        K = s["K"]
        unplayed = [i for i in range(K) if s["n"][i] == 0]
        if unplayed:
            return unplayed[0]
        total = sum(s["n"])
        scores = [s["s"][i]/s["n"][i] + math.sqrt(2*math.log(total)/s["n"][i]) for i in range(K)]
        m = max(scores)
        return rng.choice([i for i in range(K) if scores[i] == m])
    def update(self, s, a, r):
        s["n"][a] += 1
        s["s"][a] += r


class Thompson:
    def init(self, K): return {"K": K, "succ": [0]*K, "fail": [0]*K}
    def pick(self, s, rng, t):
        samples = [rng.betavariate(s["succ"][i]+1, s["fail"][i]+1) for i in range(s["K"])]
        m = max(samples)
        return rng.choice([i for i in range(s["K"]) if samples[i] == m])
    def update(self, s, a, r):
        if r:
            s["succ"][a] += 1
        else:
            s["fail"][a] += 1


def main():
    path = Path(__file__).parent.parent / "data" / "bench_subsets" / "bandit.jsonl"
    examples = [json.loads(l) for l in path.open() if l.strip()]
    print(f"loaded {len(examples)} envs, T={T}, M={M} seeds/env\n")

    algos = [
        ("Uniform random",   Uniform()),
        ("ε-greedy (ε=0.1)", EpsilonGreedy(0.1)),
        ("ε-greedy (ε=0.05)", EpsilonGreedy(0.05)),
        ("UCB1",             UCB1()),
        ("Thompson sampling", Thompson()),
    ]

    for name, algo in algos:
        regrets = []
        for ex_idx, ex in enumerate(examples):
            probs = ex["gold"]["probs"]
            for s in range(M):
                seed = ex_idx * 10_000 + s + 1
                regrets.append(play(probs, algo, seed))
        mean = statistics.mean(regrets)
        stdev = statistics.stdev(regrets)
        sem = stdev / math.sqrt(len(regrets))
        print(f"{name:22s}  mean_regret={mean:.4f}  95% CI ±{1.96*sem:.4f}  (n={len(regrets)})")


if __name__ == "__main__":
    main()
