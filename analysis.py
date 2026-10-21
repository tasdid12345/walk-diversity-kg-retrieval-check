"""Two-hop KG questions: plain random-walk baseline vs diversity-aware walk sampler.
Metric: supporting-fact hit rate (both gold triples appear in the K=20 retrieved walks), higher is better.
Seeds 42/43/44 are used for the randomized samplers; the test set is evaluated once per model."""
import json, os, random, sys, time
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
SEEDS = [42, 43, 44]                      # randomized samplers: run each with these seeds
URL = "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/train.txt"
N_QUESTIONS = 2500                        # sampled two-hop questions
N_TEST = 500                              # fixed held-out test size (20%)
WALKS_PER_NODE = 60                       # walk pool size per start node
K = 20                                    # retrieval budget (walks per question)
WALK_LEN = 3                              # hops per walk


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download(url, retries=5):
    """GET with timeout=60; retry with exponential backoff on 429/5xx and network errors."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            wait = 2 ** attempt
            log(f"download failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("download failed after retries")


def load_triples(text):
    """Parse tab-separated (head, relation, tail) lines."""
    out = []
    for line in text.splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 3:
            out.append(tuple(parts))
    return out


def build_adj(triples):
    """Directed adjacency: head -> list of (relation, tail)."""
    adj = {}
    for h, r, t in triples:
        adj.setdefault(h, []).append((r, t))
    return adj


def sample_questions(adj, rng, n):
    """Sample two-hop paths e1 -r1-> e2 -r2-> e3 from the graph; gold evidence = the two triples."""
    heads = list(adj.keys())
    seen, qs = set(), []
    while len(qs) < n:
        e1 = rng.choice(heads)
        r1, e2 = rng.choice(adj[e1])
        if e2 not in adj:
            continue
        r2, e3 = rng.choice(adj[e2])
        if e3 in (e1, e2):
            continue
        key = (e1, r1, e2, r2, e3)
        if key not in seen:
            seen.add(key)
            qs.append(key)
    return qs


def random_walk(adj, rng, start, length, non_backtracking):
    """Uniform random walk over outgoing edges; returns a tuple of (h, r, t) triples.
    Non-backtracking walks never immediately return to the previous node."""
    walk, node, prev = [], start, None
    for _ in range(length):
        opts = [(r, t) for r, t in adj.get(node, []) if not (non_backtracking and t == prev)]
        if not opts:
            break
        r, t = rng.choice(opts)
        walk.append((node, r, t))
        prev, node = node, t
    return tuple(walk)


def diverse_retrieve(pool):
    """Diversity-aware selection: drop duplicate and empty walks, then greedily pick the walk
    that adds the most unseen triples until the budget K is filled."""
    uniq = list(dict.fromkeys(w for w in pool if w))
    covered, chosen = set(), []
    while uniq and len(chosen) < K:
        best = max(range(len(uniq)), key=lambda i: len(set(uniq[i]) - covered))
        w = uniq.pop(best)
        covered |= set(w)
        chosen.append(w)
    return chosen


def run(method, questions, adj, seed):
    """Evaluate one method with one seed. Only the sampler (non-backtracking flag) and the
    retrieval rule differ between the two methods; data and evaluation are shared."""
    rng = random.Random(seed)
    improved = method == "improved"
    hits, uniq, dist = [], [], []
    for e1, r1, e2, r2, e3 in questions:
        pool = [random_walk(adj, rng, e1, WALK_LEN, improved) for _ in range(WALKS_PER_NODE)]
        ctx = diverse_retrieve(pool) if improved else pool[:K]   # baseline: first K, duplicates kept
        triples = {t for w in ctx for t in w}
        gold = {(e1, r1, e2), (e2, r2, e3)}
        hits.append(gold <= triples)
        uniq.append(len({w for w in ctx if w}))
        dist.append(len(triples))
    return {"hit": float(np.mean(hits)), "uniq_walks": float(np.mean(uniq)), "distinct": float(np.mean(dist))}


def main():
    log("downloading FB15k-237 train.txt")
    triples = load_triples(download(URL))
    log(f"loaded {len(triples)} triples")
    adj = build_adj(triples)
    qs = sample_questions(adj, random.Random(SEED), N_QUESTIONS)
    # Fixed-seed split of the sampled questions, done once.
    idx = list(range(len(qs)))
    random.Random(SEED).shuffle(idx)
    test = [qs[i] for i in idx[:N_TEST]]
    train = [qs[i] for i in idx[N_TEST:]]
    log(f"questions: {len(train)} train pool, {len(test)} test")
    if len(test) < N_TEST:
        log("not enough questions; exiting")
        sys.exit(1)

    res = {}
    for method in ["baseline", "improved"]:
        runs = []
        for s in SEEDS:
            m = run(method, test, adj, s)
            log(f"{method} seed {s}: hit={m['hit']:.3f} uniq_walks={m['uniq_walks']:.2f} distinct={m['distinct']:.1f}")
            runs.append(m)
        res[method] = {k: [r[k] for r in runs] for k in runs[0]}

    b, i = res["baseline"], res["improved"]
    b_mean, i_mean = np.mean(b["hit"]), np.mean(i["hit"])
    b_std, i_std = np.std(b["hit"], ddof=1), np.std(i["hit"], ddof=1)
    if not np.isfinite([b_mean, i_mean]).all():
        sys.exit(1)

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["baseline", "improved"], [b_mean, i_mean], yerr=[b_std, i_std], capsize=5,
           color=["gray", "steelblue"])
    ax.set_ylabel("supporting-fact hit rate")
    ax.set_title("Test hit rate (mean ± std, 3 seeds)")
    fig.tight_layout()
    fig.savefig("figures/hit_rate.png", dpi=120)

    out = {
        "question": "Can a diversity-aware walk sampler that avoids duplicate walks cover the supporting facts "
                    "of two-hop questions better than the paper's plain random-walk sampler under the same "
                    "retrieval budget on a public knowledge graph?",
        "method": "Two-hop KG questions sampled as paths from FB15k-237 train; retrieval of K=20 walks per "
                  "question; supporting-fact hit rate on a fixed 500-question test set; 3 seeds.",
        "dataset": {
            "name": "FB15k-237 (train split)",
            "source_url": URL,
            "n_rows": len(triples),
            "n_train": len(train),
            "n_test": len(test),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "supporting-fact hit rate (both gold triples retrieved)",
        "higher_is_better": True,
        "baseline": "Uniform random-walk sampler, 60 walks per start node, duplicates kept, first K=20 walks used.",
        "improved": "Same walk budget K=20, non-backtracking walks, duplicates removed, and greedy selection "
                    "of walks that add the most unseen triples.",
        "metrics": {
            "baseline_score": float(b_mean),
            "improved_score": float(i_mean),
            "improvement": float(i_mean - b_mean),
            "baseline_score_std": float(b_std),
            "improved_score_std": float(i_std),
            "baseline_unique_walks_per_node": float(np.mean(b["uniq_walks"])),
            "improved_unique_walks_per_node": float(np.mean(i["uniq_walks"])),
            "baseline_distinct_triples": float(np.mean(b["distinct"])),
            "improved_distinct_triples": float(np.mean(i["distinct"])),
        },
        "notes": "Walks are generated from the full train graph, which contains the gold triples, so the hit rate "
                 "measures retrieval coverage rather than generalisation. The train pool of questions is not used "
                 "for fitting because neither method has learned parameters or tuned hyper-parameters. The LLM "
                 "verbalisation step of the paper is omitted. Std uses ddof=1 over 3 seeds. Only 500 test "
                 "questions, so differences of a few points may be noise.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"baseline={b_mean:.3f} improved={i_mean:.3f} -> results.json written")


if __name__ == "__main__":
    main()
