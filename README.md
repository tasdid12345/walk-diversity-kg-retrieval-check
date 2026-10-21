# Diversity-aware walk sampling for two-hop KG retrieval

Motivation: "Walk&retrieve: Simple yet effective zero-shot retrieval-augmented generation via knowledge graph walks" (2025, https://arxiv.org/abs/2505.16849) shows that verbalised random walks over a knowledge graph can serve as zero-shot retrieval context for LLMs. Its walk corpus is built by plain random walks, which often repeat edges and miss facts. This experiment asks whether a sampler that avoids duplicate walks covers more of the supporting facts under the same budget.

**Question:** Can a diversity-aware walk sampler that avoids duplicate walks cover the supporting facts of two-hop questions better than the paper's plain random-walk sampler under the same retrieval budget on a public knowledge graph?

**Dataset:** FB15k-237 train split, downloaded over HTTP from a public GitHub raw mirror. Two-hop questions are sampled as paths (e1, r1, e2, r2, e3), with the two triples as gold evidence. A fixed-seed split gives a 500-question test set. This is not the paper's own dataset.

**Baseline:** uniform random-walk sampler with 60 walks per start node, duplicates kept, first K=20 walks used as context. This is a stand-in for the paper's walk corpus; the LLM verbalisation step is omitted. The baseline is not the paper's method.

**Improved model:** same K=20 budget, but with non-backtracking walks, duplicates removed, and greedy selection of walks that add unseen triples.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results are written to results.json (and figures/hit_rate.png).

## Results

Results are written to `results.json`.

## Results (latest run)

On FB15k-237 two-hop questions (500 test questions), a standard uniform random-walk sampler with 60 walks per node and duplicates kept reached an 18.9% supporting-fact hit rate, while my diversity-aware sampler reached 23.3%, a modest 4.5-point gain that may be noise.

| Metric | Value |
|---|---|
| baseline_score | 0.1887 |
| improved_score | 0.2333 |
| improvement | 0.04467 |
| baseline_score_std | 0.008083 |
| improved_score_std | 0.01361 |
| baseline_unique_walks_per_node | 17.59 |
| improved_unique_walks_per_node | 19.71 |
| baseline_distinct_triples | 39.12 |
| improved_distinct_triples | 49.35 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Walks are generated from the full train graph, which contains the gold triples, so the hit rate measures retrieval coverage rather than generalisation. The train pool of questions is not used for fitting because neither method has learned parameters or tuned hyper-parameters. The LLM verbalisation step of the paper is omitted. Std uses ddof=1 over 3 seeds. Only 500 test questions, so differences of a few points may be noise.

## Reference

This project is motivated by the following paper:

> M Böckling, H Paulheim, A Iana (2025). *Walk&retrieve: Simple yet effective zero-shot retrieval-augmented generation via knowledge graph walks*. arXiv preprint arXiv:2505.16849. https://arxiv.org/abs/2505.16849

Paper author: Prof. Dr. Heiko Paulheim, Universität Mannheim (heiko.paulheim@uni-mannheim.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
