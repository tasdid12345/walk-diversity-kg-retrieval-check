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

## Reference

This project is motivated by the following paper:

> M Böckling, H Paulheim, A Iana (2025). *Walk&retrieve: Simple yet effective zero-shot retrieval-augmented generation via knowledge graph walks*. arXiv preprint arXiv:2505.16849. https://arxiv.org/abs/2505.16849

Paper author: Prof. Dr. Heiko Paulheim, Universität Mannheim (heiko.paulheim@uni-mannheim.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
