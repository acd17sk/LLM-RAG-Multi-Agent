Superseded runs, kept as evidence (test split):

- `dpo_refusal_v1_collapsed`: refusal pairs without mirrored anti-refusal pairs; the adapter refuses everything.
- `iterative_retrieval_v1_repeats`: iterative retrieval before the loop stopped on repeated follow-up queries.

| variant | R@5 | multi_All@5 | correct | multi_correct | faithful | cited | cites_gold | multi_cites_all | false_refusal | unans_refusal | latency_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dpo_refusal_v1_collapsed | 0.942 ±0.043 | 0.489 ±0.149 | – | – | – | – | 0.0 | 0.0 | 1.0 | 1.0 | 0.5 |
| iterative_retrieval_v1_repeats | 0.952 ±0.043 | 0.468 ±0.149 | 0.812 ±0.05 | 0.585 ±0.112 | 0.892 ±0.035 | 1.0 | 0.933 | 0.404 | 0.007 | 0.059 | 1.45 |
