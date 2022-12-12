# P-tuning v2

Natural Language Processing Prompt Tuning Experiment

All run & util file (source code) in https://github.com/THUDM/P-tuning-v2.

I reproduce code Sequence_classification.py.

Experiment 

1. Each layer has different length of prompt.

-- Determine Number of length w.r.t attention score of baseline (prompt sequence 40).

-- Use dataset : Boolq dataset, Superglue.

-- result




# Citation

```console
@article{DBLP:journals/corr/abs-2110-07602,
  author    = {Xiao Liu and
               Kaixuan Ji and
               Yicheng Fu and
               Zhengxiao Du and
               Zhilin Yang and
               Jie Tang},
  title     = {P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally
               Across Scales and Tasks},
  journal   = {CoRR},
  volume    = {abs/2110.07602},
  year      = {2021},
  url       = {https://arxiv.org/abs/2110.07602},
  eprinttype = {arXiv},
  eprint    = {2110.07602},
  timestamp = {Fri, 22 Oct 2021 13:33:09 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2110-07602.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
