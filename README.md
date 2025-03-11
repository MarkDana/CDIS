# When Selection Meets Intervention: Additional Complexities in Causal Discovery

[Paper](https://arxiv.org/abs/2403.15500) by [Haoyue Dai](https://hyda.cc), [Ignavier Ng](https://ignavierng.github.io/), [Jianle Sun](https://sjl-sjtu.github.io), [Zeyu Tang](https://zeyu.one), [Gongxu Luo](https://scholar.google.com/citations?hl=zh-CN&user=1qoKnMQAAAAJ&view_op=list_works&sortby=pubdate), [Xinshuai Dong](https://dongxinshuai.github.io), [Peter Spirtes](https://www.cmu.edu/dietrich/philosophy/people/faculty/spirtes.html), [Kun Zhang](https://www.andrew.cmu.edu/user/kunz1/index.html). Appears at ICLR 2025 (oral).

*In experimental studies, subjects are usually selectively enrolled–for example, participants in a drug trial are typically already patients of the relevant disease. What causal relations can we reliably identify from such biased interventional data, and how? The answer might be more complex than it seems at first glance.*

This repository contains the code for the **CDIS** (Causal Discovery from Interventional data under potential Selection bias) algorithm. CDIS identifies causal relations and selection structures up to the **Markov equivalence class** from interventional data with **soft interventions, unknown targets, and potential selection bias.**

---




## 1. Getting Started

Run the following command to reproduce several provided examples:

```bash
python main.py
```

The main section of `main.py` shows example usages of the code. The expected output should look like:

<details>
  <summary>Click to expand</summary>

```text
In what follows we show several examples of <DAG, intervention, selection> configurations,
    and see what causal realtions and selection mechanisms can be identified, from both oracle setting and real data.
    One may check if the results from oracle and real data are consistent.

Eg1, I have two originally independent variables, 0 and 1, selected; if we only intervene on 0:
  [oracle] pure observational PAG: {'->': set(), '<->': set(), '--': set(), '⚬--': set(), '⚬->': set(), '⚬-⚬': {(0, 1), (1, 0)}}
  [oracle] final PAG: {'->': set(), '<->': set(), '--': set(), '⚬--': set(), '⚬->': set(), '⚬-⚬': {(0, 1), (1, 0)}}
  [data] pure observational PAG: {'->': set(), '<->': set(), '--': set(), '⚬--': set(), '⚬->': set(), '⚬-⚬': {(0, 1), (1, 0)}}
  [data] final PAG: {'->': set(), '<->': set(), '--': set(), '⚬--': set(), '⚬->': set(), '⚬-⚬': {(0, 1), (1, 0)}}
=> The relation between 0 and 1 is not identifiable; e.g., 0->S<-1 and 1->0 are both possible to produce the data.

Eg2, I have two originally independent variables, 0 and 1, selected; if we have two datasets intervening on 0 and 1 respectively:
  [oracle] pure observational PAG: {'->': set(), '<->': set(), '--': set(), '⚬--': set(), '⚬->': set(), '⚬-⚬': {(0, 1), (1, 0)}}
  [oracle] final PAG: {'->': set(), '<->': set(), '--': {(0, 1), (1, 0)}, '⚬--': set(), '⚬->': set(), '⚬-⚬': set()}
  [data] pure observational PAG: {'->': set(), '<->': set(), '--': set(), '⚬--': set(), '⚬->': set(), '⚬-⚬': {(0, 1), (1, 0)}}
  [data] final PAG: {'->': set(), '<->': set(), '--': {(0, 1), (1, 0)}, '⚬--': set(), '⚬->': set(), '⚬-⚬': set()}
=> This time, we can be sure about the existence of selection bias on 0 and 1.

Eg3, I have 0->1->2 without selection, if we intervene on 0:
  [oracle] pure observational PAG: {'->': set(), '<->': set(), '--': set(), '⚬--': set(), '⚬->': set(), '⚬-⚬': {(0, 1), (1, 0), (1, 2), (2, 1)}}
  [oracle] final PAG: {'->': {(0, 1), (1, 2)}, '<->': set(), '--': set(), '⚬--': set(), '⚬->': set(), '⚬-⚬': set()}
  [data] pure observational PAG: {'->': set(), '<->': set(), '--': set(), '⚬--': set(), '⚬->': set(), '⚬-⚬': {(0, 1), (1, 0), (1, 2), (2, 1)}}
  [data] final PAG: {'->': {(0, 1), (1, 2)}, '<->': set(), '--': set(), '⚬--': set(), '⚬->': set(), '⚬-⚬': set()}
=> Without selection (though we dont know apriori), we can identify the causal relations using only one intervention.
```
</details>




## 2. Running CDIS on Your Own Data

Prepare your datasets as a `data_list`:

 + A list of `np.ndarray`, each in shape `(n_samples, n_vars)`. 
 + The first dataset should be purely observational (control group); the rests are datsets from various interventions (treatment groups).
 + Interventions can be soft and with unknown targets.
 + `n_vars` should be the same across datasets, i.e., they measure the same set of variables.
 + `n_samples` can vary across datasets, and sample individuals do not need to be matched across datasets.
 + We do assume that samples from each dataset, before receiving their interventions, come from a same distribution (as the control group).
 + We also assume causal sufficiency among `n_vars` variables, and faithfulness.



Then, in `main.py`, run:

```python
results = cdis_from_data(data_list)
final_PAG = results['final_PAG']
pure_observational_PAG = results['pure_observational_PAG']
```

where

+ `final_PAG`, a partial ancestral graph (PAG) over `n_vars` variables, is the output of CDIS. In it,
  - Each `i->j` edge means that `i` is the direct cause of `j`, and `j` is not involved in any selection mechanisms.
  - Each `i--j` edge means that `i` and `j` together are involved in a selection, either directly applied on them or through their common child.
  - The `○` endmarks on other edges mean that this endmark on the edge cannot be identified–both `-` and `>` are possible.
+ `pure_observational_PAG` is the PAG obtained by FCI on observational data only. This is just for record and comparison.





---

## Citation

If you use this code for your research, please cite our paper:

```bibtex
@inproceedings{dai2025when,
  title={When Selection Meets Intervention: Additional Complexities in Causal Discovery},
  author={Haoyue Dai and Ignavier Ng and Jianle Sun and Zeyu Tang and Gongxu Luo and Xinshuai Dong and Peter Spirtes and Kun Zhang},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=gFR4QwK53h}
}
```
