# On Promoting Individual-Level Fairness in Automated Scoring

Research code accompanying the paper **"On Promoting Individual-Level Fairness in Automated Scoring"** by Michael Fauss, Matthew S. Johnson, and Ikkyu Choi (AIME 2026).

The code computes automated essay scores that satisfy an *individual fairness* constraint: essays that a reference rater scores similarly must also be scored similarly by the machine. Scores are treated as probability distributions over the score scale, and similarity is measured by a statistical distance (Jeffreys divergence, Jensen–Shannon divergence, or total variation distance). The repository contains the solver for the resulting optimization problem and the notebooks that reproduce all results and figures in the paper.

This is research-quality code. It is meant to document and reproduce the paper's results, not to be used as a library.

## Method in brief

Following Dwork et al. (2012), a scorer is individually fair if similar responses receive similar scores. Here, both notions of similarity are measured by the same distance $D$ between score distributions: the distance between two responses is the distance between their score distributions under a trusted reference scorer.

Let $\mathbf{p}_n$ be the reference score distribution of essay $n$ (one of $N$ training essays, scores $1, \ldots, K$) and $\mathbf{q}_n$ the score distribution of the new scorer. The expected squared error between reference score $S$ and machine score $M$ is

$$
\mathbb{E}\bigl[(S - M)^2 \mid w_n\bigr] = \sum_{k=1}^K \sum_{j=1}^K (k-j)^2 \, p_{n,k} \, q_{n,j} = \mathbf{p}_n^\top \mathbf{C} \mathbf{q}_n,
$$

and the fair score distributions on the training set solve

$$
\min_{\mathbf{q}_1,\dots,\mathbf{q}_N} \; \frac{1}{N}\sum_{n=1}^N \mathbf{p}_n^\top \mathbf{C} \mathbf{q}_n
\quad \text{s.t.} \quad D(\mathbf{q}_n, \mathbf{q}_m) \le D(\mathbf{p}_n, \mathbf{p}_m) \;\; \text{for all } n < m,
$$

i.e., the machine may not separate any pair of essays more than the reference does, and among all such score assignments, the one closest to the reference in expected squared error is chosen. Optionally (not used in the paper), the marginal score distribution of the machine can be forced to match that of the reference.

The problem is convex for all three distances. Total variation yields a linear program; Jeffreys and Jensen–Shannon are expressed with exponential cones. `individual_fairness.py` builds the conic program directly in [SCS](https://www.cvxgrp.org/scs/) format, which keeps memory use manageable; the formulation is derived in the module docstring. The Jensen–Shannon divergence is measured in bits, so it lies in $[0, 1]$.

Because the solution is only defined on the training essays, a final step fits an ordinal regression model to the constrained distributions so that fair scores can be produced for unseen essays.

## Pipeline

The experiments use two cumulative-logit ordinal regression models (`mord.LogisticAT`) trained on essay embeddings:

- a **reference model** on 1024-dimensional embeddings, which stands in for the human rater, and
- an **unconstrained model** on 512-dimensional embeddings, the baseline automated scorer without fairness constraints.

The full pipeline is:

| Step | Notebook / file | Input | Output |
|---|---|---|---|
| 1. Fit reference and unconstrained models | `fit_reference_model.ipynb` | `data_train.csv` | `model_reference.npz`, `model_unconstrained.npz` |
| 2. Solve the fairness-constrained problem on the training set | `solve_optimization.ipynb` (calls `individual_fairness.py`) | `P_reference_train.npz` | `P_constrained_{tv,jf,js}.npz` |
| 3. Fit a fair model to the constrained distributions | `extrapolate_fair_model.ipynb` | `data_train.csv`, `P_constrained_*.npz`, `model_unconstrained.npz` | `model_constrained_{tv,jf,js}.npz` |
| 4. Evaluate on the test set; reproduce tables and figures | `results.ipynb` | `data_test.csv`, all `model_*.npz` | Table 1, Figures 1–3, constraint-violation statistics |

Step 2 is run once per distance (change the `distance` argument), and step 3 once per resulting `P_constrained_*.npz` file. In step 3, the fair model is initialized at the unconstrained model's parameters and fitted by minimizing the squared error between its predicted distributions and the constrained ones with L-BFGS-B.

`P_reference_train.npz` contains the reference model's predicted score distributions on the training essays (`model_reference.predict_proba(...)` on the 1024-dimensional training embeddings).

**All intermediate results are included in `data/`**, so `results.ipynb` can be run directly without repeating steps 1–3. Steps 2 and 3 are computationally expensive (see below).

## Repository structure

```
.
├── individual_fairness.py          # SCS-based solver for the constrained problem + distance helpers
├── fit_reference_model.ipynb       # Step 1: reference and unconstrained ordinal regression models
├── solve_optimization.ipynb        # Step 2: solve the fairness-constrained problem
├── extrapolate_fair_model.ipynb    # Step 3: fit a fair model to the constrained distributions
├── results.ipynb                   # Step 4: all results and figures in the paper
└── data/
    ├── data_train.csv              # Training essays (1,498)
    ├── data_test.csv               # Test essays (659)
    ├── P_reference_train.npz       # Reference score distributions on the training set
    ├── P_constrained_tv.npz        # Constrained distributions, total variation
    ├── P_constrained_jf.npz        # Constrained distributions, Jeffreys
    ├── P_constrained_js.npz        # Constrained distributions, Jensen–Shannon
    ├── model_reference.npz         # Reference model parameters (theta, coef)
    ├── model_unconstrained.npz     # Unconstrained model parameters
    └── model_constrained_{tv,jf,js}.npz  # Fair model parameters per distance
```

The CSV files contain the columns `full_text`, `holistic_essay_score` (scores 1–6), `embeddings_1024`, and `embeddings_512`, with embeddings stored as string-encoded lists. The essays are responses to the "Distance Learning" prompt of the [PERSUADE 2.0](https://github.com/scrosseye/persuade_corpus_2.0) corpus; the embeddings are Titan text embeddings at two sizes.

Model files store the `LogisticAT` thresholds (`theta`) and coefficients (`coef`). Score distribution files store a single `(N, K)` array under the default key `arr_0`. The suffixes `tv`, `jf`, and `js` stand for total variation, Jeffreys, and Jensen–Shannon.

## Installation

Python ≥ 3.10 is required (the notebooks were run with Python 3.14). NumPy ≥ 2.0 is needed for `np.concat`.

```bash
pip install numpy scipy pandas matplotlib scs mord jupyter
```

## Using the solver

```python
import numpy as np
from individual_fairness import get_fair_score_distributions

P_reference = np.load("data/P_reference_train.npz")["arr_0"]   # shape (N, K)

P_fair, mse = get_fair_score_distributions(
    P_reference,
    distance="total-variation",   # or "Jeffreys", "Jensen-Shannon"
    match_score_dist=False,       # True: also match the marginal score distribution
    verbose=True,
)
```

The function returns the constrained distributions (shape `(N, K)`) and the attained mean squared error. SCS tolerances and the iteration limit can be passed via `eps_abs`, `eps_rel`, `eps_infeas`, and `max_iters`. The first argument is called `P_human` in the code; in the experiments, the reference model's distributions take that role.

### Computational cost

The problem has one constraint per pair of essays, so its size grows quadratically in $N$. With the 1,498 training essays there are roughly 1.1 million pairwise constraints, and the Jeffreys and Jensen–Shannon variants add several exponential cones per pair and score level. Expect long run times and substantial memory use for step 2 (the total variation LP is the cheapest). Step 3 also takes a while, since L-BFGS-B uses finite-difference gradients over 517 parameters.

For quick experiments, use a random subset of the essays. Running `python individual_fairness.py` solves a small random example (N = 50) for all three distances and reports the run time, attained MSE, and largest constraint violation.

The solver clips small negative values returned by SCS to zero. Entries that are exactly zero make the Jeffreys divergence between some pairs infinite when evaluated on the raw solution; this does not affect the fitted fair models, whose predicted distributions are strictly positive.

## Citation

```bibtex
@inproceedings{fauss2026promoting,
  title     = {On Promoting Individual-Level Fairness in Automated Scoring},
  author    = {Fauss, Michael and Johnson, Matthew S. and Choi, Ikkyu},
  booktitle = {[AIME 2026 PROCEEDINGS]},
  year      = {2026}
}
```
