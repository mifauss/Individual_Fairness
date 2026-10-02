"""
individual_fairness.py
======================
Score distributions for automated scoring under individual fairness constraints.

Given reference score distributions p_1, ..., p_N on the score scale
{0, ..., K-1} (e.g., from human raters or a trusted reference model), this
module computes machine score distributions q_1, ..., q_N that solve

    min   (1/N) sum_n  E[(S_n - T_n)^2],   S_n ~ p_n,  T_n ~ q_n  independent
    s.t.  D(q_n, q_m) <= D(p_n, p_m)        for all pairs n < m
          q_n in the probability simplex     for all n

i.e., the machine may not separate any pair of responses more than the
reference does, and among all such score assignments, the one closest to the
reference in expected squared error is selected.  Optionally, the marginal
score distribution of the machine can be constrained to match that of the
reference:  sum_n q_n = sum_n p_n.

Supported distances D:

    Jeffreys          D(p, q) = KL(p || q) + KL(q || p)                 [nats]
    Jensen-Shannon    D(p, q) = [KL(p || m) + KL(q || m)] / 2,
                      m = (p + q) / 2                                   [bits]
    Total variation   D(p, q) = ||p - q||_1 / 2

The Jensen-Shannon divergence uses binary logarithms, so that it lies in [0, 1].

All three problems are convex.  They are formulated directly as conic
programs and solved with SCS (https://www.cvxgrp.org/scs/).  Building the
sparse constraint matrix by hand avoids the overhead of a modeling layer,
which matters because the number of constraints grows quadratically in N.

Dependencies: numpy, scipy, scs, matplotlib (plotting helper only)

──────────────────────────────────────────────────────────────────────────────
OBJECTIVE
──────────────────────────────────────────────────────────────────────────────

With scores s_k = k and mu_n = sum_k p[n,k] s_k,

    E[(S_n - T_n)^2] = sum_k q[n,k] (s_k^2 - 2 mu_n s_k)  +  sum_k p[n,k] s_k^2.

The first term is linear in q; the second is a constant that is added back
after solving.  The objective is invariant to shifting the score scale, so
labeling scores 0..K-1 instead of 1..K does not change the solution.

──────────────────────────────────────────────────────────────────────────────
CONIC FORM
──────────────────────────────────────────────────────────────────────────────

SCS solves

    min   c' x
    s.t.  A x + s = b,   s in K,

where K is a product of cones in the order: zero cone (equalities),
non-negative cone, exponential cone.  SCS defines the exponential cone as

    K_exp = closure{ (x, y, z) : y > 0,  y exp(x / y) <= z }.

Divergence terms are written in terms of the elementwise function

    kl_div(a, b) := a log(a / b) - a + b,

which sums to KL(p || q) over k for probability vectors p, q (the affine
terms cancel).  Its epigraph has an exponential cone representation:

    kl_div(a, b) <= t   <=>   (b - t,  a,  e b) in K_exp.

Proof:  a exp((b - t) / a) <= e b
        <=>  (b - t) / a - 1 <= log(b / a)
        <=>  a log(a / b) - a + b <= t.

The constraints use natural logarithms throughout; Jensen-Shannon bounds
given in bits are converted to nats by multiplying with log(2).

Notation:  P = N (N - 1) / 2 is the number of unique pairs; pair index pi
corresponds to (n, m) with n < m; p[n,k] below denotes the decision
variable (machine distribution), and d_nm = D(P_ref[n], P_ref[m]) is the
fairness bound computed from the reference distributions.

──────────────────────────────────────────────────────────────────────────────
JEFFREYS
──────────────────────────────────────────────────────────────────────────────

Variables:  x = [ p (N K) | t (P K) | u (P K) | slack (P) ]

Zero cone:
    sum_k p[n,k] = 1                                     n = 0..N-1
    sum_k (t[pi,k] + u[pi,k]) + slack[pi] = d_nm         pi = 0..P-1
    [optional] sum_n p[n,k] = sum_n P_ref[n,k]           k = 0..K-1
Non-negative cone:
    p[n,k] >= 0,   slack[pi] >= 0
Exponential cone (two triples per pair and score):
    kl_div(p[n,k], p[m,k]) <= t[pi,k]:  (p[m,k] - t[pi,k],  p[n,k],  e p[m,k])
    kl_div(p[m,k], p[n,k]) <= u[pi,k]:  (p[n,k] - u[pi,k],  p[m,k],  e p[n,k])

──────────────────────────────────────────────────────────────────────────────
JENSEN-SHANNON
──────────────────────────────────────────────────────────────────────────────

Variables:  x = [ p (N K) | mid (P K) | t (P K) | u (P K) | slack (P) ]

Zero cone:
    sum_k p[n,k] = 1
    2 mid[pi,k] - p[n,k] - p[m,k] = 0                    (midpoint)
    sum_k (t[pi,k] + u[pi,k]) + slack[pi] = 2 log(2) d_nm
    [optional] marginal matching as above
Non-negative cone:
    p[n,k] >= 0,   slack[pi] >= 0
Exponential cone:
    kl_div(p[n,k], mid[pi,k]) <= t[pi,k]:  (mid - t,  p[n,k],  e mid)
    kl_div(p[m,k], mid[pi,k]) <= u[pi,k]:  (mid - u,  p[m,k],  e mid)

The factor 2 accounts for the 1/2 in the definition of the divergence, the
factor log(2) for the conversion from bits to nats.

──────────────────────────────────────────────────────────────────────────────
TOTAL VARIATION
──────────────────────────────────────────────────────────────────────────────

A linear program.  The absolute values are linearized with auxiliary
variables r[pi,k] >= |p[n,k] - p[m,k]|.

Variables:  x = [ p (N K) | r (P K) | slack (P) ]

Zero cone:
    sum_k p[n,k] = 1
    sum_k r[pi,k] + slack[pi] = 2 d_nm
    [optional] marginal matching as above
Non-negative cone:
    p[n,k] >= 0,   r[pi,k] >= 0,   slack[pi] >= 0
    r[pi,k] - p[n,k] + p[m,k] >= 0
    r[pi,k] - p[m,k] + p[n,k] >= 0
"""

import warnings
from typing import Tuple

import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt
from scipy.special import rel_entr
from scipy.stats import binom


def _scs_zero_cone_key() -> str:
    """
    Returns the correct zero-cone key for the installed SCS version.
    SCS uses 'z' for the zero (equality) cone. This helper exists for
    forward-compatibility in case the key ever changes.
    """
    return "z"


# ── Distance helpers (numpy) ───────────────────────────────────────────────────

def jeffreys_dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(rel_entr(p, q)) + np.sum(rel_entr(q, p)))


def jensen_shannon_dist(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits (binary logarithm), in [0, 1]."""
    m = (p + q) / 2
    return float((np.sum(rel_entr(p, m)) + np.sum(rel_entr(q, m))) / (2 * np.log(2)))


def total_variation_dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p - q, 1) / 2)


# ── Main solver ────────────────────────────────────────────────────────────────

def get_fair_score_distributions(
    P_human: np.ndarray,
    distance: str = "Jeffreys",
    match_score_dist: bool = False,
    verbose: bool = False,
    eps_abs: float = 5e-5,
    eps_rel: float = 5e-5,
    eps_infeas: float = 1e-5,
    max_iters: int = 100_000,
) -> Tuple[np.ndarray | None, float | None]:
    """
    Compute MSE-optimal machine score distributions subject to individual
    fairness constraints (see the module docstring for the problem).

    Parameters
    ----------
    P_human          : ndarray of shape (N, K), reference score distributions
                       (from human raters or a trusted reference model)
    distance         : "Jeffreys", "Jensen-Shannon" (in bits), or
                       "Total-Variation" (also "TV"); case-insensitive
    match_score_dist : if True, enforce that the marginal score distribution of
                       P_machine matches that of P_human, i.e.:
                           sum_n P_machine[n, k] = sum_n P_human[n, k]  for all k
                       This adds K linear equality constraints to the problem.
    verbose          : if True, print SCS solver output
    eps_abs          : SCS absolute convergence tolerance
    eps_rel          : SCS relative convergence tolerance
    eps_infeas       : SCS infeasibility tolerance
    max_iters        : SCS maximum iterations

    Returns
    -------
    P_machine : ndarray of shape (N, K), or None if the distance is not supported
    min_mse   : float, attained mean squared error, or None
    """
    try:
        import scs  # type: ignore
    except ImportError:
        raise ImportError(
            "The 'scs' package is required for this solver.  Install with:\n"
            "    pip install scs"
        )

    if distance.lower() == "jeffreys":
        dist_func = jeffreys_dist
    elif distance.lower() == "jensen-shannon":
        dist_func = jensen_shannon_dist
    elif distance.lower() in ("total-variation", "tv"):
        dist_func = total_variation_dist
    else:
        warnings.warn(f"Distance '{distance}' not supported.")
        return None, None

    if distance.lower() == "jensen-shannon":
        return _solve_jsd(P_human, dist_func, match_score_dist, verbose, eps_abs, eps_rel, eps_infeas, max_iters)
    elif distance.lower() in ("total-variation", "tv"):
        return _solve_tv(P_human, dist_func, match_score_dist, verbose, eps_abs, eps_rel, eps_infeas, max_iters)
    else:
        return _solve_jeffreys(P_human, dist_func, match_score_dist, verbose, eps_abs, eps_rel, eps_infeas, max_iters)



def _add_marginal_constraints(
    I, J, V, b_list, row: int,
    P_human: np.ndarray,
    p_idx,
) -> int:
    """
    Append K zero-cone (equality) rows enforcing that the column sums of
    P_machine match those of P_human:

        sum_n P_machine[n, k] = sum_n P_human[n, k]   for k = 0..K-1

    Returns the updated row counter.
    """
    N, K = P_human.shape
    col_sums = P_human.sum(axis=0)   # (K,) — target marginal counts
    for k in range(K):
        for n in range(N):
            I.append(row); J.append(p_idx(n, k)); V.append(1.0)
        b_list.append(float(col_sums[k]))
        row += 1
    return row


# ── Jeffreys solver ─────────────────────────────────────────────────────────────

def _solve_jeffreys(P_human, dist_func, match_score_dist, verbose, eps_abs, eps_rel, eps_infeas, max_iters):
    import scs

    N, K = P_human.shape
    scores = np.arange(K, dtype=float)
    sq = scores ** 2
    m1 = P_human @ scores   # (N,) — mean human score per sample

    pairs = [(n, m) for n in range(N) for m in range(n + 1, N)]
    P_count = len(pairs)

    NK = N * K
    n_t = P_count * K       # aux vars for kl_div(p_n, p_m) elementwise
    n_u = P_count * K       # aux vars for kl_div(p_m, p_n) elementwise
    n_slack = P_count       # slack for fairness inequalities
    n_vars = NK + n_t + n_u + n_slack

    # Index helpers
    def p_idx(n, k):  return n * K + k
    def t_idx(pi, k): return NK + pi * K + k
    def u_idx(pi, k): return NK + n_t + pi * K + k
    def s_idx(pi):    return NK + n_t + n_u + pi

    # ── Objective ──────────────────────────────────────────────────────────────
    # MSE = (1/N) * sum_{n,k} p[n,k] * (s_k^2 - 2*mu_n*s_k)  +  constant
    c = np.zeros(n_vars)
    cost_nk = (sq[None, :] - 2 * m1[:, None] * scores[None, :]) / N  # (N, K)
    c[:NK] = cost_nk.ravel()

    # Constant part of MSE (does not affect argmin, added back after solve)
    mse_constant = float(np.sum(P_human @ sq) / N)

    # ── Build A, b incrementally using COO lists ───────────────────────────────
    I, J, V = [], [], []   # row, col, value
    b_list = []
    row = 0

    def add(r, col, val):
        I.append(r); J.append(col); V.append(val)

    # ── Block 1: Zero cone (equalities) ───────────────────────────────────────

    # 1a. Simplex: sum_k p[n,k] = 1
    for n in range(N):
        for k in range(K):
            add(row + n, p_idx(n, k), 1.0)
        b_list.append(1.0)
    row += N

    # 1b. Fairness: sum_k (t[pi,k] + u[pi,k]) + slack[pi] = d_nm
    for pi, (n, m) in enumerate(pairs):
        d_nm = dist_func(P_human[n], P_human[m])
        for k in range(K):
            add(row, t_idx(pi, k), 1.0)
            add(row, u_idx(pi, k), 1.0)
        add(row, s_idx(pi), 1.0)
        b_list.append(d_nm)
        row += 1

    if match_score_dist:
        row = _add_marginal_constraints(I, J, V, b_list, row, P_human, p_idx)

    n_zero = row  # number of equality rows

    # ── Block 2: Non-negative cone ─────────────────────────────────────────────

    # 2a. p[n,k] >= 0
    for n in range(N):
        for k in range(K):
            add(row, p_idx(n, k), -1.0)
            b_list.append(0.0)
            row += 1

    # 2b. slack[pi] >= 0
    for pi in range(P_count):
        add(row, s_idx(pi), -1.0)
        b_list.append(0.0)
        row += 1

    n_nn = row - n_zero

    # ── Block 3: Exponential cone ─────────────────────────────────────────────
    # 3 rows per (a,b,c) triple, 2 triples per (pi, k)

    for pi, (n, m) in enumerate(pairs):
        for k in range(K):
            # kl_div(p[n,k], p[m,k]) <= t[pi,k]
            # SCS K_exp = {(x,y,z) : y*exp(x/y) <= z}  ->  row order: x, y, z
            # Set x = p[m,k]-t[pi,k],  y = p[n,k],  z = e*p[m,k]
            add(row,     p_idx(m, k), -1.0)   # x = p[m,k] - t[pi,k]
            add(row,     t_idx(pi, k), 1.0)
            b_list.append(0.0)
            add(row + 1, p_idx(n, k), -1.0)   # y = p[n,k]
            b_list.append(0.0)
            add(row + 2, p_idx(m, k), -np.e)  # z = e*p[m,k]
            b_list.append(0.0)
            row += 3

            # kl_div(p[m,k], p[n,k]) <= u[pi,k]
            # Set x = p[n,k]-u[pi,k],  y = p[m,k],  z = e*p[n,k]
            add(row,     p_idx(n, k), -1.0)   # x = p[n,k] - u[pi,k]
            add(row,     u_idx(pi, k), 1.0)
            b_list.append(0.0)
            add(row + 1, p_idx(m, k), -1.0)   # y = p[m,k]
            b_list.append(0.0)
            add(row + 2, p_idx(n, k), -np.e)  # z = e*p[n,k]
            b_list.append(0.0)
            row += 3

    n_exp = (row - n_zero - n_nn) // 3  # number of exp-cone triples

    # ── Assemble and solve ─────────────────────────────────────────────────────
    A = sp.csc_matrix((V, (I, J)), shape=(row, n_vars))
    b_vec = np.array(b_list, dtype=float)

    cone = {_scs_zero_cone_key(): n_zero, 'l': n_nn, 'ep': n_exp}

    sol = scs.solve(
        {'c': c, 'A': A, 'b': b_vec},
        cone,
        verbose=verbose,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        eps_infeas=eps_infeas,
        max_iters=max_iters,
    )

    status = sol['info']['status']
    if 'solved' not in status.lower():
        warnings.warn(f"SCS status: {status}")

    P_machine = sol['x'][:NK].reshape(N, K)
    # Clip small negatives from numerical noise
    P_machine = np.clip(P_machine, 0, None)
    P_machine /= P_machine.sum(axis=1, keepdims=True)

    min_mse = float(c[:NK] @ sol['x'][:NK]) + mse_constant

    return P_machine, min_mse


# ── Total variation solver (LP) ────────────────────────────────────────────────

def _solve_tv(P_human, dist_func, match_score_dist, verbose, eps_abs, eps_rel, eps_infeas, max_iters):
    """
    Total variation variant.  TV(p,q) = ½·‖p−q‖₁

    This is a pure LP — no exponential cone required.  The absolute value
    |p[n,k] − p[m,k]| is linearised via auxiliary variables r[pi,k]:

        r[pi,k] ≥  p[n,k] − p[m,k]
        r[pi,k] ≥  p[m,k] − p[n,k]
        ½·sum_k r[pi,k] ≤ d_nm   ⟺   sum_k r[pi,k] + slack[pi] = 2·d_nm

    Variable layout:  x = [ p_flat (N·K) | r (P·K) | slack (P) ]
    """
    import scs

    N, K = P_human.shape
    scores = np.arange(K, dtype=float)
    sq = scores ** 2
    m1 = P_human @ scores

    pairs = [(n, m) for n in range(N) for m in range(n + 1, N)]
    P_count = len(pairs)

    NK = N * K
    n_r = P_count * K
    n_slack = P_count
    n_vars = NK + n_r + n_slack

    def p_idx(n, k):  return n * K + k
    def r_idx(pi, k): return NK + pi * K + k
    def s_idx(pi):    return NK + n_r + pi

    c = np.zeros(n_vars)
    cost_nk = (sq[None, :] - 2 * m1[:, None] * scores[None, :]) / N
    c[:NK] = cost_nk.ravel()
    mse_constant = float(np.sum(P_human @ sq) / N)

    I, J, V = [], [], []
    b_list = []
    row = 0

    def add(r, col, val):
        I.append(r); J.append(col); V.append(val)

    # ── Block 1: Zero cone (equalities) ───────────────────────────────────────

    # 1a. Simplex: sum_k p[n,k] = 1
    for n in range(N):
        for k in range(K):
            add(row + n, p_idx(n, k), 1.0)
        b_list.append(1.0)
    row += N

    # 1b. Fairness: sum_k r[pi,k] + slack[pi] = 2*d_nm
    for pi, (n, m) in enumerate(pairs):
        d_nm = dist_func(P_human[n], P_human[m])
        for k in range(K):
            add(row, r_idx(pi, k), 1.0)
        add(row, s_idx(pi), 1.0)
        b_list.append(2.0 * d_nm)
        row += 1

    # 1c. Optional marginal matching: sum_n p[n,k] = sum_n P_human[n,k]
    if match_score_dist:
        row = _add_marginal_constraints(I, J, V, b_list, row, P_human, p_idx)

    n_zero = row

    # ── Block 2: Non-negative cone ─────────────────────────────────────────────

    # 2a. p[n,k] >= 0
    for n in range(N):
        for k in range(K):
            add(row, p_idx(n, k), -1.0)
            b_list.append(0.0)
            row += 1

    # 2b. r[pi,k] >= 0
    for pi in range(P_count):
        for k in range(K):
            add(row, r_idx(pi, k), -1.0)
            b_list.append(0.0)
            row += 1

    # 2c. slack[pi] >= 0
    for pi in range(P_count):
        add(row, s_idx(pi), -1.0)
        b_list.append(0.0)
        row += 1

    # 2d. p[n,k] - p[m,k] <= r[pi,k]
    #     s = 0 - (p[n,k] - p[m,k] - r[pi,k]) = r[pi,k] - p[n,k] + p[m,k] >= 0
    for pi, (n, m) in enumerate(pairs):
        for k in range(K):
            add(row, p_idx(n, k),   1.0)
            add(row, p_idx(m, k),  -1.0)
            add(row, r_idx(pi, k), -1.0)
            b_list.append(0.0)
            row += 1

    # 2e. p[m,k] - p[n,k] <= r[pi,k]
    #     s = 0 - (p[m,k] - p[n,k] - r[pi,k]) = r[pi,k] - p[m,k] + p[n,k] >= 0
    for pi, (n, m) in enumerate(pairs):
        for k in range(K):
            add(row, p_idx(m, k),   1.0)
            add(row, p_idx(n, k),  -1.0)
            add(row, r_idx(pi, k), -1.0)
            b_list.append(0.0)
            row += 1

    n_nn = row - n_zero

    A = sp.csc_matrix((V, (I, J)), shape=(row, n_vars))
    b_vec = np.array(b_list, dtype=float)
    cone = {_scs_zero_cone_key(): n_zero, 'l': n_nn}   # pure LP, no 'ep'

    sol = scs.solve(
        {'c': c, 'A': A, 'b': b_vec},
        cone,
        verbose=verbose,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        eps_infeas=eps_infeas,
        max_iters=max_iters,
    )

    status = sol['info']['status']
    if 'solved' not in status.lower():
        warnings.warn(f"SCS status: {status}")

    P_machine = sol['x'][:NK].reshape(N, K)
    P_machine = np.clip(P_machine, 0, None)
    P_machine /= P_machine.sum(axis=1, keepdims=True)
    min_mse = float(c[:NK] @ sol['x'][:NK]) + mse_constant

    return P_machine, min_mse


# ── Jensen-Shannon solver ───────────────────────────────────────────────────────

def _solve_jsd(P_human, dist_func, match_score_dist, verbose, eps_abs, eps_rel, eps_infeas, max_iters):
    """
    Jensen-Shannon variant.  JSD(p,q) = [KL(p||m) + KL(q||m)] / 2
    where m = (p+q)/2, measured in bits.

    Extra variables: m_flat (P*K) — the elementwise midpoints.
    Extra equalities: m[pi,k] = (p[n,k] + p[m,k]) / 2   for each (pi,k)
    The kl_div constraints then use m[pi,k] as the second argument.
    """
    import scs

    N, K = P_human.shape
    scores = np.arange(K, dtype=float)
    sq = scores ** 2
    m1 = P_human @ scores

    pairs = [(n, m) for n in range(N) for m in range(n + 1, N)]
    P_count = len(pairs)

    NK = N * K
    n_mid = P_count * K     # midpoint variables m[pi,k]
    n_t = P_count * K       # aux for kl_div(p_n, mid)
    n_u = P_count * K       # aux for kl_div(p_m, mid)
    n_slack = P_count
    n_vars = NK + n_mid + n_t + n_u + n_slack

    def p_idx(n, k):   return n * K + k
    def mid_idx(pi, k): return NK + pi * K + k
    def t_idx(pi, k):  return NK + n_mid + pi * K + k
    def u_idx(pi, k):  return NK + n_mid + n_t + pi * K + k
    def s_idx(pi):     return NK + n_mid + n_t + n_u + pi

    c = np.zeros(n_vars)
    cost_nk = (sq[None, :] - 2 * m1[:, None] * scores[None, :]) / N
    c[:NK] = cost_nk.ravel()
    mse_constant = float(np.sum(P_human @ sq) / N)

    I, J, V = [], [], []
    b_list = []
    row = 0

    def add(r, col, val):
        I.append(r); J.append(col); V.append(val)

    # ── Zero cone ──────────────────────────────────────────────────────────────

    # Simplex
    for n in range(N):
        for k in range(K):
            add(row + n, p_idx(n, k), 1.0)
        b_list.append(1.0)
    row += N

    # Midpoint definition: m[pi,k] = (p[n,k] + p[m,k]) / 2
    # => 2*m[pi,k] - p[n,k] - p[m,k] = 0
    for pi, (n, m) in enumerate(pairs):
        for k in range(K):
            add(row, mid_idx(pi, k), 2.0)
            add(row, p_idx(n, k), -1.0)
            add(row, p_idx(m, k), -1.0)
            b_list.append(0.0)
            row += 1

    # Fairness: sum_k (t[pi,k] + u[pi,k]) + slack[pi] = 2*log(2)*d_nm
    # Factor 2: JSD = [KL(p||m)+KL(q||m)]/2 and the /2 is omitted in the constraint.
    # Factor log(2): d_nm is in bits, while the exponential cones use natural logs.
    for pi, (n, m) in enumerate(pairs):
        d_nm = dist_func(P_human[n], P_human[m])
        for k in range(K):
            add(row, t_idx(pi, k), 1.0)
            add(row, u_idx(pi, k), 1.0)
        add(row, s_idx(pi), 1.0)
        b_list.append(2.0 * np.log(2) * d_nm)
        row += 1

    if match_score_dist:
        row = _add_marginal_constraints(I, J, V, b_list, row, P_human, p_idx)

    n_zero = row

    # ── Non-negative cone ──────────────────────────────────────────────────────

    for n in range(N):
        for k in range(K):
            add(row, p_idx(n, k), -1.0)
            b_list.append(0.0)
            row += 1

    for pi in range(P_count):
        add(row, s_idx(pi), -1.0)
        b_list.append(0.0)
        row += 1

    n_nn = row - n_zero

    # ── Exponential cone ──────────────────────────────────────────────────────
    # kl_div(p[n,k], mid[pi,k]) <= t[pi,k]: triple (e*mid, p_n, mid - t) in K_exp
    # kl_div(p[m,k], mid[pi,k]) <= u[pi,k]: triple (e*mid, p_m, mid - u) in K_exp

    for pi, (n, m) in enumerate(pairs):
        for k in range(K):
            # kl_div(p_n_k, mid_k) <= t[pi,k]
            # SCS triple (x,y,z): x=mid-t, y=p_n, z=e*mid
            add(row,     mid_idx(pi, k), -1.0)  # x = mid - t
            add(row,     t_idx(pi, k), 1.0)
            b_list.append(0.0)
            add(row + 1, p_idx(n, k), -1.0)     # y = p_n
            b_list.append(0.0)
            add(row + 2, mid_idx(pi, k), -np.e) # z = e*mid
            b_list.append(0.0)
            row += 3

            # kl_div(p_m_k, mid_k) <= u[pi,k]
            # SCS triple (x,y,z): x=mid-u, y=p_m, z=e*mid
            add(row,     mid_idx(pi, k), -1.0)  # x = mid - u
            add(row,     u_idx(pi, k), 1.0)
            b_list.append(0.0)
            add(row + 1, p_idx(m, k), -1.0)     # y = p_m
            b_list.append(0.0)
            add(row + 2, mid_idx(pi, k), -np.e) # z = e*mid
            b_list.append(0.0)
            row += 3

    n_exp = (row - n_zero - n_nn) // 3

    A = sp.csc_matrix((V, (I, J)), shape=(row, n_vars))
    b_vec = np.array(b_list, dtype=float)
    cone = {_scs_zero_cone_key(): n_zero, 'l': n_nn, 'ep': n_exp}

    sol = scs.solve(
        {'c': c, 'A': A, 'b': b_vec},
        cone,
        verbose=verbose,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        eps_infeas=eps_infeas,
        max_iters=max_iters,
    )

    status = sol['info']['status']
    if 'solved' not in status.lower():
        warnings.warn(f"SCS status: {status}")

    P_machine = sol['x'][:NK].reshape(N, K)
    P_machine = np.clip(P_machine, 0, None)
    P_machine /= P_machine.sum(axis=1, keepdims=True)
    min_mse = float(c[:NK] @ sol['x'][:NK]) + mse_constant

    return P_machine, min_mse


def simulate_human_scores(N: int, K: int) -> np.ndarray:
    """
    Radomly generate human score distributions using binomial PMFs with random success probabilities.
    """
    return np.array([binom.pmf(np.arange(K), K, np.clip(np.random.rand(), a_min=0.1, a_max=0.9)) for _ in range(N)])


def compare_score_distributions(P_human: np.ndarray, P_machine: np.ndarray, filename: str | None = None) -> None:
    """
    Plot side-by-side bar charts of the human and machine score distributions.

    Parameters
    ----------
    P_human, P_machine : np.ndarray
        (N, K) arrays of score distributions

    filename : str or None
        Optional. If provided, the figure will be saved to this filename.
        If `None`, the plot will only be displayed.
    """
    N, K = P_human.shape

    width = 1 / 3
    fig, axes = plt.subplots(int(N / 2), 2, figsize=(10, 12), sharey=True)
    axes = axes.flatten()

    k_vec = np.arange(K)
    for n in range(len(axes)):
        ax = axes[n]
        ax.bar(k_vec - width / 2, P_human[n], width, label="Human")
        ax.bar(k_vec + width / 2, P_machine[n], width, label="Machine")
        ax.set_xlabel("Score")
        ax.set_ylabel("Probability")
        ax.legend()

    plt.tight_layout()
    plt.show()

    if filename:
        fig.savefig(filename)


# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    # Small random example: N responses, K score levels
    N, K = 50, 6
    rng = np.random.default_rng(42)
    P_ref = rng.dirichlet(np.ones(K), size=N)

    dist_funcs = {
        "Jeffreys": jeffreys_dist,
        "Jensen-Shannon": jensen_shannon_dist,
        "Total-Variation": total_variation_dist,
    }

    print(f"N = {N}, K = {K}, {N * (N - 1) // 2} pairwise constraints\n")
    print(f"{'Distance':<16} {'Time [s]':>9} {'MSE':>8} {'Max. violation':>15} {'Inf. pairs':>11}")
    print("-" * 63)
    for name, dist in dist_funcs.items():
        t0 = time.perf_counter()
        P_fair, mse = get_fair_score_distributions(P_ref, name)
        t1 = time.perf_counter()
        violations = np.array([
            dist(P_fair[n], P_fair[m]) - dist(P_ref[n], P_ref[m])
            for n in range(N) for m in range(n + 1, N)
        ])
        # Entries clipped to exactly zero make the Jeffreys divergence infinite;
        # count those pairs separately
        finite = np.isfinite(violations)
        print(f"{name:<16} {t1 - t0:9.2f} {mse:8.4f} "
              f"{violations[finite].max():15.2e} {np.sum(~finite):11d}")
