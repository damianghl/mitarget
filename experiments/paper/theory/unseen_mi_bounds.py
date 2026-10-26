"""Moment-identified bounds on I(X;Y) for a sparse categorical X and binary Y.

Companion to unseen_mi_limits.tex. Under exchangeable conditionals (q_x iid ~ pi,
independent of popularity) and many levels,
    I(X;Y) = h(mu) - E_pi[h(q)],   h = binary entropy (nats),
and levels seen at most L times identify only m_1..m_L of pi. This script computes
  1. E_L(h): best uniform degree-L polynomial approximation error of h on [0,1]
     (half the worst-case ambiguity of E_pi h given L moments);
  2. the identified interval [I_lo, I_hi] given the first L moments of a few priors,
     and the value the moment-matched beta (what a beta / Dirichlet estimator
     converges to when all levels have n <= 2) assigns to I;
  3. a finite-sample demo: interval from estimated moments vs. a beta-ML plug-in.
Run: python3 unseen_mi_bounds.py
"""
import numpy as np
from scipy.optimize import linprog, minimize_scalar
from scipy.special import digamma, gammaln

GRID = np.linspace(0.0, 1.0, 2001)


def h(q):
    q = np.clip(q, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(q > 0, -q * np.log(q), 0.0)
        b = np.where(q < 1, -(1 - q) * np.log(1 - q), 0.0)
    return a + b


def best_poly_error(L, grid=GRID):
    """min_t s.t. |h(q) - P(q)| <= t on the grid, P of degree L (Chebyshev basis)."""
    x = 2 * grid - 1
    V = np.polynomial.chebyshev.chebvander(x, L)  # (G, L+1)
    G = len(grid)
    hv = h(grid)
    # variables: c_0..c_L, t ; minimise t
    c = np.zeros(L + 2); c[-1] = 1
    A = np.vstack([np.hstack([V, -np.ones((G, 1))]), np.hstack([-V, -np.ones((G, 1))])])
    b = np.concatenate([hv, -hv])
    r = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * (L + 1) + [(0, None)], method="highs")
    return r.x[-1]


def eh_range(moments, symmetric=False, grid=GRID):
    """min and max of E_pi h(q) over priors on the grid with E q^j = moments[j-1]."""
    L = len(moments)
    A = np.vstack([np.ones_like(grid)] + [grid ** j for j in range(1, L + 1)])
    b = np.concatenate([[1.0], moments])
    if symmetric:  # pi(q) = pi(1-q)
        G = len(grid)
        S = np.zeros((G // 2, G))
        for i in range(G // 2):
            S[i, i] = 1; S[i, G - 1 - i] = -1
        A = np.vstack([A, S]); b = np.concatenate([b, np.zeros(G // 2)])
    out = []
    for sgn in (1, -1):
        r = linprog(sgn * h(grid), A_eq=A, b_eq=b, bounds=(0, None), method="highs")
        if r.status != 0:
            return np.nan, np.nan
        out.append(sgn * r.fun)
    return out[0], out[1]


def beta_eh(a, b):
    s = a + b
    eqlogq = (a / s) * (digamma(a + 1) - digamma(s + 1))
    eq1log = (b / s) * (digamma(b + 1) - digamma(s + 1))
    return -(eqlogq + eq1log)


def matched_beta(m1, m2):
    v = m2 - m1 ** 2
    s = m1 * (1 - m1) / v - 1
    return m1 * s, (1 - m1) * s


def atoms_moments(atoms, w, L):
    atoms, w = np.asarray(atoms), np.asarray(w)
    return np.array([np.sum(w * atoms ** j) for j in range(1, L + 1)])


def beta_moments(a, b, L):
    m, out = 1.0, []
    for j in range(L):
        m *= (a + j) / (a + b + j)
        out.append(m)
    return np.array(out)


PRIORS = {
    "junk f=0.5, q0=0.1 (symmetric)": dict(atoms=[0.1, 0.5, 0.9], w=[0.25, 0.5, 0.25], sym=True),
    "junk f=0.8, q0=0.1 (symmetric)": dict(atoms=[0.1, 0.5, 0.9], w=[0.1, 0.8, 0.1], sym=True),
    "3 atoms {.2,.3,.85}; {.3,.4,.3}": dict(atoms=[0.2, 0.3, 0.85], w=[0.3, 0.4, 0.3], sym=False),
    "Beta(0.5, 0.5)": dict(beta=(0.5, 0.5), sym=True),
    "Beta(1, 3)": dict(beta=(1.0, 3.0), sym=False),
}


def prior_info(p, L):
    if "beta" in p:
        a, b = p["beta"]
        return beta_moments(a, b, L), beta_eh(a, b)
    m = atoms_moments(p["atoms"], p["w"], L)
    return m, float(np.sum(np.asarray(p["w"]) * h(np.asarray(p["atoms"]))))


def part1():
    print("1. Best uniform polynomial approximation error E_L(h) on [0,1] (nats)")
    print("   L   E_L(h)   L^2*E_L")
    for L in range(1, 13):
        e = best_poly_error(L)
        print(f"  {L:2d}  {e:.4f}   {L*L*e:.3f}")
    print()


def part2(Lmax=6):
    print("2. Identified interval for I(X;Y) = h(mu) - E h(q) given m_1..m_L (nats)")
    for name, p in PRIORS.items():
        m, eh_true = prior_info(p, Lmax)
        mu = m[0]
        I_true = h(mu) - eh_true
        a, b = matched_beta(m[0], m[1])
        I_beta = h(mu) - beta_eh(a, b)
        print(f"  {name}: true I = {I_true:.4f}, moment-matched beta I = {I_beta:.4f}")
        print("     L   I_lo     I_hi    width   (symmetry imposed: lo, hi)" if p["sym"] else "     L   I_lo     I_hi    width")
        for L in range(1, Lmax + 1):
            lo, hi = eh_range(m[:L])
            line = f"    {L:2d}  {h(mu)-hi:.4f}  {h(mu)-lo:.4f}  {hi-lo:.4f}"
            if p["sym"]:
                slo, shi = eh_range(m[:L], symmetric=True)
                line += f"   ({h(mu)-shi:.4f}, {h(mu)-slo:.4f})"
            print(line)
    print()


def beta_ml_I(n, k):
    """Beta-binomial type-II ML with mu fixed at the pooled label rate; plug-in I."""
    mu = k.sum() / n.sum()
    sel = n >= 2
    nn, kk = n[sel], k[sel]

    def nll(logs):
        s = np.exp(logs); a, b = mu * s, (1 - mu) * s
        return -np.sum(gammaln(a + kk) + gammaln(b + nn - kk) - gammaln(s + nn)
                       - gammaln(a) - gammaln(b) + gammaln(s))
    r = minimize_scalar(nll, bounds=(-6, 8), method="bounded")
    s = np.exp(r.x)
    return h(mu) - beta_eh(mu * s, (1 - mu) * s)


def comb(n, j):
    return np.exp(gammaln(n + 1) - gammaln(j + 1) - gammaln(n - j + 1))


def part3(seed=0):
    print("3. Finite sample: K levels, uniform popularity, N = lam*K samples")
    print("   prior = junk f=0.5, q0=0.1. Interval uses moments 1..L with +-2 SE slack.")
    rng = np.random.default_rng(seed)
    atoms, w = np.array([0.1, 0.5, 0.9]), np.array([0.25, 0.5, 0.25])
    I_true = h(0.5) - np.sum(w * h(atoms))
    print(f"   true I = {I_true:.4f}")
    print("      K     lam  F1     F2    F3+   beta-ML I   L  interval")
    for K, lam in [(100_000, 0.3), (100_000, 1.0), (1_000_000, 0.3), (1_000_000, 1.0)]:
        q = rng.choice(atoms, size=K, p=w)
        n = rng.poisson(lam, size=K)
        k = rng.binomial(n, q)
        F1, F2, F3 = (n == 1).sum(), (n == 2).sum(), (n >= 3).sum()
        Ib = beta_ml_I(n, k)
        for L in (2, 3, 4):
            est, se = [], []
            for j in range(1, L + 1):
                sel = n >= j
                u = comb(k[sel], j) / comb(n[sel], j)
                est.append(u.mean()); se.append(u.std() / np.sqrt(sel.sum()))
            est, se = np.array(est), np.array(se)
            lo_hi = interval_with_slack(est, 2 * se)
            mu = est[0]
            print(f"  {K:8d}  {lam:.1f}  {F1:6d} {F2:6d} {F3:6d}   {Ib:.4f}     {L}  "
                  f"[{lo_hi[0]:.4f}, {lo_hi[1]:.4f}]")
    print()


def interval_with_slack(m, slack, grid=GRID):
    """Range of h(mu) - E h over priors whose first L moments lie in m +- slack.
    mu itself is a variable, so the objective h(m1) - E h is handled by bounding
    h(m1) over its slack interval (concave h, small slack: conservative)."""
    L = len(m)
    A = np.vstack([grid ** j for j in range(1, L + 1)])
    A_ub = np.vstack([A, -A]); b_ub = np.concatenate([m + slack, -(m - slack)])
    A_eq = np.ones((1, len(grid))); b_eq = [1.0]
    rmin = linprog(h(grid), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    rmax = linprog(-h(grid), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    mus = np.linspace(m[0] - slack[0], m[0] + slack[0], 21)
    hmu_lo, hmu_hi = h(mus).min(), h(mus).max()
    return hmu_lo + rmax.fun, hmu_hi - rmin.fun


if __name__ == "__main__":
    part1()
    part2()
    part3()
