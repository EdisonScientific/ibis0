"""Weight-of-Evidence reliability: is a system's *claimed* incremental weight of evidence matched by
the *realized* movement in the odds of the truth, independent of where its belief started?

For each system m, domain d, episode i, and stub position j (context length j-1 -> j on the nested
sensitivity sweep), we form
    w   = (logit(p_j) - logit(p_{j-1})) * 10/ln10   decibans   (claimed weight of evidence)
    ell = logit(p_{j-1})                            nats       (incoming belief; the nuisance state)
    theta in {0,1}                                             (episode ground-truth label)
Quantile-bin `ell` into strata S (the belief state we make the diagram invariant to) and `w` into
per-system tiers K. Relative to the near-neutral reference tier (the tier whose mean w, averaged over
domains, is closest to zero; chosen once per system), the *implied* gap of a tier is its mean w (in
nats); the *realized* gap is the Mantel-Haenszel pooled log odds-ratio of theta across the ell strata
(holding incoming belief fixed). A calibrated system lies on the identity. Two slopes summarise it, both
exchange rates (<1 overclaims, >1 underclaims): `fit_slope`, the through-origin least-squares slope of
the reliability points (the paper's beta and the figure's bars), and beta_WoE, a stratum-adjusted
logistic fit of theta on w.

Reads the published test set's sensitivity sweeps (N=256 episodes/domain, k=0..16, one draw each), so
there is no model inference here. beta_WoE is therefore a single-draw estimate: `ell` carries finite-
sample noise that attenuates it toward 0 (regression dilution), so beta_WoE is conservative. The
deterministic classifier has noise-free `ell` and needs no such correction.

Emits two tidy CSVs the renderer consumes: `<out>_points.csv` (per system/scope/tier reliability
points with bootstrap bands) and `<out>_beta.csv` (per system/scope fit_slope and beta_WoE with CIs,
zero-atom fraction, n).

Run: uv run --no-sync python scripts/analysis/rq2_evidence_deliberation/analyze_woe_reliability.py \\
         [--root ...] [--out outputs/bf/woe_reliability]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "scripts") / "lib"))
from roster_palette import ORDER, ROSTER  # noqa: E402

R2_DEFAULT = "ibis_data/eval/EDB-Bio@df43070d"
DOMAINS = ["tahoe", "depmap", "gwas"]
DECIBAN = 10.0 / np.log(10.0)
EPS = 1e-3
KMAX = 16
K_TIERS = 5
S_STRATA = 5


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load_curves(root, model_dir, domain):
    """Return {k: {episode_id: parsed_prob}} for k=0..KMAX from <root>/<model_dir>/<domain>_ctx{k}.json."""
    curves = {}
    for k in range(KMAX + 1):
        f = Path(root) / model_dir / f"{domain}_ctx{k}.json"
        if not f.exists():
            continue
        rows = json.loads(f.read_text())["results"]
        curves[k] = {r["episode_id"]: (r["parsed_prob"], r["label"]) for r in rows if r.get("parsed_prob") is not None}
    return curves


def build_rows(root, key):
    """One row per (episode, stub position j=1..KMAX): claimed w, incoming belief ell, truth theta."""
    model_dir = ROSTER[key].eval_key
    out = []
    for domain in DOMAINS:
        curves = load_curves(root, model_dir, domain)
        for k in range(1, KMAX + 1):
            cur, prev = curves.get(k, {}), curves.get(k - 1, {})
            for ep in cur.keys() & prev.keys():
                p_cur, theta = cur[ep]
                p_prev, _ = prev[ep]
                out.append(
                    {
                        "system": key,
                        "domain": domain,
                        "episode_id": ep,
                        "j": k,
                        "w_db": (logit(p_cur) - logit(p_prev)) * DECIBAN,
                        "ell": float(logit(p_prev)),
                        "theta": int(theta),
                    }
                )
    return pd.DataFrame(out)


def _bin_edges(values, nbins):
    """Quantile edges with tied-edge dedup; returns interior edges for np.digitize."""
    qs = np.quantile(values, np.linspace(0, 1, nbins + 1))
    return np.unique(qs)[1:-1]


def _assign_bins(df, edges_w, edges_s):
    tier = np.digitize(df["w_db"].to_numpy(), edges_w)
    strat = np.digitize(df["ell"].to_numpy(), edges_s)
    return tier, strat


def mh_logor(theta, group, strat, _corrected=False):
    """Mantel-Haenszel pooled log odds-ratio of theta for group==1 vs group==0, across strata. A
    perfectly separable tier (belief saturation) collapses num or den to 0; a Haldane-Anscombe 0.5
    correction on every cell then yields a large finite log-OR (correctly reading as off-scale)."""
    cc = 0.5 if _corrected else 0.0
    num = den = 0.0
    for s in np.unique(strat):
        m = strat == s
        t, g = theta[m], group[m]
        a = np.sum((g == 1) & (t == 1)) + cc
        b = np.sum((g == 1) & (t == 0)) + cc
        c = np.sum((g == 0) & (t == 1)) + cc
        d = np.sum((g == 0) & (t == 0)) + cc
        n = a + b + c + d
        if n == 4 * cc:  # stratum empty (only the correction mass)
            continue
        num += a * d / n
        den += b * c / n
    if num <= 0 or den <= 0:
        return mh_logor(theta, group, strat, _corrected=True) if not _corrected else np.nan
    return float(np.log(num / den))


def fe_logistic_beta(w_nats, theta, strat):
    """Stratum-adjusted logistic slope of theta on w (nats): theta ~ w + C(strat). The unconditional
    analogue of conditional logistic; with these row counts and few strata the two agree closely."""
    if len(np.unique(theta)) < 2:
        return np.nan
    dummies = pd.get_dummies(strat, drop_first=True).to_numpy(dtype=float)
    X = np.column_stack([w_nats, dummies]) if dummies.size else w_nats.reshape(-1, 1)
    clf = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000)
    clf.fit(X, theta)
    return float(clf.coef_[0, 0])


def scope_stats(df, edges_w, edges_s, ref_rank):
    """Reliability points + beta_WoE for one (system, domain) subset. Fixed bin edges keep bootstrap
    resamples comparable; the exact-zero-w atom is excluded.

    Both axes are measured relative to the near-neutral (w~=0) reference tier `ref_rank`, so the origin
    is "no net evidence": tiers of more-negative claimed weight fall to negative x, and a system whose
    claimed disbelief is realized lands them at negative y (the third quadrant). Anchoring to the neutral
    tier rather than the weakest one is what lets the diagram show both signs."""
    df = df[df["w_db"] != 0.0]
    if len(df) < 4 * (K_TIERS + S_STRATA):
        return None
    tier, strat = _assign_bins(df, edges_w, edges_s)
    theta = df["theta"].to_numpy()
    w_nats = df["w_db"].to_numpy() / DECIBAN
    tiers = sorted(np.unique(tier))
    ref = ref_rank if ref_rank in tiers else min(tiers, key=lambda t: abs(np.mean(w_nats[tier == t])))
    wref = float(np.mean(w_nats[tier == ref]))
    pts = []
    for t in tiers:
        sub = np.isin(tier, [ref, t])
        y = 0.0 if t == ref else mh_logor(theta[sub], (tier[sub] == t).astype(int), strat[sub])
        x = float(np.mean(w_nats[tier == t]) - wref)
        pts.append({"tier": int(t), "x_implied": x, "y_realized": y})
    beta = fe_logistic_beta(w_nats, theta, strat)
    return pts, beta


def cluster_resample(df, rng):
    ids = df["episode_id"].unique()
    take = rng.choice(ids, size=len(ids), replace=True)
    idx = df.groupby("episode_id").indices
    return df.iloc[np.concatenate([idx[e] for e in take])]


def linfit(x, y):
    """Straight-line summary of the five reliability points: the through-origin slope (the exchange rate,
    since the reference tier is pinned at the origin) and the free-fit R^2 (goodness of the linear fit).
    Slope is scale-free, so it is the same in decibans or nats."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    sxx = float(np.sum(x * x))
    slope = float(np.sum(x * y) / sxx) if sxx > 0 else float("nan")
    b, a = np.polyfit(x, y, 1)
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum((y - (a + b * x)) ** 2) / sst) if sst > 0 else float("nan")
    return slope, r2


def analyze(root, n_boot, seed):
    rng = np.random.default_rng(seed)
    point_rows, beta_rows = [], []
    for key in ORDER:
        full = build_rows(root, key)
        if full.empty:
            print(f"  {key}: no data, skipping", file=sys.stderr)
            continue
        # per-domain fixed edges (nonzero-w rows) shared by point estimate and every bootstrap resample
        edges = {}
        for d in DOMAINS:
            dd = full[(full["domain"] == d) & (full["w_db"] != 0.0)]
            edges[d] = (_bin_edges(dd["w_db"], K_TIERS), _bin_edges(dd["ell"], S_STRATA))

        # Reference tier rank = the near-neutral (w~=0) tier, chosen once across domains so the macro
        # average-by-rank keeps a stable origin. Signed axes then show whether claimed refuting evidence
        # (negative WoE) actually lowers the odds of the truth.
        rank_means: dict[int, list[float]] = {}
        for d in DOMAINS:
            dd = full[(full["domain"] == d) & (full["w_db"] != 0.0)]
            tt = np.digitize(dd["w_db"].to_numpy(), edges[d][0])
            wn = dd["w_db"].to_numpy() / DECIBAN
            for t in np.unique(tt):
                rank_means.setdefault(int(t), []).append(float(np.mean(wn[tt == t])))
        ref_rank = min(rank_means, key=lambda t: abs(float(np.mean(rank_means[t]))))

        def per_domain(frame):
            res = {}
            for d in DOMAINS:
                sub = frame[frame["domain"] == d]
                s = scope_stats(sub, *edges[d], ref_rank) if not sub.empty else None
                if s:
                    res[d] = s
            return res

        base = per_domain(full)
        zero_frac = float(np.mean(full["w_db"] == 0.0))

        # bootstrap: resample episodes within each domain, recompute; macro = equal-weight domain mean
        boot = {d: {"x": [], "y": [], "beta": []} for d in DOMAINS}
        boot_macro = {"x": [], "y": [], "beta": []}
        for _ in range(n_boot):
            rs = per_domain(pd.concat([cluster_resample(full[full["domain"] == d], rng) for d in DOMAINS]))
            xs, ys, bs = [], [], []
            for d in DOMAINS:
                if d not in rs:
                    continue
                pts, beta = rs[d]
                xv = [p["x_implied"] for p in pts]
                yv = [p["y_realized"] for p in pts]
                boot[d]["x"].append(xv)
                boot[d]["y"].append(yv)
                boot[d]["beta"].append(beta)
                xs.append(xv)
                ys.append(yv)
                bs.append(beta)
            if len(xs) == len(DOMAINS):  # macro only when all domains present in the resample
                boot_macro["x"].append(np.mean(xs, axis=0))
                boot_macro["y"].append(np.mean(ys, axis=0))
                boot_macro["beta"].append(np.nanmean(bs))

        def ci(arr, axis=0):
            a = np.asarray(arr, float)
            return np.nanpercentile(a, 2.5, axis=axis), np.nanpercentile(a, 97.5, axis=axis)

        # emit per-domain scopes
        for d in DOMAINS:
            if d not in base:
                continue
            pts, beta = base[d]
            xl, xh = ci([b for b in boot[d]["x"]])
            yl, yh = ci([b for b in boot[d]["y"]])
            for i, p in enumerate(pts):
                point_rows.append(
                    dict(
                        system=key,
                        scope=d,
                        tier=p["tier"],
                        x_implied=p["x_implied"],
                        x_lo=float(xl[i]),
                        x_hi=float(xh[i]),
                        y_realized=p["y_realized"],
                        y_lo=float(yl[i]),
                        y_hi=float(yh[i]),
                    )
                )
            bl, bh = ci(boot[d]["beta"])
            slope, r2 = linfit([p["x_implied"] for p in pts], [p["y_realized"] for p in pts])
            slb = [linfit(bx, by)[0] for bx, by in zip(boot[d]["x"], boot[d]["y"])]
            sl, sh = (np.nanpercentile(slb, 2.5), np.nanpercentile(slb, 97.5)) if slb else (np.nan, np.nan)
            beta_rows.append(
                dict(
                    system=key,
                    scope=d,
                    beta=beta,
                    beta_lo=float(bl),
                    beta_hi=float(bh),
                    fit_slope=slope,
                    fit_slope_lo=float(sl),
                    fit_slope_hi=float(sh),
                    fit_r2=r2,
                    zero_frac=zero_frac,
                    n_rows=int((full["domain"] == d).sum()),
                )
            )

        # emit macro scope (equal-weight over domains present in every base domain)
        if len(base) == len(DOMAINS):
            mx = np.mean([[p["x_implied"] for p in base[d][0]] for d in DOMAINS], axis=0)
            my = np.mean([[p["y_realized"] for p in base[d][0]] for d in DOMAINS], axis=0)
            mbeta = float(np.nanmean([base[d][1] for d in DOMAINS]))
            xl, xh = ci(boot_macro["x"])
            yl, yh = ci(boot_macro["y"])
            for i in range(len(mx)):
                point_rows.append(
                    dict(
                        system=key,
                        scope="macro",
                        tier=i,
                        x_implied=float(mx[i]),
                        x_lo=float(xl[i]),
                        x_hi=float(xh[i]),
                        y_realized=float(my[i]),
                        y_lo=float(yl[i]),
                        y_hi=float(yh[i]),
                    )
                )
            bl, bh = ci(boot_macro["beta"])
            slope, r2 = linfit(mx, my)
            slb = [linfit(bx, by)[0] for bx, by in zip(boot_macro["x"], boot_macro["y"])]
            sl, sh = (np.nanpercentile(slb, 2.5), np.nanpercentile(slb, 97.5)) if slb else (np.nan, np.nan)
            beta_rows.append(
                dict(
                    system=key,
                    scope="macro",
                    beta=mbeta,
                    beta_lo=float(bl),
                    beta_hi=float(bh),
                    fit_slope=slope,
                    fit_slope_lo=float(sl),
                    fit_slope_hi=float(sh),
                    fit_r2=r2,
                    zero_frac=zero_frac,
                    n_rows=int(len(full)),
                )
            )
        print(
            f"  {key}: macro slope={beta_rows[-1].get('fit_slope', float('nan')):.2f} "
            f"R2={beta_rows[-1].get('fit_r2', float('nan')):.2f} "
            f"beta_WoE={beta_rows[-1]['beta']:.2f} zero_atom={zero_frac:.2f}"
        )
    return pd.DataFrame(point_rows), pd.DataFrame(beta_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=R2_DEFAULT, help="eval tree of the published test set's sensitivity sweeps")
    ap.add_argument("--out", default="outputs/bf/woe_reliability", help="output stem (_points.csv/_beta.csv)")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    points, betas = analyze(args.root, args.n_boot, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(f"{out}_points.csv", index=False)
    betas.to_csv(f"{out}_beta.csv", index=False)
    print(f"wrote {out}_points.csv ({len(points)} rows), {out}_beta.csv ({len(betas)} rows)")


if __name__ == "__main__":
    main()
