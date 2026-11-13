"""Campaign lift estimation (difference-in-differences) with placebo test and bootstrap CI.

Why this is built the way it is
-------------------------------
Campaigns were sent to households that already spent more, so targeted vs. non-targeted
must be compared within groups of households that looked alike *before* the campaign.

* `strat="prepre"` (recommended) groups households by spend in the window BEFORE the
  pre-window, which is independent of the noise in the pre-window itself.
* `strat="pre"` groups by the pre-window spend that is also subtracted in `lift`.
  That mechanically correlates with the outcome (regression to the mean) and produces
  a positive "lift" even when no campaign ran -> the placebo test below exposes this.
* Segments are NOT used as strata: they are defined from the whole 2-year period, i.e.
  partly from behaviour AFTER the campaign, which would bias the comparison.

placebo (`shift=1`) moves every campaign window one window-length earlier, when that
campaign cannot have had any effect. A valid estimator must return ~0 there.
"""
import numpy as np
import pandas as pd

from .config import MAX_DAY


def daily_spend_matrix(tx: pd.DataFrame):
    """Return (household ids, cumulative spend array) with cum[:, d] = spend on days 1..d."""
    daily = tx.pivot_table(index="household_key", columns="DAY", values="SALES_VALUE",
                           aggfunc="sum", fill_value=0)
    daily = daily.reindex(columns=range(1, MAX_DAY + 1), fill_value=0)
    cum = np.concatenate([np.zeros((len(daily), 1)), daily.to_numpy().cumsum(axis=1)], axis=1)
    return daily.index.to_numpy(), cum


def build_cells(tx, first_day: pd.Series, campaign_table: pd.DataFrame,
                campaign_desc: pd.DataFrame, shift: int = 0, strat: str = "prepre",
                n_bins: int = 5):
    """Household x campaign table used for DiD.

    For campaign window [s, e] of length L (shifted `shift` windows earlier for placebo):
      pre_prev = avg weekly spend in the L days before the pre-window
      pre      = avg weekly spend in the L days before the campaign
      dur      = avg weekly spend during the campaign
      lift     = dur - pre
    Only households whose first purchase is before the pre_prev window are kept.
    Returns (cells DataFrame, number of households).
    """
    hh, cum = daily_spend_matrix(tx)
    first = first_day.reindex(hh).to_numpy()

    def spend(a, b):
        return cum[:, b] - cum[:, a - 1]

    parts = []
    for _, c in campaign_desc.iterrows():
        s0, e0 = int(c.START_DAY), int(min(c.END_DAY, MAX_DAY))
        L = e0 - s0 + 1
        s, e = s0 - shift * L, e0 - shift * L
        if s - 2 * L < 1:
            continue
        pre = spend(s - L, s - 1) / L * 7
        dur = spend(s, e) / L * 7
        pre_prev = spend(s - 2 * L, s - L - 1) / L * 7
        targeted = np.isin(hh, campaign_table.loc[campaign_table.CAMPAIGN == c.CAMPAIGN, "household_key"])
        ok = first < s - 2 * L
        key = pre if strat == "pre" else pre_prev
        stratum = pd.qcut(pd.Series(key[ok]).rank(method="first"), n_bins, labels=False).to_numpy()
        parts.append(pd.DataFrame({
            "hh_idx": np.where(ok)[0], "campaign": int(c.CAMPAIGN), "type": c.DESCRIPTION,
            "stratum": stratum, "targeted": targeted[ok], "pre": pre[ok], "lift": (dur - pre)[ok],
        }))
    return pd.concat(parts, ignore_index=True), len(hh)


def estimate_lift(cells: pd.DataFrame, weights: np.ndarray | None = None, by: str | None = None,
                  min_n: int = 3):
    """Weighted DiD. `weights` are per-household (bootstrap multiplicities); `by` in {None, 'type', 'campaign'}.

    DiD per (campaign, stratum) cell = mean lift targeted - mean lift not targeted,
    averaged with weights = number of targeted households.
    """
    w = np.ones(len(cells)) if weights is None else weights[cells.hh_idx.to_numpy()]
    t = cells.targeted.to_numpy()
    d = pd.DataFrame({
        "campaign": cells.campaign.to_numpy(), "type": cells.type.to_numpy(),
        "stratum": cells.stratum.to_numpy(),
        "wt": w * t, "wc": w * ~t, "lt": w * t * cells.lift.to_numpy(), "lc": w * ~t * cells.lift.to_numpy(),
        "pre_t": w * t * cells.pre.to_numpy(),
    })
    a = d.groupby(["campaign", "type", "stratum"])[["wt", "wc", "lt", "lc", "pre_t"]].sum()
    a = a[(a.wt >= min_n) & (a.wc >= min_n)]
    a["did"] = a["lt"] / a["wt"] - a["lc"] / a["wc"]
    if by is None:
        return float(np.average(a["did"], weights=a["wt"]))
    return a.groupby(level=by).apply(lambda x: np.average(x["did"], weights=x["wt"]))


def bootstrap_lift(cells: pd.DataFrame, n_households: int, by: str | None = None,
                   n_boot: int = 200, seed: int = 0):
    """Household-level bootstrap. Returns array (overall) or DataFrame (by group)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n_households, n_households), minlength=n_households).astype(float)
        out.append(estimate_lift(cells, w, by))
    return np.array(out) if by is None else pd.DataFrame(out)


def summarize(cells: pd.DataFrame, n_households: int, by: str | None = None, n_boot: int = 200):
    """Point estimate with 95% bootstrap CI (as a small DataFrame)."""
    est = estimate_lift(cells, by=by)
    boot = bootstrap_lift(cells, n_households, by, n_boot)
    if by is None:
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return pd.DataFrame({"lift_per_week": [est], "ci_low": [lo], "ci_high": [hi]}, index=["overall"])
    return pd.DataFrame({"lift_per_week": est, "ci_low": boot.quantile(.025), "ci_high": boot.quantile(.975)})
