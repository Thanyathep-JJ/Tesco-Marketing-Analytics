"""Household-level features: RFM, RFM scores and CLV."""
import numpy as np
import pandas as pd

from .config import SNAPSHOT_DAY, MAX_DAY


def build_rfm(tx: pd.DataFrame, snapshot_day: int = SNAPSHOT_DAY) -> pd.DataFrame:
    """One row per household with Recency, Frequency, Monetary and tenure.

    Pass `snapshot_day` (= last observed day + 1) together with a truncated `tx`
    to compute RFM "as of" an earlier date, e.g. for forward-looking validation.
    """
    tx = tx[tx["DAY"] < snapshot_day]
    rfm = tx.groupby("household_key").agg(
        first_day=("DAY", "min"),
        last_day=("DAY", "max"),
        frequency=("BASKET_ID", "nunique"),
        monetary=("SALES_VALUE", "sum"),
    )
    rfm["recency"] = snapshot_day - rfm["last_day"]
    rfm["tenure_weeks"] = (snapshot_day - rfm["first_day"]) / 7
    rfm["baskets_per_week"] = rfm["frequency"] / rfm["tenure_weeks"]
    rfm["avg_basket"] = rfm["monetary"] / rfm["frequency"]
    return rfm[["recency", "frequency", "monetary", "baskets_per_week",
                "avg_basket", "tenure_weeks", "first_day", "last_day"]]


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Quintile scores 1-5 (5 = best). Recency is reversed: recent = 5."""
    out = rfm.copy()
    out["R_score"] = pd.qcut(out["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    out["F_score"] = pd.qcut(out["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["M_score"] = pd.qcut(out["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["RFM_score"] = out["R_score"] + out["F_score"] + out["M_score"]
    return out


def build_clv(tx: pd.DataFrame, recent_weeks: int = 26) -> pd.DataFrame:
    """Revenue-based CLV (no margin data available).

    - clv_hist      : total sales over the 2-year window
    - weekly_recent : average weekly spend over the last `recent_weeks` weeks
    - clv_12m       : projected next-12-month revenue = weekly_recent * 52
    """
    cutoff = MAX_DAY - recent_weeks * 7
    hist = tx.groupby("household_key")["SALES_VALUE"].sum().rename("clv_hist")
    recent = (tx[tx["DAY"] > cutoff].groupby("household_key")["SALES_VALUE"].sum()
              / recent_weeks).rename("weekly_recent")
    clv = pd.concat([hist, recent], axis=1).fillna({"weekly_recent": 0.0})
    clv["clv_12m"] = clv["weekly_recent"] * 52
    return clv


def label_segments(rfm: pd.DataFrame, cluster_col: str = "cluster") -> pd.Series:
    """Map 4 K-Means cluster ids to business names using cluster medians.

    - Champions : highest median monetary
    - Occasional: lowest median frequency
    - At Risk   : of the remaining two, the one with the longer recency
    - Loyal     : the other one
    """
    med = rfm.groupby(cluster_col)[["recency", "frequency", "monetary"]].median()
    champ = med["monetary"].idxmax()
    occ = med["frequency"].idxmin()
    rest = med.drop(index=[champ, occ])
    risk = rest["recency"].idxmax()
    loyal = rest["recency"].idxmin()
    names = {champ: "Champions", loyal: "Loyal", risk: "At Risk", occ: "Occasional"}
    return rfm[cluster_col].map(names)


def log_scale(df: pd.DataFrame, cols) -> np.ndarray:
    """log1p transform used before K-Means (RFM is heavily right-skewed)."""
    return np.log1p(df[list(cols)].to_numpy(dtype=float))
