"""Churn prediction: point-in-time features and labels at rolling snapshots.

Churn (partial) = spend in the next 8 weeks < 30% of the household's usual 8-week spend,
where "usual" = average over the 26 weeks before the snapshot. Stopping completely counts as churn.

Everything in build_features uses only DAY <= snap_day (asserted) so there is no leakage.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE
from .features import build_rfm, label_segments, log_scale

HIST_DAYS = 26 * 7     # look-back used for "usual" spend
RECENT_DAYS = 8 * 7    # recent window / label horizon
HORIZON_DAYS = 8 * 7
CHURN_RATIO = 0.3

FEATURES = [
    "recency", "recency_ratio", "median_gap", "baskets_8w", "baskets_26w", "spend_8w", "spend_26w",
    "trend_spend", "trend_baskets", "weekly_cv", "avg_basket", "n_commodities", "n_stores",
    "promo_share", "fuel_share", "coupon_uses", "campaigns_received", "tenure_days",
]


def snap_day_for_week(tx: pd.DataFrame, week: int) -> int:
    """Last DAY that belongs to WEEK_NO <= week (WEEK_NO is not DAY/7, so derive from data)."""
    return int(tx.loc[tx["WEEK_NO"] <= week, "DAY"].max())


def eligible(tx: pd.DataFrame, snap_day: int) -> pd.Index:
    """Households with full 26-week history and at least one basket in the last 8 weeks."""
    past = tx[tx["DAY"] <= snap_day]
    g = past.groupby("household_key")["DAY"].agg(["min", "max"])
    ok = (g["min"] <= snap_day - HIST_DAYS) & (g["max"] > snap_day - RECENT_DAYS)
    return g.index[ok]


def build_features(tx: pd.DataFrame, snap_day: int, campaign_table=None, campaign_desc=None) -> pd.DataFrame:
    past = tx[tx["DAY"] <= snap_day]
    assert past["DAY"].max() <= snap_day
    hh = eligible(tx, snap_day)
    past = past[past["household_key"].isin(hh)]
    h26 = past[past["DAY"] > snap_day - HIST_DAYS]
    h8 = past[past["DAY"] > snap_day - RECENT_DAYS]

    f = pd.DataFrame(index=hh)
    days = past[["household_key", "DAY"]].drop_duplicates().sort_values(["household_key", "DAY"])
    gaps = days.groupby("household_key")["DAY"].diff()
    f["median_gap"] = gaps.groupby(days["household_key"]).median().clip(lower=1)
    f["recency"] = snap_day - past.groupby("household_key")["DAY"].max()
    f["recency_ratio"] = f["recency"] / f["median_gap"]
    f["tenure_days"] = snap_day - past.groupby("household_key")["DAY"].min()

    f["baskets_8w"] = h8.groupby("household_key")["BASKET_ID"].nunique()
    f["baskets_26w"] = h26.groupby("household_key")["BASKET_ID"].nunique()
    f["spend_8w"] = h8.groupby("household_key")["SALES_VALUE"].sum()
    f["spend_26w"] = h26.groupby("household_key")["SALES_VALUE"].sum()
    f["trend_spend"] = f["spend_8w"] / (f["spend_26w"] * 8 / 26)
    f["trend_baskets"] = f["baskets_8w"] / (f["baskets_26w"] * 8 / 26)
    f["avg_basket"] = f["spend_26w"] / f["baskets_26w"]

    wk = h26.assign(w=(snap_day - h26["DAY"]) // 7).groupby(["household_key", "w"])["SALES_VALUE"].sum()
    wk = wk.unstack(fill_value=0).reindex(columns=range(26), fill_value=0)
    f["weekly_cv"] = wk.std(axis=1) / wk.mean(axis=1)

    merch = h26[~h26["is_non_merch"]]
    f["n_commodities"] = merch.groupby("household_key")["COMMODITY_DESC"].nunique()
    f["n_stores"] = h26.groupby("household_key")["STORE_ID"].nunique()
    f["promo_share"] = (merch[merch["on_promo"]].groupby("household_key")["SALES_VALUE"].sum()
                        / merch.groupby("household_key")["SALES_VALUE"].sum())
    f["fuel_share"] = (h26[h26["DEPARTMENT"] == "KIOSK-GAS"].groupby("household_key")["SALES_VALUE"].sum()
                       / f["spend_26w"])
    f["coupon_uses"] = h26[h26["used_coupon"]].groupby("household_key")["BASKET_ID"].nunique()

    if campaign_table is not None:
        started = campaign_desc.loc[campaign_desc["START_DAY"] <= snap_day, "CAMPAIGN"]
        f["campaigns_received"] = (campaign_table[campaign_table["CAMPAIGN"].isin(started)]
                                   .groupby("household_key")["CAMPAIGN"].nunique())
    else:
        f["campaigns_received"] = 0
    f = f.fillna(0)
    f["snap_day"] = snap_day
    return f


def build_label(tx: pd.DataFrame, snap_day: int, feats: pd.DataFrame) -> pd.Series:
    """1 = next-8-week spend below 30% of usual 8-week spend (from the 26 weeks before)."""
    fut = tx[(tx["DAY"] > snap_day) & (tx["DAY"] <= snap_day + HORIZON_DAYS)]
    assert snap_day + HORIZON_DAYS <= tx["DAY"].max(), "label window runs past the data"
    spend_next = fut.groupby("household_key")["SALES_VALUE"].sum().reindex(feats.index, fill_value=0)
    usual = feats["spend_26w"] * 8 / 26
    return (spend_next < CHURN_RATIO * usual).astype(int).rename("churn")


def at_risk_as_of(tx: pd.DataFrame, snap_day: int) -> pd.Series:
    """Re-run the notebook-03 segmentation (RFM + K-Means k=4) using data up to snap_day only."""
    rfm = build_rfm(tx, snapshot_day=snap_day + 1)
    X = StandardScaler().fit_transform(log_scale(rfm, ["recency", "frequency", "monetary"]))
    rfm["cluster"] = KMeans(4, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
    return label_segments(rfm)


def precision_at(y, score, frac=0.10):
    """Share of true churners among the top `frac` highest scores."""
    y, score = np.asarray(y), np.asarray(score)
    k = max(1, int(round(len(y) * frac)))
    top = np.argsort(-score, kind="stable")[:k]
    return y[top].mean()


def recall_at(y, score, frac=0.10):
    y, score = np.asarray(y), np.asarray(score)
    k = max(1, int(round(len(y) * frac)))
    top = np.argsort(-score, kind="stable")[:k]
    return y[top].sum() / y.sum()
