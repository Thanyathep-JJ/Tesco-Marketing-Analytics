"""Market basket analysis at COMMODITY level (association rules via FP-Growth)."""
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth


def basket_matrix(tx: pd.DataFrame, item_col: str = "COMMODITY_DESC") -> pd.DataFrame:
    """One-hot basket x item matrix (bool). Fuel / misc. transactions are excluded."""
    m = tx[~tx["is_non_merch"] & tx[item_col].notna()]
    pairs = m[["BASKET_ID", item_col]].drop_duplicates()
    codes, items = pd.factorize(pairs[item_col], sort=True)
    rows, baskets = pd.factorize(pairs["BASKET_ID"])
    arr = np.zeros((len(baskets), len(items)), dtype=bool)
    arr[rows, codes] = True
    return pd.DataFrame(arr, index=pd.Index(baskets, name="BASKET_ID"), columns=items)


def mine_rules(onehot: pd.DataFrame, min_support=0.01, max_len=2,
               min_lift=1.5, min_rule_support=0.005, min_confidence=0.2) -> pd.DataFrame:
    """FP-Growth (same itemsets as Apriori, faster) + association rules, filtered."""
    keep = onehot.columns[onehot.mean() >= min_support]
    itemsets = fpgrowth(onehot[keep], min_support=min_support, use_colnames=True, max_len=max_len)
    rules = association_rules(itemsets, metric="lift", min_threshold=1.0)
    rules = rules[(rules["lift"] >= min_lift) & (rules["support"] >= min_rule_support)
                  & (rules["confidence"] >= min_confidence)].copy()
    rules["antecedent"] = rules["antecedents"].apply(lambda s: " + ".join(sorted(s)))
    rules["consequent"] = rules["consequents"].apply(lambda s: " + ".join(sorted(s)))
    cols = ["antecedent", "consequent", "antecedent support", "consequent support",
            "support", "confidence", "lift"]
    return rules[cols].sort_values("lift", ascending=False).reset_index(drop=True)


def rule_metrics(onehot: pd.DataFrame, a: str, c: str) -> dict:
    """Support / confidence / lift of a single-item rule a -> c on another basket set."""
    A, C = onehot[a].to_numpy(), onehot[c].to_numpy()
    sa, sc, sac = A.mean(), C.mean(), (A & C).mean()
    return {"support": sac, "confidence": sac / sa if sa else np.nan,
            "lift": sac / (sa * sc) if sa and sc else np.nan}


def pair_table(rules: pd.DataFrame) -> pd.DataFrame:
    """Collapse A->B and B->A into one row per unordered pair (keep the higher-confidence direction)."""
    r = rules[~rules["antecedent"].str.contains(r" \+ ") & ~rules["consequent"].str.contains(r" \+ ")].copy()
    r["pair"] = r.apply(lambda x: " | ".join(sorted([x["antecedent"], x["consequent"]])), axis=1)
    return (r.sort_values("confidence", ascending=False)
             .drop_duplicates("pair").sort_values("lift", ascending=False).reset_index(drop=True))


def cross_sell_opportunity(tx: pd.DataFrame, onehot: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Per rule A->B: baskets with A but not B, and average spend on B when it is bought."""
    m = tx[~tx["is_non_merch"]]
    spend_b = m.groupby(["BASKET_ID", "COMMODITY_DESC"])["SALES_VALUE"].sum()
    avg_b = spend_b.groupby(level="COMMODITY_DESC").mean()
    out = pairs.copy()
    out["baskets_A_not_B"] = [int((onehot[a] & ~onehot[c]).sum())
                              for a, c in zip(out["antecedent"], out["consequent"])]
    out["avg_spend_B"] = out["consequent"].map(avg_b)
    return out
