"""Loading and cleaning the dunnhumby Complete Journey tables."""
import pandas as pd

from .config import DATA_DIR, PROCESSED_DIR

TABLES = [
    "transaction_data", "hh_demographic", "product", "campaign_table",
    "campaign_desc", "coupon", "coupon_redempt",
]

EXCLUDE_DEPTS = {"KIOSK-GAS", "MISC. TRANS."}  # fuel / non-merchandise, flagged not dropped


def load(name: str, **kwargs) -> pd.DataFrame:
    """Read one raw table from DATA_DIR."""
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put the parquet files in {DATA_DIR} "
            "or set the TESCO_DATA_DIR environment variable."
        )
    return pd.read_parquet(path, **kwargs)


def clean_transactions(tx: pd.DataFrame, product: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning rules from 01_data_preparation and add derived columns."""
    tx = tx[(tx["QUANTITY"] > 0) & (tx["SALES_VALUE"] > 0)].copy()

    tx = tx.merge(
        product[["PRODUCT_ID", "DEPARTMENT", "BRAND", "COMMODITY_DESC"]],
        on="PRODUCT_ID", how="left",
    )
    tx["DEPARTMENT"] = tx["DEPARTMENT"].fillna("UNKNOWN").str.strip()
    tx["is_non_merch"] = tx["DEPARTMENT"].isin(EXCLUDE_DEPTS)

    # Discounts are stored as negative numbers; flip to positive amounts.
    tx["retail_disc"] = tx["RETAIL_DISC"].clip(upper=0).abs()
    tx["coupon_disc"] = tx["COUPON_DISC"].clip(upper=0).abs()
    tx["coupon_match_disc"] = tx["COUPON_MATCH_DISC"].clip(upper=0).abs()
    tx["discount_total"] = tx["retail_disc"] + tx["coupon_disc"] + tx["coupon_match_disc"]
    tx["on_promo"] = tx["retail_disc"] > 0
    tx["used_coupon"] = tx["coupon_disc"] > 0
    tx["hour"] = tx["TRANS_TIME"] // 100
    tx["weekday"] = tx["DAY"] % 7  # day-of-week index (actual calendar weekday unknown)
    return tx


def load_transactions_clean() -> pd.DataFrame:
    """Load the cleaned transactions saved by notebook 01 (fallback: rebuild)."""
    path = PROCESSED_DIR / "transactions_clean.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return clean_transactions(load("transaction_data"), load("product"))
