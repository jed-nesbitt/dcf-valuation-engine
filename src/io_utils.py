from __future__ import annotations
import pandas as pd

def read_tickers(path: str) -> pd.DataFrame:
    if path.lower().endswith(".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)

def to_powerbi_long(results_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Convert scenario wide columns into long format for Power BI:
      ticker | scenario | dcf_price
    """
    id_cols = [c for c in results_wide.columns if c not in ("conservative_price", "base_price", "optimistic_price")]
    long_df = results_wide.melt(
        id_vars=id_cols,
        value_vars=["conservative_price", "base_price", "optimistic_price"],
        var_name="scenario",
        value_name="dcf_price"
    )
    return long_df
