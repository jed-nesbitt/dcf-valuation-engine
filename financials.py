from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf

def safe_history_close(ticker: yf.Ticker) -> float:
    h = ticker.history(period="5d")["Close"].dropna()
    if h.empty:
        raise ValueError("Missing price history")
    return float(h.iloc[-1])

def get_row(df: pd.DataFrame | None, labels: list[str]) -> float:
    """
    Try multiple possible row names. Return np.nan if none found.
    """
    if df is None or df.empty:
        return float("nan")
    for lab in labels:
        if lab in df.index:
            s = df.loc[lab].dropna()
            if not s.empty:
                return float(s.iloc[0])
    return float("nan")

def get_series(df: pd.DataFrame | None, label: str) -> pd.Series:
    """
    Get a time series for one row label (if present). Return empty series otherwise.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if label not in df.index:
        return pd.Series(dtype=float)
    s = df.loc[label].dropna()
    return s

def best_effort_cash(balance_sheet: pd.DataFrame | None) -> float:
    cash = get_row(balance_sheet, [
        "Cash And Cash Equivalents",
        "Cash",
        "Cash And Short Term Investments"
    ])
    return 0.0 if not np.isfinite(cash) else float(cash)

def best_effort_total_debt(balance_sheet: pd.DataFrame | None) -> float:
    td = get_row(balance_sheet, ["Total Debt"])
    if np.isfinite(td):
        return float(td)

    ltd = get_row(balance_sheet, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
    std = get_row(balance_sheet, ["Short Long Term Debt", "Short Term Debt", "Current Debt"])
    ltd = 0.0 if not np.isfinite(ltd) else float(ltd)
    std = 0.0 if not np.isfinite(std) else float(std)
    return float(ltd + std)

def best_effort_revenue_series(income_statement: pd.DataFrame | None) -> pd.Series:
    # Yahoo: financials columns often latest->oldest; we will reverse later
    s = get_series(income_statement, "Total Revenue")
    return s

def best_effort_ebit_series(income_statement: pd.DataFrame | None) -> pd.Series:
    # EBIT may be missing; could use Operating Income
    ebit = get_series(income_statement, "EBIT")
    if not ebit.empty:
        return ebit
    op = get_series(income_statement, "Operating Income")
    return op

def best_effort_da_series(income_statement: pd.DataFrame | None) -> pd.Series:
    # Your original used "Reconciled Depreciation"
    da = get_series(income_statement, "Reconciled Depreciation")
    if not da.empty:
        return da
    da2 = get_series(income_statement, "Depreciation And Amortization")
    return da2

def best_effort_interest_expense(income_statement: pd.DataFrame | None) -> float:
    ie = get_row(income_statement, ["Interest Expense", "InterestExpense"])
    return float(abs(ie)) if np.isfinite(ie) else float("nan")

def best_effort_tax_and_pretax(income_statement: pd.DataFrame | None) -> tuple[float, float]:
    tax = get_row(income_statement, ["Tax Provision", "Income Tax Expense", "IncomeTaxExpense"])
    pretax = get_row(income_statement, ["Pretax Income", "Income Before Tax", "IncomeBeforeTax"])
    tax = float(abs(tax)) if np.isfinite(tax) else float("nan")
    pretax = float(pretax) if np.isfinite(pretax) else float("nan")
    return tax, pretax

def best_effort_working_capital_ratio(balance_sheet: pd.DataFrame | None, revenue: pd.Series) -> float:
    """
    Prefer 'Working Capital' row if available; fallback to (Current Assets - Current Liabilities).
    Return median WC/Revenue over available years.
    """
    if balance_sheet is None or balance_sheet.empty or revenue is None or revenue.empty:
        return 0.0

    wc = get_series(balance_sheet, "Working Capital")
    if wc.empty:
        ca = get_series(balance_sheet, "Current Assets")
        cl = get_series(balance_sheet, "Current Liabilities")
        if not ca.empty and not cl.empty:
            wc = ca - cl

    if wc.empty:
        return 0.0

    # Align on common columns and compute ratio
    common = wc.index.intersection(revenue.index)
    if len(common) == 0:
        return 0.0

    wc2 = wc.loc[common].astype(float)
    rev2 = revenue.loc[common].astype(float)

    ratio = (wc2 / rev2).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return 0.0

    return float(ratio.tail(min(5, len(ratio))).median())

def best_effort_capex_ratio(cashflow: pd.DataFrame | None, revenue: pd.Series) -> float | None:
    """
    Estimate historical CapEx as % of revenue if 'Capital Expenditure' exists.
    Yahoo often stores CapEx as negative (cash outflow). We'll take abs().
    Return median of last up to 5 periods.
    """
    if cashflow is None or cashflow.empty or revenue is None or revenue.empty:
        return None

    capex = get_series(cashflow, "Capital Expenditure")
    if capex.empty:
        capex = get_series(cashflow, "CapitalExpenditures")

    if capex.empty:
        return None

    common = capex.index.intersection(revenue.index)
    if len(common) == 0:
        return None

    capex2 = capex.loc[common].astype(float).abs()
    rev2 = revenue.loc[common].astype(float)

    ratio = (capex2 / rev2).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return None

    return float(ratio.tail(min(5, len(ratio))).median())
