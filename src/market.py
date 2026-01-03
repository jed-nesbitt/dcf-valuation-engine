from __future__ import annotations
import numpy as np
import yfinance as yf

def get_market_premium(region: str = "AU") -> float:
    r = region.upper()
    if r == "AU":
        return 0.06
    if r == "US":
        return 0.055
    return 0.06

def fetch_risk_free_rate(region: str, fallback: float) -> float:
    """
    Yahoo Finance has inconsistent tickers for non-US gov yields.
    So we:
      - Try US ^TNX only if region == US
      - Otherwise use fallback
    You can later extend this to use a proper AU 10Y series if you have one.
    """
    r = region.upper()
    if r == "US":
        try:
            ten_year = yf.Ticker("^TNX")
            rf_raw = ten_year.history(period="5d")["Close"].dropna().iloc[-1]
            rf = float(rf_raw / 100.0)
            if np.isfinite(rf) and 0.0 < rf < 0.20:
                return rf
        except Exception:
            pass

    # Non-US: default to fallback unless you plug in a local series
    return float(fallback)

def cost_of_equity(beta: float, risk_free_rate: float, market_premium: float) -> float:
    return float(risk_free_rate + beta * market_premium)

def clamp(x: float | None, lo: float, hi: float, default: float) -> float:
    if x is None:
        return float(default)
    try:
        x = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return float(min(max(x, lo), hi))

def calculate_wacc(
    total_debt: float,
    market_cap: float,
    pretax_cost_of_debt: float,
    equity_cost: float,
    tax_rate: float
) -> float:
    total_debt = max(float(total_debt), 0.0)
    market_cap = max(float(market_cap), 0.0)

    if total_debt <= 0.0 or market_cap <= 0.0:
        return float(equity_cost)

    wd = total_debt / (total_debt + market_cap)
    return float((1 - wd) * equity_cost + wd * pretax_cost_of_debt * (1 - tax_rate))
