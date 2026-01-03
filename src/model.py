from __future__ import annotations
import numpy as np
import yfinance as yf

from src.config import DCFConfig
from src.market import (
    get_market_premium, fetch_risk_free_rate,
    cost_of_equity, clamp, calculate_wacc
)
from src.financials import (
    safe_history_close,
    best_effort_cash, best_effort_total_debt,
    best_effort_revenue_series, best_effort_ebit_series, best_effort_da_series,
    best_effort_interest_expense, best_effort_tax_and_pretax,
    best_effort_working_capital_ratio, best_effort_capex_ratio
)

def present_value(cash_flows: np.ndarray, discount_rate: float) -> np.ndarray:
    cash_flows = np.asarray(cash_flows, dtype=float)
    return np.array([cf / (1 + discount_rate) ** t for t, cf in enumerate(cash_flows, start=1)], dtype=float)

def terminal_value(last_fcf: float, g: float, r: float) -> float:
    # ensure r > g
    if r <= g:
        r = g + 0.01
    return float((last_fcf * (1 + g)) / (r - g))

def build_growth_path(initial_growth: float, years: int, terminal_g: float, fade_start: int) -> np.ndarray:
    growth_rates = []
    for year in range(1, years + 1):
        if year <= fade_start:
            g = initial_growth
        else:
            t = year - fade_start
            total_fade_years = max(years - fade_start, 1)
            g = initial_growth + (terminal_g - initial_growth) * (t / total_fade_years)
        growth_rates.append(g)
    return np.array(growth_rates, dtype=float)

def dcf_one(
    symbol: str,
    cfg: DCFConfig,
    growth_multiplier: float = 1.0
) -> dict:
    t = yf.Ticker(symbol)

    info = t.info or {}
    cashflow = t.cashflow
    balance_sheet = t.balancesheet
    income_statement = t.financials

    # Price & shares
    current_price = safe_history_close(t)
    shares = info.get("sharesOutstanding")
    if shares is None or float(shares) <= 0:
        raise ValueError("Missing or zero sharesOutstanding")

    name = info.get("longName") or info.get("shortName") or symbol

    # Market assumptions
    rf = fetch_risk_free_rate(cfg.region, cfg.fallback_rf)
    market_premium = get_market_premium(cfg.region)

    beta = info.get("beta")
    if beta is None or not np.isfinite(beta):
        raise ValueError("Missing beta")

    market_cap = info.get("marketCap")
    if market_cap is None or float(market_cap) <= 0:
        raise ValueError("Missing or zero marketCap")

    # Balance sheet inputs
    cash = best_effort_cash(balance_sheet)
    total_debt = best_effort_total_debt(balance_sheet)

    # Tax rate + cost of debt with guards
    interest_expense = best_effort_interest_expense(income_statement)
    tax_expense, pretax_income = best_effort_tax_and_pretax(income_statement)

    raw_tax = np.nan
    if np.isfinite(tax_expense) and np.isfinite(pretax_income) and pretax_income > 0:
        raw_tax = tax_expense / pretax_income
    tax_rate = clamp(raw_tax, cfg.tax_rate_min, cfg.tax_rate_max, cfg.default_tax_rate)

    raw_cod = np.nan
    if np.isfinite(interest_expense) and total_debt > 0:
        raw_cod = float(interest_expense / total_debt)
    pretax_cost_of_debt = clamp(raw_cod, cfg.cod_min, cfg.cod_max, cfg.default_cost_of_debt)

    # WACC
    eq_cost = cost_of_equity(float(beta), rf, market_premium)
    wacc = calculate_wacc(total_debt, float(market_cap), pretax_cost_of_debt, eq_cost, tax_rate)

    # Historical series
    revenue = best_effort_revenue_series(income_statement)
    ebit = best_effort_ebit_series(income_statement)

    if revenue.empty or ebit.empty:
        raise ValueError("Missing revenue/EBIT series")

    # reverse to oldest -> newest
    revenue = revenue.dropna()[::-1].astype(float)
    ebit = ebit.dropna()[::-1].astype(float)

    if len(revenue) < 2:
        raise ValueError("Not enough revenue history")

    revenue_growth = revenue.pct_change().dropna()
    base_avg_growth = float(revenue_growth.mean())
    scenario_initial_growth = base_avg_growth * growth_multiplier

    ebit_margin = (ebit / revenue).replace([np.inf, -np.inf], np.nan).dropna()
    if ebit_margin.empty:
        raise ValueError("Cannot compute EBIT margin")
    ebit_margin_median = float(ebit_margin.tail(min(5, len(ebit_margin))).median())

    da = best_effort_da_series(income_statement).dropna()[::-1].astype(float)
    da_ratio = (da / revenue).replace([np.inf, -np.inf], np.nan).dropna()
    da_ratio_median = float(da_ratio.tail(min(5, len(da_ratio))).median()) if not da_ratio.empty else 0.03

    wc_ratio_median = best_effort_working_capital_ratio(balance_sheet, revenue)

    # Capex ratio from history if possible
    capex_ratio = best_effort_capex_ratio(cashflow, revenue)

    # Forecast revenue with faded growth
    growth_path = build_growth_path(
        initial_growth=scenario_initial_growth,
        years=cfg.forecast_years,
        terminal_g=cfg.terminal_growth,
        fade_start=cfg.fade_start
    )

    last_revenue = float(revenue.iloc[-1])
    forecast_revenues = []
    rev = last_revenue
    for g in growth_path:
        rev *= (1 + g)
        forecast_revenues.append(rev)
    forecast_revenues = np.array(forecast_revenues, dtype=float)

    # Forecast operating items
    forecast_ebit = forecast_revenues * ebit_margin_median
    forecast_ebiat = forecast_ebit * (1 - tax_rate)

    forecast_da = forecast_revenues * da_ratio_median

    if capex_ratio is not None and np.isfinite(capex_ratio):
        forecast_capex = forecast_revenues * float(capex_ratio)
    else:
        # fallback heuristic: D&A plus small growth reinvestment
        forecast_capex = forecast_da + cfg.fallback_capex_extra_pct_of_rev * forecast_revenues

    # ΔWC = (WC/Rev) * ΔRev
    rev_series = np.concatenate(([last_revenue], forecast_revenues))
    delta_wc = np.array([wc_ratio_median * (rev_series[i] - rev_series[i - 1]) for i in range(1, len(rev_series))], dtype=float)

    unlevered_fcf = forecast_ebiat + forecast_da - forecast_capex - delta_wc

    # PV + Terminal
    pv_fcf = present_value(unlevered_fcf, wacc)
    tv = terminal_value(float(unlevered_fcf[-1]), cfg.terminal_growth, wacc)
    tv_pv = tv / (1 + wacc) ** cfg.terminal_year

    ev = float(pv_fcf.sum() + tv_pv)
    equity_value = ev + cash - total_debt
    dcf_price = float(equity_value / float(shares))

    return {
        "ticker": symbol,
        "name": name,
        "current_price": float(current_price),
        "dcf_price": dcf_price,
        "wacc": float(wacc),
        "rf": float(rf),
        "market_premium": float(market_premium),
        "tax_rate": float(tax_rate),
        "pretax_cost_of_debt": float(pretax_cost_of_debt),
        "base_avg_growth": float(base_avg_growth),
        "scenario_initial_growth": float(scenario_initial_growth),
        "ebit_margin_median": float(ebit_margin_median),
        "da_ratio_median": float(da_ratio_median),
        "wc_ratio_median": float(wc_ratio_median),
        "capex_ratio_used": float(capex_ratio) if capex_ratio is not None and np.isfinite(capex_ratio) else np.nan
    }
