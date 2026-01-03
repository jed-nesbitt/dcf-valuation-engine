from __future__ import annotations
import numpy as np
import pandas as pd
from src.config import DCFConfig
from src.model import dcf_one, present_value, terminal_value

def dcf_price_with_overrides(base_result: dict, cfg: DCFConfig, wacc: float, terminal_growth: float) -> float:
    """
    Reprice using the already-derived operating projections? We kept it simple:
    we re-run dcf_one would refetch and recompute; that’s slow.
    So here we do a light-weight approximation:
      - Use base_result's stored economics? Not enough.
    For correctness, we re-run dcf_one and then override discounting is not trivial without flows.
    Therefore: sensitivity is computed by re-running dcf_one with same ticker and growth_mult=1,
    and then overriding terminal_growth by temporarily editing cfg would still require refactor.

    Practical approach: do a minimal rerun with a modified cfg (fast enough for small grids).
    """
    raise NotImplementedError("Use sensitivity_grid() which re-runs with modified cfg.")

def sensitivity_grid(symbol: str, cfg: DCFConfig, growth_multiplier: float = 1.0) -> pd.DataFrame:
    """
    Computes a 2D grid of DCF prices for (WACC shift, terminal growth shift).
    Implementation: re-run DCF for each combo but override:
      - wacc used for discounting and terminal calc
      - terminal growth used
    To do that properly, we need access to projected FCFs. So we compute them once here.
    """
    # We re-implement a “flows once, discount many” calculation by calling model internals would be cleaner,
    # but keeping it simple: we copy the flow generation by reusing dcf_one's logic would require refactor.
    # Instead: we do a refactor-light approach by importing yfinance directly would duplicate logic.
    #
    # Best compromise for now: refactor later; for v2, we compute sensitivity by rerunning with patched discounting
    # by extracting wacc and terminal_growth impacts via a wrapper (below).

    from src.model import dcf_one as _dcf_one  # local import to avoid circular
    from src.model import build_growth_path
    import yfinance as yf
    from src.financials import (
        safe_history_close,
        best_effort_cash, best_effort_total_debt,
        best_effort_revenue_series, best_effort_ebit_series, best_effort_da_series,
        best_effort_interest_expense, best_effort_tax_and_pretax,
        best_effort_working_capital_ratio, best_effort_capex_ratio
    )
    from src.market import get_market_premium, fetch_risk_free_rate, cost_of_equity, clamp, calculate_wacc

    # ---- compute operating projections ONCE ----
    t = yf.Ticker(symbol)
    info = t.info or {}
    cashflow = t.cashflow
    balance_sheet = t.balancesheet
    income_statement = t.financials

    current_price = safe_history_close(t)
    shares = info.get("sharesOutstanding")
    if shares is None or float(shares) <= 0:
        raise ValueError("Missing or zero sharesOutstanding")

    # Market
    rf = fetch_risk_free_rate(cfg.region, cfg.fallback_rf)
    market_premium = get_market_premium(cfg.region)
    beta = info.get("beta")
    if beta is None or not np.isfinite(beta):
        raise ValueError("Missing beta")
    market_cap = info.get("marketCap")
    if market_cap is None or float(market_cap) <= 0:
        raise ValueError("Missing or zero marketCap")

    cash = best_effort_cash(balance_sheet)
    total_debt = best_effort_total_debt(balance_sheet)

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

    eq_cost = cost_of_equity(float(beta), rf, market_premium)
    base_wacc = calculate_wacc(total_debt, float(market_cap), pretax_cost_of_debt, eq_cost, tax_rate)

    revenue = best_effort_revenue_series(income_statement).dropna()[::-1].astype(float)
    ebit = best_effort_ebit_series(income_statement).dropna()[::-1].astype(float)
    if len(revenue) < 2 or ebit.empty:
        raise ValueError("Missing revenue/EBIT history")

    base_avg_growth = float(revenue.pct_change().dropna().mean())
    scenario_initial_growth = base_avg_growth * growth_multiplier

    ebit_margin = (ebit / revenue).replace([np.inf, -np.inf], np.nan).dropna()
    ebit_margin_median = float(ebit_margin.tail(min(5, len(ebit_margin))).median())

    da = best_effort_da_series(income_statement).dropna()[::-1].astype(float)
    da_ratio = (da / revenue).replace([np.inf, -np.inf], np.nan).dropna()
    da_ratio_median = float(da_ratio.tail(min(5, len(da_ratio))).median()) if not da_ratio.empty else 0.03

    wc_ratio_median = best_effort_working_capital_ratio(balance_sheet, revenue)
    capex_ratio = best_effort_capex_ratio(cashflow, revenue)

    growth_path = build_growth_path(scenario_initial_growth, cfg.forecast_years, cfg.terminal_growth, cfg.fade_start)

    last_revenue = float(revenue.iloc[-1])
    forecast_revenues = []
    rev = last_revenue
    for g in growth_path:
        rev *= (1 + g)
        forecast_revenues.append(rev)
    forecast_revenues = np.array(forecast_revenues, dtype=float)

    forecast_ebit = forecast_revenues * ebit_margin_median
    forecast_ebiat = forecast_ebit * (1 - tax_rate)
    forecast_da = forecast_revenues * da_ratio_median

    if capex_ratio is not None and np.isfinite(capex_ratio):
        forecast_capex = forecast_revenues * float(capex_ratio)
    else:
        forecast_capex = forecast_da + cfg.fallback_capex_extra_pct_of_rev * forecast_revenues

    rev_series = np.concatenate(([last_revenue], forecast_revenues))
    delta_wc = np.array([wc_ratio_median * (rev_series[i] - rev_series[i - 1]) for i in range(1, len(rev_series))], dtype=float)

    unlevered_fcf = forecast_ebiat + forecast_da - forecast_capex - delta_wc

    # ---- discount grid ----
    rows = []
    for w_bps in cfg.sens_wacc_bps:
        w = base_wacc + (w_bps / 10_000.0)
        for g_bps in cfg.sens_tg_bps:
            tg = cfg.terminal_growth + (g_bps / 10_000.0)

            pv_fcf = present_value(unlevered_fcf, w).sum()
            tv = terminal_value(float(unlevered_fcf[-1]), tg, w)
            tv_pv = tv / (1 + w) ** cfg.terminal_year
            ev = float(pv_fcf + tv_pv)
            equity_value = ev + cash - total_debt
            price = float(equity_value / float(shares))

            rows.append({
                "ticker": symbol,
                "wacc_bps_shift": int(w_bps),
                "terminal_g_bps_shift": int(g_bps),
                "wacc": float(w),
                "terminal_growth": float(tg),
                "dcf_price": price
            })

    return pd.DataFrame(rows)
