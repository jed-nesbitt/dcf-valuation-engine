from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class DCFConfig:
    # Forecast settings
    forecast_years: int = 10
    fade_start: int = 5

    # Terminal assumptions
    terminal_growth: float = 0.015
    terminal_year: int = 10

    # Region / market
    region: str = "AU"

    # Defaults + clamps (industry-ish guardrails)
    default_tax_rate: float = 0.30
    tax_rate_min: float = 0.05
    tax_rate_max: float = 0.35

    default_cost_of_debt: float = 0.06
    cod_min: float = 0.02
    cod_max: float = 0.12

    # If you cannot fetch a proper local RF, use this
    fallback_rf: float = 0.04

    # Capex modelling
    fallback_capex_extra_pct_of_rev: float = 0.005  # if capex not found

    # Scenario multipliers on initial growth
    growth_mult_conservative: float = 0.75
    growth_mult_base: float = 1.00
    growth_mult_optimistic: float = 1.25

    # Sensitivity grid
    sens_wacc_bps: tuple[int, ...] = (-200, -100, 0, 100, 200)  # +/- 2%
    sens_tg_bps: tuple[int, ...] = (-50, -25, 0, 25, 50)        # +/- 0.50%
