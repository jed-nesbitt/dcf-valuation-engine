from __future__ import annotations
import pandas as pd

from config import DCFConfig
from model import dcf_one
from sensitivity import sensitivity_grid
from io_utils import read_tickers, to_powerbi_long

TICKER_FILE = "input/tickers.csv"

def run() -> None:
    cfg = DCFConfig(region="AU")

    tickers_df = read_tickers(TICKER_FILE)
    if "Ticker" not in tickers_df.columns:
        raise ValueError("Input file must contain a 'Ticker' column")

    scenario_rows = []
    base_detail_rows = []
    sens_rows = []

    for raw in tickers_df["Ticker"]:
        symbol = str(raw).strip()
        if not symbol:
            continue

        try:
            # --- scenarios (growth multiplier only) ---
            cons = dcf_one(symbol, cfg, growth_multiplier=cfg.growth_mult_conservative)
            base = dcf_one(symbol, cfg, growth_multiplier=cfg.growth_mult_base)
            opt  = dcf_one(symbol, cfg, growth_multiplier=cfg.growth_mult_optimistic)

            scenario_rows.append({
                "ticker": symbol,
                "name": base["name"],
                "current_price": base["current_price"],
                "conservative_price": cons["dcf_price"],
                "base_price": base["dcf_price"],
                "optimistic_price": opt["dcf_price"],
                "wacc": base["wacc"],
                "rf": base["rf"],
                "tax_rate": base["tax_rate"],
            })

            base_detail_rows.append(base)

            # --- sensitivity (base scenario) ---
            sens_df = sensitivity_grid(symbol, cfg, growth_multiplier=cfg.growth_mult_base)
            sens_rows.append(sens_df)

            print(f"Done: {symbol}")

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    scenarios_df = pd.DataFrame(scenario_rows)
    base_details_df = pd.DataFrame(base_detail_rows)

    scenarios_df.to_csv("output/dcf_results_scenarios_wide.csv", index=False)

    # Power BI-friendly long format
    scenarios_long = to_powerbi_long(scenarios_df)
    scenarios_long.to_csv("output/dcf_results_scenarios_long.csv", index=False)

    base_details_df.to_csv("output/dcf_results_base_details.csv", index=False)

    if sens_rows:
        sensitivity_all = pd.concat(sens_rows, ignore_index=True)
        sensitivity_all.to_csv("output/dcf_results_sensitivity_long.csv", index=False)

    print("Saved:")
    print("- dcf_results_scenarios_wide.csv")
    print("- dcf_results_scenarios_long.csv  (Power BI)")
    print("- dcf_results_base_details.csv")
    print("- dcf_results_sensitivity_long.csv (Power BI)")

if __name__ == "__main__":
    run()
