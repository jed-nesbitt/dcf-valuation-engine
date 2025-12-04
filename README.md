# Automated Multi-Ticker DCF Valuation Engine (Python)

Disclaimer:
The model is intended for educational and research use.

Some tickers (especially in emerging markets) may have incomplete or inconsistent data on Yahoo Finance; those may need manual data cleaning or should be excluded.

DCF outputs are highly sensitive to assumptions about growth, margins, WACC and terminal value – the scenarios here are meant to illustrate the framework rather than provide investment advice.

The Project

This project is a Python-based discounted cash flow (DCF) valuation engine that retrieves financial statement data from Yahoo Finance via the `yfinance` library and produces intrinsic value estimates for multiple tickers at once.

For each company, the model:

- Downloads income statement, balance sheet and cash flow statement data
- Estimates WACC using CAPM (risk-free rate, beta, market risk premium)
- Forecasts revenue for 10 years with:
  - Historical average growth for years 1–5
  - Linear fade to a 1.5% terminal growth rate by year 10
- Normalises profitability using the **median EBIT margin** of recent years
- Models:
  - D&A as a percentage of revenue
  - CapEx as `D&A + 0.5% of revenue` (maintenance + modest growth capex)
  - Working capital changes as `WC/Sales × ΔRevenue`
- Computes unlevered free cash flow (FCFF), discounts it at WACC
- Calculates terminal value using the Gordon Growth model
- Derives an intrinsic equity value per share

The engine also produces **three valuation scenarios** for each ticker:

- Conservative: 25% lower initial revenue growth
- Base: historical average revenue growth
- Optimistic: 25% higher initial revenue growth

These outputs are designed to resemble a simplified, professional-grade fundamental valuation workflow suitable for junior analyst / trading / research roles.

---

## Files

- `dcf_engine.ipynb` – Main Jupyter Notebook with the full DCF logic and multi-ticker loop.
- `tickers_example.xlsx` – Example input file with a list of tickers (must contain a `Ticker` column).
- `results_example.xlsx` – Example of output produced by the notebook (conservative/base/optimistic prices).
- `requirements.txt` – Python dependencies for reproducing the environment.

---

## Example Output

Example result table (values are illustrative):

| Ticker | Name                     | Current Price | Conservative | Base    | Optimistic |
|--------|--------------------------|---------------|--------------|---------|------------|
| COL.AX | Coles Group Limited      | 21.82         | 16.02        | 17.65   | 19.37      |
| WOW.AX | Woolworths Group Limited | 29.13         | 24.81        | 27.79   | 30.95      |

---

## How to Run

1. Install Python (or use Anaconda).
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
 