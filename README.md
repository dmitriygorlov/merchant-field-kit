# Merchant Field Kit

**English** · [Русский](docs/README.ru.md) · [Srpski (latinica)](docs/README.sr-Latn.md)

Practical data science for online shops. **How much sales revenue should you expect over the next four weeks?** Forecast revenue for planning and ordered units for stock decisions. Use XYZ to see which products sell steadily and which are harder to predict.

## Run locally

With Docker (no GPU, account or model API):

```sh
docker build -t merchant-field-kit .
docker run --rm -p 127.0.0.1:8501:8501 merchant-field-kit
```

Or with Python 3.12, create and activate a virtual environment, then run:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run forecast_app.py
```

On Windows PowerShell (no activation required):

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m streamlit run forecast_app.py
```

Open [localhost:8501](http://localhost:8501), try the synthetic shop first, then choose revenue, units or both. Download the next 28 days as CSV. Install/build needs internet; forecasting does not. Run locally to keep your CSV on your machine; a hosted Python app would receive the file. Do not include customer data.

## Bring a CSV

| Mode | Columns | Example |
|---|---|---|
| Daily totals | `date`, with `revenue` and/or `units_ordered` | [Daily CSV](examples/sample_shop.csv) |
| Order lines | `date`, `sku`, with `revenue` and/or `units_ordered`. With units, `unit_price` can be used instead of `revenue`. | [Order-line CSV](examples/sample_orders.csv) |

Use UTF-8, a file up to 1 MB, and at least **112 calendar days** for forecasting. Daily files need consecutive dates, with explicit zeros for known no-sale days. Use order-line mode only for a complete export: days without rows within its date span count as zero. No customer or order ID is required. Use one currency and a consistent revenue definition, such as item sales after discounts and before tax; don't mix currencies, sales and profit, or gross and net amounts. Normalize returns separately before import. Use store-local dates. ISO dates are safest; if a numeric date could mean either day-first or month-first, set the order explicitly.

Use decimal point or decimal comma, without thousands grouping. Values such as `1,234` or `1.234` are rejected as ambiguous: write `1234.00` for 1,234, or `1.2340` if you mean a value near one.

## Read the result

Prophet competes with repeating the last seven days and the last 28 days’ average. Models are tested on the same historical 28-day windows, using only data before each cutoff. Compare mean absolute error in currency/day or units/day for the chosen target. A revenue forecast is not a profit forecast; a units forecast is not a purchase order. Promotions, stockouts and changes in assortment can affect both.

XYZ requires order lines with both a SKU and ordered units. It compares weekly units per SKU over complete Monday–Sunday weeks. X means steadier, Y more variable, Z most variable; it does not rank products by revenue. A product without enough history is left unclassified.

[Past benchmark on a real retailer](docs/benchmark.md) · [Per-window scores](docs/benchmark_folds.csv). This benchmark is for **ordered units**, not evidence of revenue accuracy or sales lift.

Run checks with `python -m pip install -r requirements-dev.txt` and `python -m pytest -q`.
