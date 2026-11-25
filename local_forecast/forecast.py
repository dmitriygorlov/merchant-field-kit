"""Daily sales forecasts with forward-only evaluation."""
import csv
import io
import math
import random
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
from prophet import Prophet

from .sales import load_sales_csv, sales_history, xyz_analysis

HORIZON = 28
MIN_TRAIN = 56
MIN_HISTORY = 112
MAX_HISTORY = 5 * 366
MODEL = "Prophet"
NAIVE = "Repeat last week"
AVERAGE = "Last four weeks average"
HEADER = ["date", "units_ordered"]


def load_csv(raw: bytes) -> pd.DataFrame:
    if len(raw) > 1_000_000:
        raise ValueError("CSV must be 1 MB or smaller.")
    try:
        reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
        if [cell.strip() for cell in next(reader, [])] != HEADER:
            raise ValueError("Use exactly two columns: date,units_ordered.")
        values = {}
        for line, row in enumerate(reader, 2):
            if not row or all(not x.strip() for x in row):
                continue
            if len(row) != 2:
                raise ValueError(f"Line {line}: expected exactly two columns.")
            day, units = map(str.strip, row)
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
                raise ValueError(f"Line {line}: date must be YYYY-MM-DD.")
            try:
                day = date.fromisoformat(day)
            except ValueError:
                raise ValueError(f"Line {line}: invalid calendar date.") from None
            if not date(1900, 1, 1) <= day <= date(2200, 12, 1):
                raise ValueError(f"Line {line}: date outside the supported range.")
            if not re.fullmatch(r"[0-9]{1,9}", units):
                raise ValueError(f"Line {line}: use a nonnegative whole unit count.")
            if day in values:
                raise ValueError(f"Duplicate day: {day}. Aggregate to one row per day.")
            values[day] = int(units)
    except (UnicodeError, csv.Error):
        raise ValueError("Expected a valid UTF-8 CSV.") from None
    if len(values) < MIN_HISTORY:
        raise ValueError(f"Provide at least {MIN_HISTORY} daily rows (16 weeks).")
    if len(values) > MAX_HISTORY:
        raise ValueError("Use up to five years of history.")
    days = sorted(values)
    for a, b in zip(days, days[1:]):
        if b - a != timedelta(days=1):
            raise ValueError(f"Missing day: {a + timedelta(days=1)}. Use zero only if zero sales are known.")
    return pd.DataFrame({"ds": pd.to_datetime(days), "y": [values[d] for d in days]})


def sample_csv() -> bytes:
    rng = random.Random(42)
    start = date(2025, 1, 6)
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(HEADER)
    for i in range(224):
        units = 34 + 0.19 * i + 0.14 * max(0, i - 95) + [2, -5, -3, 0, 6, 16, 11][i % 7]
        writer.writerow([start + timedelta(days=i), max(0, round(units + rng.gauss(0, 2.7)))])
    return output.getvalue().encode("utf-8")


def sample_sales_csv() -> bytes:
    """Synthetic orders for a two-target forecast and a usable XYZ example."""
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["date", "sku", "units_ordered", "unit_price"])
    for i, (day, units) in enumerate(csv.reader(sample_csv().decode().splitlines()[1:])):
        total = int(units)
        sporadic = total // 2 if i % 17 == 0 or i % 17 == 1 else 0
        medium = total // 3 if i % 21 < 14 else 0
        stable = total - sporadic - medium
        # Split one SKU across two rows to exercise line aggregation.
        writer.writerow([day, "CORE-01", stable // 2, "12.00"])
        writer.writerow([day, "CORE-01", stable - stable // 2, "12.00"])
        if medium:
            writer.writerow([day, "SEASON-02", medium, "18.50"])
        if sporadic:
            writer.writerow([day, "SPOT-03", sporadic, "32.00"])
    return output.getvalue().encode("utf-8")


def predict(train, dates):
    y = train.y.to_numpy(dtype=float)
    model = Prophet(weekly_seasonality=True, daily_seasonality=False,
                    yearly_seasonality=False, uncertainty_samples=0)
    fitted = model.fit(train[["ds", "y"]])
    estimate = fitted.predict(pd.DataFrame({"ds": list(dates)})).yhat.to_numpy(dtype=float)
    if not np.isfinite(estimate).all():
        raise ValueError("Prophet returned an invalid prediction.")
    return {
        MODEL: np.maximum(estimate, 0),
        NAIVE: np.resize(y[-7:], len(dates)),
        AVERAGE: np.full(len(dates), y[-28:].mean()),
    }


def run_forecast(history, predictor=predict, *, metric="units_ordered"):
    if metric not in {"units_ordered", "revenue"}:
        raise ValueError("Choose units_ordered or revenue.")
    mae = "MAE (units/day)" if metric == "units_ordered" else "MAE (currency/day)"
    # Use the most recent 3–5 forecast origins, 14 days apart.
    n = len(history)
    if n < MIN_HISTORY:
        raise ValueError(f"Need {MIN_HISTORY} days to evaluate the models.")
    cutoffs = sorted(n - HORIZON - offset for offset in range(0, 70, 14)
                     if n - HORIZON - offset >= MIN_TRAIN)
    errors = {label: [] for label in (NAIVE, AVERAGE, MODEL)}
    folds = []
    for cutoff in cutoffs:
        train = history.iloc[:cutoff]
        target = history.iloc[cutoff:cutoff + HORIZON]
        guesses = predictor(train, target.ds)
        for label, values in guesses.items():
            err = np.abs(target.y.to_numpy() - values)
            errors[label].extend(err)
            folds.append({"Training through": train.ds.iloc[-1].date().isoformat(),
                          "Test through": target.ds.iloc[-1].date().isoformat(),
                          "Method": label, mae: float(err.mean())})
    scores = {label: math.fsum(values) / len(values) for label, values in errors.items()}
    # Prefer the simpler method if scores are exactly tied up to numerical noise.
    chosen = min(scores, key=lambda label: round(scores[label], 9))
    future = pd.date_range(history.ds.iloc[-1] + pd.Timedelta(days=1), periods=HORIZON)
    estimates = predictor(history, future)
    if metric == "units_ordered":
        values = np.floor(estimates[chosen] + 0.5).astype(int)
    else:
        values = np.round(estimates[chosen], 2)
    forecast = pd.DataFrame({"date": future.strftime("%Y-%m-%d"), f"{metric}_forecast": values})
    return chosen, scores, pd.DataFrame(folds), forecast
