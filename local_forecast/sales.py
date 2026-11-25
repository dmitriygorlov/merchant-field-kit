"""Normalize retail exports without guessing ambiguous dates or currencies."""
from __future__ import annotations

import csv
import io
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal

import pandas as pd

MAX_BYTES = 1_000_000
DATE_MIN = date(1900, 1, 1)
DATE_MAX = date(2200, 12, 1)
METRICS = ("units_ordered", "revenue")
COLUMNS = {"date", "sku", "units_ordered", "revenue", "unit_price"}
YEAR_FIRST = re.compile(r"(\d{4})([-/.])(\d{1,2})\2(\d{1,2})(.*)")
SHORT_FIRST = re.compile(r"(\d{1,2})([-/.])(\d{1,2})\2(\d{4})(.*)")
TIME = re.compile(r"(?:T| )\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-]\d{2}:?\d{2})?")
NUMERIC = re.compile(r"[0-9]{1,9}(?:[.,][0-9]{1,4})?")
AMBIGUOUS_MONEY = re.compile(r"[1-9][0-9]{0,2}[.,][0-9]{3}")


class AmbiguousMoneyError(ValueError):
    def __init__(self, line: int, field: str):
        self.line, self.field = line, field
        super().__init__(f"Line {line}: ambiguous {field}; remove thousands separators or write explicit decimals.")


def _calendar(year, month, day, suffix):
    result = date(year, month, day)
    if suffix:
        if not TIME.fullmatch(suffix):
            raise ValueError("unsupported time format")
        # Keep the day written in the export rather than converting time zones.
        datetime.fromisoformat(result.isoformat() + suffix)
    if not DATE_MIN <= result <= DATE_MAX:
        raise ValueError("date outside supported range")
    return result


def _date_options(value):
    match = YEAR_FIRST.fullmatch(value)
    if match:
        year, sep, month, day, suffix = match.groups()
        if sep == "." and suffix == ".":
            suffix = ""
        return {"dmy": _calendar(int(year), int(month), int(day), suffix),
                "mdy": _calendar(int(year), int(month), int(day), suffix)}
    match = SHORT_FIRST.fullmatch(value)
    if not match:
        raise ValueError("unsupported date format")
    first, sep, second, year, suffix = match.groups()
    if sep == "." and suffix == ".":
        suffix = ""
    options = {}
    # Day-first dots are the conventional dotted date representation.
    orders = ("dmy",) if sep == "." else ("dmy", "mdy")
    for order in orders:
        month, day = (second, first) if order == "dmy" else (first, second)
        try:
            options[order] = _calendar(int(year), int(month), int(day), suffix)
        except ValueError:
            continue
    if not options:
        raise ValueError("invalid calendar date")
    if sep == ".":
        # dotted day-first rows do not force the order of slash/dash rows
        return {order: options["dmy"] for order in ("dmy", "mdy")}
    return options


def _money(value, line, name):
    if not NUMERIC.fullmatch(value):
        raise ValueError(f"Line {line}: {name} must be nonnegative; omit currency symbols and thousands separators.")
    if AMBIGUOUS_MONEY.fullmatch(value):
        raise AmbiguousMoneyError(line, name)
    return Decimal(value.replace(",", "."))


def load_sales_csv(raw: bytes, *, input_kind="daily", date_order="auto") -> pd.DataFrame:
    """Return one row per observed day, or one row per observed day and SKU.

    For transactions, only pass a complete export: days without rows within
    the file's first and last days are treated as zero in subsequent analysis.
    Revenue must be in one consistent currency; no conversion is performed.
    """
    if input_kind not in {"daily", "orders"} or date_order not in {"auto", "dmy", "mdy"}:
        raise ValueError("Choose a supported input type and date order.")
    if len(raw) > MAX_BYTES:
        raise ValueError("CSV must be 1 MB or smaller.")
    try:
        source = io.StringIO(raw.decode("utf-8-sig"), newline="")
        heading = source.readline()
        separator = max((",", ";", "\t"), key=heading.count)
        source.seek(0)
        reader = csv.reader(source, delimiter=separator, strict=True)
        headers = [cell.strip().lower() for cell in next(reader, [])]
        fields = set(headers)
        if (len(headers) != len(fields) or fields - COLUMNS or "date" not in fields
                or not fields.intersection(METRICS)):
            raise ValueError("Use date and units_ordered and/or revenue; order lines also need sku.")
        if input_kind == "orders" and "sku" not in fields:
            raise ValueError("Order lines require a sku column.")
        if input_kind == "daily" and fields.intersection({"sku", "unit_price"}):
            raise ValueError("Daily totals cannot include sku or unit_price; select order lines instead.")
        if "unit_price" in fields and ("units_ordered" not in fields or "revenue" in fields):
            raise ValueError("Use unit_price with units_ordered only when revenue is not supplied.")
        records = []
        conventions = {"dmy", "mdy"} if date_order == "auto" else {date_order}
        for row in reader:
            line = reader.line_num
            if not row or all(not item.strip() for item in row):
                continue
            if len(row) != len(headers):
                raise ValueError(f"Line {line}: expected {len(headers)} columns.")
            row = dict(zip(headers, map(str.strip, row)))
            try:
                options = _date_options(row["date"])
            except ValueError:
                raise ValueError(f"Line {line}: invalid or unsupported date.") from None
            conventions &= options.keys()
            if not conventions:
                raise ValueError(f"Line {line}: dates conflict with the chosen date order or mix day-first and month-first.")
            metrics = {}
            if "units_ordered" in fields:
                if not re.fullmatch(r"[0-9]{1,9}", row["units_ordered"]):
                    raise ValueError(f"Line {line}: use a nonnegative whole unit count.")
                metrics["units_ordered"] = int(row["units_ordered"])
            if "revenue" in fields:
                metrics["revenue"] = _money(row["revenue"], line, "revenue")
            elif "unit_price" in fields:
                metrics["revenue"] = _money(row["unit_price"], line, "unit_price") * metrics["units_ordered"]
            sku = row.get("sku")
            if sku is not None and (not sku or len(sku) > 128):
                raise ValueError(f"Line {line}: sku must contain 1–128 characters.")
            records.append((line, options, sku, metrics))
    except (UnicodeError, csv.Error):
        raise ValueError("Expected a valid UTF-8 CSV.") from None
    if not records:
        raise ValueError("CSV contains no data rows.")
    if len(conventions) > 1 and any(len(set(options.values())) > 1 for _, options, _, _ in records):
        raise ValueError("Ambiguous dates such as 03/04/2025; choose day-first or month-first explicitly.")
    convention = min(conventions)
    totals = {}
    for line, options, sku, metrics in records:
        day = options[convention]
        key = (day, sku)
        if key in totals and input_kind == "daily":
            raise ValueError(f"Duplicate day: {day}. Upload order lines or use one daily total per day.")
        bucket = totals.setdefault(key, defaultdict(int))
        for metric, value in metrics.items():
            bucket[metric] += value
    frame = pd.DataFrame([dict(ds=day, **({"sku": sku} if sku is not None else {}),
                               **{k: float(v) if k == "revenue" else v for k, v in metrics.items()})
                          for (day, sku), metrics in sorted(totals.items())])
    if (frame.ds.iloc[-1] - frame.ds.iloc[0]).days + 1 > 5 * 366:
        raise ValueError("Use up to five years of history.")
    frame["ds"] = pd.to_datetime(frame.ds)
    frame.attrs["input_kind"] = input_kind
    return frame


def sales_history(data, metric="revenue") -> pd.DataFrame:
    """Construct a continuous daily forecast series for one available target."""
    from .forecast import MIN_HISTORY

    if metric not in METRICS or metric not in data:
        raise ValueError(f"This CSV has no {metric} values.")
    totals = data.groupby("ds", sort=True)[metric].sum()
    if len(totals) == 0:
        raise ValueError("CSV contains no data rows.")
    calendar = pd.date_range(totals.index[0], totals.index[-1], freq="D", name="ds")
    if len(calendar) < MIN_HISTORY:
        raise ValueError(f"Provide at least {MIN_HISTORY} calendar days of history (16 weeks).")
    if data.attrs.get("input_kind") != "orders":
        missing = calendar.difference(totals.index)
        if len(missing):
            raise ValueError(f"Missing day: {missing[0].date()}. Use zero only if zero sales are known.")
    return totals.reindex(calendar, fill_value=0).rename("y").reset_index()


def xyz_analysis(data) -> pd.DataFrame:
    """Classify variation of weekly ordered units per SKU (not by revenue).

    Use full Monday–Sunday weeks; ignore weeks before the SKU was observed.
    Include zero weeks since, and require >=8 full weeks and positive mean.
    X: CV <=0.5; Y: CV <=1; Z: CV >1. No demand forecast is inferred from XYZ.
    """
    columns = ["sku", "weeks", "mean_weekly_units", "cv", "xyz_class"]
    if "sku" not in data or "units_ordered" not in data or data.empty:
        return pd.DataFrame(columns=columns)
    first, last = data.ds.min(), data.ds.max()
    last_monday = last - pd.Timedelta(days=last.weekday() + (0 if last.weekday() == 6 else 7))
    rows = []
    for sku, group in data.groupby("sku", sort=True):
        active = max(first, group.ds.min())
        start = active + pd.Timedelta(days=(-active.weekday()) % 7)
        calendar = pd.date_range(start, last_monday, freq="W-MON")
        weeks = (
            group.assign(week=group.ds - pd.to_timedelta(group.ds.dt.day_of_week, unit="D"))
            .groupby("week").units_ordered.sum().reindex(calendar, fill_value=0).to_numpy(dtype=float)
        )
        mean = weeks.mean() if len(weeks) else None
        cv = float(weeks.std(ddof=0) / mean) if len(weeks) >= 8 and mean and mean > 0 else None
        label = None if cv is None else ("X" if cv <= 0.5 else "Y" if cv <= 1 else "Z")
        rows.append((sku, len(weeks), mean, cv, label))
    return pd.DataFrame(rows, columns=columns)
