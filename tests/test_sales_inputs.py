from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from local_forecast.forecast import (
    HORIZON, load_sales_csv, run_forecast, sales_history, sample_sales_csv, xyz_analysis,
)


def test_sample_revenue_units_and_all_xyz_classes():
    fixture = Path(__file__).resolve().parents[1] / "examples" / "sample_orders.csv"
    assert fixture.read_bytes() == sample_sales_csv()
    data = load_sales_csv(fixture.read_bytes(), input_kind="orders")
    assert data[["ds", "sku"]].duplicated().sum() == 0
    ref = load_sales_csv((fixture.parent / "sample_shop.csv").read_bytes(), input_kind="daily")
    original_units = sales_history(ref, "units_ordered")
    pd.testing.assert_frame_equal(sales_history(data, "units_ordered"), original_units)
    money = sales_history(data)
    assert money.y.iloc[0] == 870
    _, scores, folds, result = run_forecast(money, metric="revenue")
    assert len(result) == HORIZON and len(scores) == 3
    assert len(folds) == 15 and "MAE (currency/day)" in folds
    assert list(result) == ["date", "revenue_forecast"]
    assert (result.revenue_forecast >= 0).all()
    assert xyz_analysis(data).set_index("sku").xyz_class.to_dict() == {
        "CORE-01": "X", "SEASON-02": "Y", "SPOT-03": "Z",
    }


def test_order_lines_aggregate_and_fill_known_zero_days():
    data = load_sales_csv(
        b"date;sku;units_ordered;unit_price\n"
        b"06.01.2025.;A;2;10,50\n"
        b"2025-01-06T21:55:00-05:00;A;1;9.50\n"
        b"04/30/2025;0008;4;1,00\n"
        b"05/01/2025;0008;1;2,00\n", input_kind="orders")
    assert len(data) == 3
    assert data.sku.tolist() == ["A", "0008", "0008"]
    assert data.units_ordered.tolist() == [3, 4, 1]
    assert data.revenue.tolist() == [30.5, 4, 2]
    history = sales_history(data)
    assert history.y.iloc[1] == 0 and history.y.iloc[-1] == 2
    assert len(history) == (date(2025, 5, 1) - date(2025, 1, 6)).days + 1


@pytest.mark.parametrize(("mode", "expected"), [("dmy", "2025-04-03"), ("mdy", "2025-03-04")])
def test_ambiguous_dates_require_confirmation(mode, expected):
    raw = b"date,revenue\n03/04/2025,10.0\n"
    with pytest.raises(ValueError, match="Ambiguous dates"):
        load_sales_csv(raw)
    assert load_sales_csv(raw, date_order=mode).ds.iloc[0].date().isoformat() == expected


@pytest.mark.parametrize(("raw", "mode", "message"), [
    (b"date,sku,units_ordered\n2025-01-01,A,1\n", "daily", "select order lines"),
    (b"date,units_ordered\n2025-01-01,1\n", "orders", "require a sku column"),
    (b"date,revenue\n31/02/2025,5\n", "daily", "invalid or unsupported date"),
    (b"date,revenue\n21/01/2025,5\n01/22/2025,6\n", "daily", "mix day-first and month-first"),
    (b"date,revenue\n2025-01-01,NaN\n", "daily", "must be nonnegative"),
    (b"date,units_ordered,customer_email\n2025-01-01,1,person@example.org\n", "daily", "Use date and"),
    (b"date,revenue\n2025-01-01,12\n01.01.2025,13\n", "daily", "Duplicate day"),
])
def test_bad_csv_is_rejected(raw, mode, message):
    with pytest.raises(ValueError, match=message):
        load_sales_csv(raw, input_kind=mode)


def test_revenue_only_daily_requires_all_days_but_not_units():
    dates = [date(2025, 1, 6) + timedelta(days=i) for i in range(114)]
    sample = ("date,revenue\n" + "\n".join(f"{d},5.50" for d in dates)).encode()
    data = load_sales_csv(sample)
    assert sales_history(data).y.sum() == pytest.approx(114 * 5.50)
    with pytest.raises(ValueError, match="no units_ordered"):
        sales_history(data, "units_ordered")
    with pytest.raises(ValueError, match="Missing day"):
        sales_history(data.drop(index=20))
    assert xyz_analysis(data).empty


def test_xyz_leaves_new_and_zero_skus_unclassified_and_drops_partial_weeks():
    dates = pd.date_range("2025-01-08", periods=67)
    data = pd.DataFrame(
        [(day, "old", 2) for day in dates] +
        [(day, "new", 2) for day in dates[-15:]] +
        [(day, "zero", 0) for day in dates],
        columns=["ds", "sku", "units_ordered"],
    )
    labels = xyz_analysis(data).set_index("sku")
    assert labels.loc["old", "weeks"] == 8 and labels.loc["old", "xyz_class"] == "X"
    assert pd.isna(labels.loc["new", "xyz_class"])
    assert pd.isna(labels.loc["zero", "xyz_class"])


@pytest.mark.parametrize("value", ["1,234", "1.234", "12,345", "123.456"])
@pytest.mark.parametrize("mode,header,row", [
    ("daily", "date;revenue", "2025-01-06;{}"),
    ("orders", "date;sku;units_ordered;unit_price", "2025-01-06;A;2;{}"),
])
def test_rejects_money_with_ambiguous_separator(value, mode, header, row):
    from local_forecast.sales import AmbiguousMoneyError
    with pytest.raises(AmbiguousMoneyError, match="Line 2: ambiguous"):
        load_sales_csv(f"{header}\n{row.format(value)}\n".encode(), input_kind=mode)


@pytest.mark.parametrize("value,amount", [("1234.00", 1234), ("12,50", 12.5), ("1.2340", 1.234), ("0.125", .125)])
def test_keeps_unambiguous_money(value, amount):
    raw = f"date;sku;units_ordered;unit_price\n2025-01-06;A;2;{value}\n".encode()
    assert load_sales_csv(raw, input_kind="orders").revenue.iloc[0] == pytest.approx(2 * amount)
