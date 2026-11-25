from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from local_forecast.forecast import (
    AVERAGE, HORIZON, MODEL, NAIVE, load_csv, run_forecast, sample_csv,
)


def test_sample_uses_real_prophet_and_forecasts_future():
    sample_path = Path(__file__).resolve().parents[1] / "examples" / "sample_shop.csv"
    assert sample_path.read_bytes() == sample_csv()
    history = load_csv(sample_path.read_bytes())
    winner, scores, folds, forecast = run_forecast(history)
    assert winner == MODEL
    assert scores[MODEL] < scores[NAIVE] and scores[MODEL] < scores[AVERAGE]
    assert len(folds) == 15
    assert len(forecast) == HORIZON
    assert forecast.date.iloc[0] == (history.ds.iloc[-1] + pd.Timedelta(days=1)).date().isoformat()
    assert (forecast.units_ordered_forecast >= 0).all()
    csv_lines = forecast.to_csv(index=False).splitlines()
    assert csv_lines[0] == "date,units_ordered_forecast"
    assert len(csv_lines) == HORIZON + 1


@pytest.mark.parametrize(("change", "message"), [
    (lambda rows: [*rows[:80], *rows[81:]], "Missing day"),
    (lambda rows: [*rows, rows[10]], "Duplicate day"),
    (lambda rows: [*rows[:2], rows[2].split(",")[0] + ",-1", *rows[3:]], "nonnegative"),
    (lambda rows: rows[:80], "112 daily rows"),
    (lambda rows: ["date,customer_email", *rows[1:]], "exactly two columns"),
])
def test_validates_csv(change, message):
    rows = sample_csv().decode().splitlines()
    with pytest.raises(ValueError, match=message):
        load_csv("\n".join(change(rows)).encode())


def test_each_method_compares_identical_unseen_days_and_ties_go_to_baseline():
    history = load_csv(sample_csv()).iloc[:112]
    calls = []

    def predictor(train, dates):
        assert train.ds.iloc[-1] < min(dates)
        calls.append((len(train), len(dates)))
        return {label: np.zeros(len(dates)) for label in (MODEL, AVERAGE, NAIVE)}

    winner, _, folds, _ = run_forecast(history, predictor)
    assert winner == NAIVE
    assert calls == [(56, 28), (70, 28), (84, 28), (112, 28)]
    assert set(folds.groupby("Training through").size()) == {3}


def test_zero_demand_stays_zero():
    history = load_csv(sample_csv()).iloc[:112].copy()
    history["y"] = 0
    winner, scores, _, forecast = run_forecast(history)
    assert winner == NAIVE and all(score == 0 for score in scores.values())
    assert forecast.units_ordered_forecast.sum() == 0


def test_streamlit_sample_flow():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "forecast_app.py"),
                            default_timeout=45).run()
    assert not app.exception
    assert app.radio[1].value == "revenue"
    app.radio[1].set_value("both").run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert app.subheader[0].value.startswith("Revenue · Method:")
    assert app.subheader[1].value.startswith("Ordered units · Method:")
    assert app.subheader[2].value == "XYZ: sales stability by SKU"
    assert len(app.metric) == 2
    assert app.get("download_button")[1].label == "Download forecast CSV"
    for code, title in (("ru", "Спланируйте продажи на четыре недели"),
                        ("sr", "Planirajte prodaju za naredne četiri nedelje")):
        app.selectbox[0].set_value(code).run()
        assert not app.exception and not app.error
        assert app.title[0].value == title
        assert len(app.metric) == 2
    app.radio[0].set_value("daily").run()
    assert not app.exception
    assert app.button[0].disabled
    assert not app.metric


def test_streamlit_order_upload_and_ambiguous_date_override():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "forecast_app.py"),
                            default_timeout=45).run()
    app.radio[0].set_value("orders").run()
    fixture = Path(__file__).resolve().parents[1] / "examples" / "sample_orders.csv"
    raw = fixture.read_bytes().replace(b"2025-01-06", b"01/06/2025")
    app.get("file_uploader")[0].upload("order_lines.csv", raw, "text/csv").run()
    assert not app.button[0].disabled
    app.button[0].click().run()
    assert app.error[0].value == "Choose day first or month first for ambiguous dates."
    app.selectbox[1].set_value("mdy").run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert len(app.metric) == 1
    assert app.subheader[1].value == "XYZ: sales stability by SKU"
    assert app.get("download_button")[0].label == "Download forecast CSV"


def test_streamlit_legacy_daily_upload_select_units():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "forecast_app.py"),
                            default_timeout=45).run()
    app.radio[0].set_value("daily").run()
    app.get("file_uploader")[0].upload("daily.csv", sample_csv(), "text/csv").run()
    app.button[0].click().run()
    assert app.error[0].value == "This file does not contain the selected target."
    app.radio[1].set_value("units_ordered").run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert app.subheader[0].value == "Ordered units · Method: Prophet"
    assert len(app.subheader) == 1
