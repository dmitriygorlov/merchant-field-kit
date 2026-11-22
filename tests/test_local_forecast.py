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
                            default_timeout=25).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert app.subheader[0].value == "Forecast method: Prophet"
    assert app.get("download_button")[1].label == "Download forecast CSV"
    app.radio[0].set_value("My daily totals").run()
    assert not app.exception
    assert app.button[0].disabled
    assert not app.metric
