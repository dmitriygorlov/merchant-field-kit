"""Local, CSV-in / forecast-out prototype."""
import pandas as pd
import streamlit as st

from local_forecast.forecast import load_csv, run_forecast, sample_csv

st.set_page_config(page_title="Merchant Field Kit | Sales Forecast", page_icon="📦")
st.title("Plan the next four weeks")
st.write("Forecast daily ordered units, compare methods on past sales, and download a CSV.")
st.caption("This local app processes files on the machine running it. It does not call a model API.")
mode = st.radio("Data source", ["Sample shop", "My daily totals"], horizontal=True)
upload = None
if mode == "My daily totals":
    upload = st.file_uploader("Daily totals CSV", type=["csv"],
                              help="Use date,units_ordered and at least 112 consecutive days.")
else:
    st.download_button("Get sample CSV", sample_csv(), "sample_shop.csv", "text/csv")
with st.expander("What goes in the CSV?"):
    st.code("date,units_ordered\n2025-01-06,42\n2025-01-07,35", language="text")
    st.write("Use store-local dates, one row per day, and nonnegative whole counts. "
             "Use a zero for a known zero-sales day, not a missing record. Do not include customer data.")
raw = sample_csv() if mode == "Sample shop" else upload.getvalue() if upload else None
if st.button("Run forecast", type="primary", disabled=raw is None):
    st.session_state.pop("forecast_result", None)
    try:
        history = load_csv(raw)
        with st.spinner("Fitting Prophet and comparing it with simpler forecasts…"):
            result = run_forecast(history)
        st.session_state["forecast_result"] = (raw, history, result)
    except ValueError as error:
        st.error(str(error))
    except RuntimeError:
        st.error("Could not fit the forecast. Check the local Prophet installation and try again.")
saved = st.session_state.get("forecast_result")
if saved and saved[0] == raw:
    _, history, (chosen, scores, folds, forecast) = saved
    st.subheader(f"Forecast method: {chosen}")
    st.metric("Forecast total, next 28 days", f"{forecast.units_ordered_forecast.sum():,} units")
    hist = pd.Series(history.y.to_numpy(), index=history.ds, name="Observed").tail(56)
    pred = pd.Series(forecast.units_ordered_forecast.to_numpy(),
                     index=pd.to_datetime(forecast.date), name="Forecast")
    st.line_chart(pd.concat([hist, pred], axis=1))
    st.download_button("Download forecast CSV", forecast.to_csv(index=False),
                       "sales_forecast.csv", "text/csv")
    st.subheader("How the methods did on past data")
    st.dataframe(pd.DataFrame([{"Method": label, "MAE (units/day)": round(score, 2)}
                              for label, score in sorted(scores.items(), key=lambda item: item[1])]),
                 hide_index=True)
    st.caption("Lower is better. Each method predicts the same 28 days after each cutoff. "
               "Some windows overlap; scores select the model, not an independent guarantee.")
    with st.expander("See every historical test window"):
        st.dataframe(folds.round(2), hide_index=True)
    st.info("This predicts ordered units, not unmet demand. Stockouts, promotions or a new product mix may change the result.")
