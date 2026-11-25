"""Local, CSV-in / forecast-out app."""
import pandas as pd
import streamlit as st

from local_forecast.copy import COPY, LANGUAGES
from local_forecast.forecast import (
    load_sales_csv, run_forecast, sales_history, sample_sales_csv, xyz_analysis,
)

from local_forecast.sales import AmbiguousMoneyError

st.set_page_config(page_title="Merchant Field Kit", page_icon="📦")
lang = st.selectbox("Language / Язык / Jezik", list(LANGUAGES), format_func=LANGUAGES.get, key="language")
locale = list(LANGUAGES).index(lang)


def t(key):
    return COPY[key][locale]


st.title(t("title"))
st.write(t("intro"))
st.caption(t("privacy"))
mode = st.radio(t("source"), ["sample", "daily", "orders"], format_func=t, horizontal=True, key="mode")
target = st.radio(t("target"), ["revenue", "units_ordered", "both"], format_func=t, horizontal=True, key="target")
metrics = ("revenue", "units_ordered") if target == "both" else (target,)
order = "auto"
if mode == "sample":
    raw = sample_sales_csv()
    st.download_button(t("sample_csv"), raw, "sample_orders.csv", "text/csv")
else:
    upload = st.file_uploader(t("upload"), type=["csv"], help=t("upload_help"), key="sales_csv")
    raw = upload.getvalue() if upload else None
    order = st.selectbox(t("date_order"), ["auto", "dmy", "mdy"], format_func=t, key="date_order")
with st.expander(t("shape")):
    st.code("date,units_ordered,revenue\n2025-01-06,42,429.50\n2025-01-07,35,353.25", language="text")
    st.write(t("daily_help"))
    st.code("date,sku,units_ordered,unit_price\n13.01.2025,CORE-01,2,12.00\n2025-01-14,SPOT-03,1,32.00", language="text")
    st.write(t("orders_help"))
    st.caption(t("date_help"))
    st.caption(t("money_help"))

signature = (mode, target, order, raw)
if st.button(t("run"), type="primary", disabled=raw is None, key="run"):
    st.session_state.pop("forecast_result", None)
    try:
        data = load_sales_csv(raw, input_kind="daily" if mode == "daily" else "orders", date_order=order)
        histories = {metric: sales_history(data, metric) for metric in metrics}
        with st.spinner(t("fitting")):
            runs = {metric: run_forecast(history, metric=metric) for metric, history in histories.items()}
        st.session_state["forecast_result"] = (signature, histories, runs, xyz_analysis(data))
    except AmbiguousMoneyError as error:
        st.error(t("ambiguous_money").format(line=error.line, field=error.field))
    except ValueError as error:
        if str(error).startswith("Ambiguous dates"):
            st.error(t("ambiguous"))
        elif str(error).startswith("This CSV has no "):
            st.error(t("missing_target"))
        else:
            st.error(t("bad_csv"))
            st.code(str(error), language="text")
    except RuntimeError:
        st.error(t("fit_error"))

saved = st.session_state.get("forecast_result")
if saved and saved[0] == signature:
    _, histories, runs, xyz = saved
    output = None
    for metric in metrics:
        hist = histories[metric]
        winner, scores, folds, forecast = runs[metric]
        st.subheader(f'{t(metric)} · {t("method")}: {t(winner)}')
        column = f"{metric}_forecast"
        total = forecast[column].sum()
        amount = f'{total:,.2f} {t("amount")}' if metric == "revenue" else f'{total:,} {t("units")}'
        st.metric(t("total"), amount)
        actual = pd.Series(hist.y.to_numpy(), index=hist.ds, name=t("observed")).tail(56)
        expected = pd.Series(forecast[column].to_numpy(), index=pd.to_datetime(forecast.date), name=t("forecast"))
        st.line_chart(pd.concat([actual, expected], axis=1))
        output = forecast if output is None else output.merge(forecast, on="date", validate="one_to_one")
        with st.expander(t("evaluation")):
            mae = t("mae_money" if metric == "revenue" else "mae_units")
            st.dataframe(pd.DataFrame([{t("method"): t(label), mae: round(score, 2)}
                                      for label, score in sorted(scores.items(), key=lambda item: item[1])]),
                         hide_index=True)
            st.caption(t("mae_help"))
            st.caption(t("folds"))
            folds = folds.copy()
            folds["Method"] = folds.Method.map(t)
            st.dataframe(folds.rename(columns={"Training through": t("train_through"), "Test through": t("test_through"),
                                             "Method": t("method"), "MAE (units/day)": mae, "MAE (currency/day)": mae}).round(2),
                         hide_index=True)
    st.download_button(t("download"), output.to_csv(index=False), "sales_forecast.csv", "text/csv")
    if not xyz.empty:
        st.subheader(t("xyz"))
        st.caption(t("xyz_help"))
        st.dataframe(xyz.rename(columns={c: t(c) for c in xyz.columns if c != "sku"}).round(2), hide_index=True)
    st.info(t("limits"))
