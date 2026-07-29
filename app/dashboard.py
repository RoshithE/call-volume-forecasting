"""
Streamlit dashboard: Monthly Demand Forecasting (SARIMA vs. baselines)

Run with: streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from forecasting import (
    adf_test,
    evaluate_all,
    forecast_future,
    grid_search_sarima,
    load_series,
    mape,
    naive_forecast,
    seasonal_naive_forecast,
    train_test_split,
)
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Demand Forecasting", layout="wide")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "monthly-car-sales.csv"


@st.cache_data
def get_series():
    return load_series(str(DATA_PATH))


@st.cache_resource
def run_pipeline(_series, test_periods):
    train, test = train_test_split(_series, test_periods=test_periods)
    best_model, best_row, validated_df = grid_search_sarima(train, test)
    sarima_test_fc = best_model.get_forecast(steps=len(test)).predicted_mean
    comparison = evaluate_all(train, test, sarima_test_fc.values)
    return train, test, best_model, best_row, validated_df, sarima_test_fc, comparison


st.title("Monthly Demand Forecasting: SARIMA vs. Baselines")
st.caption(
    "Replicates the methodology used to forecast call/case volume at Barclays for "
    "proactive staffing decisions. Dataset here is a public retail sales series, "
    "used as a proxy since the original data isn't publishable."
)

series = get_series()

with st.sidebar:
    st.header("Settings")
    test_periods = st.slider("Holdout period (months)", 6, 18, 12)
    forecast_horizon = st.slider("Forward forecast horizon (months)", 3, 24, 12)
    st.markdown("---")
    st.markdown(
        "**Stack:** Python, statsmodels (SARIMAX), pandas, matplotlib, Streamlit"
    )

with st.spinner("Running stationarity tests, grid search, and validation..."):
    train, test, best_model, best_row, validated_df, sarima_test_fc, comparison = run_pipeline(
        series, test_periods
    )

col1, col2, col3 = st.columns(3)
col1.metric("Selected model", f"SARIMA{best_row['order']}x{best_row['seasonal_order']}")
col2.metric("Out-of-sample MAPE", f"{best_row['test_mape']:.2f}%")
best_baseline = comparison[comparison["model"] != "SARIMA (grid searched)"]["mape"].min()
improvement = (1 - best_row["test_mape"] / best_baseline) * 100
col3.metric("Improvement vs. best baseline", f"{improvement:.1f}%")

st.markdown("### Stationarity check (ADF test)")
raw_adf = adf_test(series)
diff_adf = adf_test(series.diff().dropna())
c1, c2 = st.columns(2)
c1.write(f"**Raw series:** ADF p-value = {raw_adf['p_value']:.4f} → "
         f"{'stationary' if raw_adf['is_stationary'] else 'non-stationary'}")
c2.write(f"**Differenced series:** ADF p-value = {diff_adf['p_value']:.2e} → "
         f"{'stationary' if diff_adf['is_stationary'] else 'non-stationary'}")

st.markdown("### Model comparison (out-of-sample, lower is better)")
fig1, ax1 = plt.subplots(figsize=(8, 3.5))
ax1.bar(comparison["model"], comparison["mape"], color=["#c0392b" if m.startswith("SARIMA") else "#7f8c8d" for m in comparison["model"]])
ax1.set_ylabel("MAPE (%)")
ax1.set_xticklabels(comparison["model"], rotation=15, ha="right")
st.pyplot(fig1)

st.markdown("### Holdout forecast vs. actual")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(train.index[-24:], train.values[-24:], label="Train (last 24mo)", color="steelblue")
ax2.plot(test.index, test.values, label="Actual", color="black", marker="o")
ax2.plot(test.index, sarima_test_fc.values, label="SARIMA forecast", color="crimson", marker="o", linestyle="--")
ax2.legend()
ax2.set_ylabel("Units")
st.pyplot(fig2)

st.markdown(f"### {forecast_horizon}-month forward forecast (refit on full history)")
final_model = SARIMAX(
    series,
    order=best_row["order"],
    seasonal_order=best_row["seasonal_order"],
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)
future_mean, future_ci = forecast_future(final_model, steps=forecast_horizon)

fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(series.index[-24:], series.values[-24:], label="History (last 24mo)", color="steelblue")
ax3.plot(future_mean.index, future_mean.values, label="Forecast", color="crimson")
ax3.fill_between(future_ci.index, future_ci.iloc[:, 0], future_ci.iloc[:, 1], color="crimson", alpha=0.15, label="95% CI")
ax3.legend()
ax3.set_ylabel("Units")
st.pyplot(fig3)

forecast_df = pd.DataFrame({
    "month": future_mean.index,
    "forecast": future_mean.values,
    "lower_95": future_ci.iloc[:, 0].values,
    "upper_95": future_ci.iloc[:, 1].values,
})
st.dataframe(forecast_df, use_container_width=True)
st.download_button(
    "Download forecast as CSV",
    forecast_df.to_csv(index=False),
    file_name="forecast.csv",
    mime="text/csv",
)

with st.expander("Validated model candidates (AIC shortlist → out-of-sample MAPE)"):
    st.dataframe(validated_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Claude operational interpretation
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### AI Operational Commentary")
st.caption(
    "Claude translates the forecast into staffing and capacity planning language "
    "for operations managers. Requires ANTHROPIC_API_KEY in a .env file."
)

import os
from dotenv import load_dotenv
load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    st.info(
        "Add ANTHROPIC_API_KEY to a .env file in the project root to enable "
        "plain-language operational commentary from Claude."
    )
else:
    use_case = st.text_input(
        "Planning use case",
        value="call centre staffing and capacity planning",
        help="Describe how this forecast will be used operationally.",
    )
    if st.button("Generate operational commentary", type="primary"):
        from claude_interpreter import ForecastInterpreter
        best_baseline_mape = comparison[
            comparison["model"] != "SARIMA (grid searched)"
        ]["mape"].min()
        model_label = f"SARIMA{best_row['order']}x{best_row['seasonal_order']}"
        with st.spinner("Asking Claude to interpret the forecast..."):
            try:
                interpreter = ForecastInterpreter()
                commentary = interpreter.interpret(
                    model_label=model_label,
                    test_mape=best_row["test_mape"],
                    baseline_mape=best_baseline_mape,
                    improvement_pct=improvement,
                    forecast_df=forecast_df,
                    series_tail=series.iloc[-12:],
                    horizon=forecast_horizon,
                    use_case=use_case,
                )
                st.markdown(commentary)
            except Exception as e:
                st.error(f"Claude API error: {e}")
