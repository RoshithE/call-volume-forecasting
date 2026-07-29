"""Forecasting Copilot - Streamlit app.

Upload historical call/demand volume data (or generate synthetic data),
validate and clean it, explore patterns, train an ARIMA model, evaluate
accuracy, forecast future demand, estimate staffing needs, and optionally
generate an executive summary using the Claude API.
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src import claude_summary, data_loader, data_validation, evaluation
from src import exploratory_analysis as eda
from src import forecasting, staffing
from src.synthetic_data import generate_synthetic_data

st.set_page_config(page_title="Forecasting Copilot", layout="wide")

HORIZON_OPTIONS = {"7 days": 7, "14 days": 14, "30 days": 30, "60 days": 60, "90 days": 90}


def main():
    st.title("Forecasting Copilot")
    st.caption(
        "Upload historical call/demand volume data, evaluate an ARIMA "
        "forecasting model, and generate a plain-language executive summary."
    )

    df_raw, date_col_guess, volume_col_guess = sidebar_data_input()

    if df_raw is None:
        st.info("Upload a CSV or click 'Generate sample data' in the sidebar to get started.")
        return

    st.header("1. Data Preview")
    st.dataframe(df_raw.head(20), use_container_width=True)

    with st.sidebar:
        st.subheader("Column Mapping")
        columns = list(df_raw.columns)
        date_index = columns.index(date_col_guess) if date_col_guess in columns else 0
        volume_index = columns.index(volume_col_guess) if volume_col_guess in columns else 0
        date_col = st.selectbox("Date column", options=columns, index=date_index)
        volume_col = st.selectbox("Volume column", options=columns, index=volume_index)
        horizon_label = st.selectbox("Forecast horizon", options=list(HORIZON_OPTIONS.keys()), index=2)
        horizon = HORIZON_OPTIONS[horizon_label]

        st.subheader("Staffing Inputs")
        available_staff = st.number_input("Available staff", min_value=1.0, value=20.0, step=1.0)
        cases_per_employee = st.number_input("Cases handled per employee per day", min_value=0.1, value=25.0, step=1.0)
        target_utilization = st.slider("Target utilization (%)", min_value=50, max_value=100, value=85)

        st.subheader("Claude Summary")
        use_claude = st.checkbox("Enable executive summary generation", value=True)

    st.header("2. Data Quality")
    clean_df, report = data_validation.validate_and_clean(df_raw, date_col, volume_col)

    for err in report.errors:
        st.error(err)
    for change in report.changes:
        st.success(change)
    for warn in report.warnings:
        st.warning(warn)

    if clean_df is None:
        st.stop()

    st.header("3. Historical Patterns")
    stats = eda.summary_statistics(clean_df)
    render_summary_stats(stats)
    st.plotly_chart(eda.historical_chart(clean_df), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        dow_df = eda.day_of_week_pattern(clean_df)
        st.plotly_chart(eda.day_of_week_chart(dow_df), use_container_width=True)
    with col2:
        month_df = eda.monthly_trend(clean_df)
        st.plotly_chart(eda.monthly_trend_chart(month_df), use_container_width=True)

    st.header("4. Train / Test Split")
    train_df, test_df = forecasting.train_test_split_series(clean_df)
    st.write(
        f"Training range: {train_df['date'].min().date()} to {train_df['date'].max().date()} "
        f"({len(train_df)} days)"
    )
    st.write(
        f"Testing range: {test_df['date'].min().date()} to {test_df['date'].max().date()} "
        f"({len(test_df)} days)"
    )

    st.header("5. Model Selection & Evaluation")
    try:
        with st.spinner("Evaluating ARIMA parameter combinations..."):
            best = forecasting.select_best_model(train_df, test_df)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    st.write(f"Selected ARIMA order (p,d,q): **{best['order']}**")
    render_metrics(best["metrics"])
    actual_vs_predicted_chart(test_df, best["test_predictions"])

    st.header("6. Forecast")
    full_model = forecasting.refit_on_full_history(clean_df, best["order"])
    forecast_df = forecasting.forecast_future(full_model, clean_df["date"].max(), horizon)
    st.plotly_chart(forecast_chart(clean_df, forecast_df), use_container_width=True)
    st.dataframe(forecast_df, use_container_width=True)

    csv_buffer = io.StringIO()
    forecast_df.to_csv(csv_buffer, index=False)
    st.download_button("Download forecast CSV", data=csv_buffer.getvalue(),
                        file_name="forecast.csv", mime="text/csv")

    st.header("7. Staffing Estimate")
    st.caption("This is a simplified planning estimate, not a full workforce-management model.")
    staffing_result = staffing.estimate_staffing(
        forecast_df, available_staff, cases_per_employee, target_utilization
    )
    render_staffing(staffing_result)

    st.header("8. Executive Summary")
    render_claude_section(use_claude, best, stats, forecast_df, staffing_result)

    st.header("9. Model Limitations")
    render_limitations()


def sidebar_data_input():
    st.sidebar.header("Data Input")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    generate_clicked = st.sidebar.button("Generate sample data")

    if "df_raw" not in st.session_state:
        st.session_state["df_raw"] = None

    if uploaded is not None:
        st.session_state["df_raw"] = data_loader.load_csv(uploaded)
    elif generate_clicked:
        st.session_state["df_raw"] = generate_synthetic_data()

    df_raw = st.session_state["df_raw"]
    if df_raw is None:
        return None, None, None

    date_guess = data_loader.guess_date_column(df_raw)
    volume_guess = data_loader.guess_volume_column(df_raw)
    return df_raw, date_guess, volume_guess


def render_summary_stats(stats: dict):
    cols = st.columns(4)
    cols[0].metric("Records", stats["num_records"])
    cols[1].metric("Average Volume", f"{stats['average_volume']:.1f}")
    cols[2].metric("Median Volume", f"{stats['median_volume']:.1f}")
    cols[3].metric("Recent % Change", f"{stats['recent_pct_change']:.1f}%")

    cols2 = st.columns(4)
    cols2[0].metric("Max Volume", f"{stats['max_volume']:.0f}", help=str(stats["highest_volume_date"].date()))
    cols2[1].metric("Min Volume", f"{stats['min_volume']:.0f}", help=str(stats["lowest_volume_date"].date()))
    cols2[2].metric("Start Date", str(stats["date_start"].date()))
    cols2[3].metric("End Date", str(stats["date_end"].date()))


def render_metrics(metrics: dict):
    cols = st.columns(3)
    cols[0].metric("MAE", f"{metrics['mae']:.2f}")
    cols[1].metric("RMSE", f"{metrics['rmse']:.2f}")
    mape_display = f"{metrics['mape']:.2f}%" if metrics["mape"] == metrics["mape"] else "N/A"
    cols[2].metric("MAPE", mape_display)

    with st.expander("What do these metrics mean?"):
        for key, explanation in evaluation.METRIC_EXPLANATIONS.items():
            st.write(f"**{key.upper()}**: {explanation}")


def actual_vs_predicted_chart(test_df: pd.DataFrame, predictions):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df["date"], y=test_df["volume"], name="Actual", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=test_df["date"], y=predictions, name="Predicted", mode="lines+markers"))
    fig.update_layout(title="Actual vs Predicted (Test Period)", xaxis_title="Date", yaxis_title="Volume")
    st.plotly_chart(fig, use_container_width=True)


def forecast_chart(history_df: pd.DataFrame, forecast_df: pd.DataFrame):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df["date"], y=history_df["volume"], name="History", mode="lines"))
    fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["forecast"], name="Forecast", mode="lines"))
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
        y=pd.concat([forecast_df["upper_bound"], forecast_df["lower_bound"][::-1]]),
        fill="toself", fillcolor="rgba(99,110,250,0.15)", line=dict(color="rgba(255,255,255,0)"),
        name="Confidence Interval", showlegend=True,
    ))
    fig.update_layout(title="Forecast", xaxis_title="Date", yaxis_title="Volume")
    return fig


def render_staffing(result: dict):
    cols = st.columns(4)
    cols[0].metric("Average Required Staff", f"{result['average_required_staff']:.1f}")
    cols[1].metric("Peak Required Staff", f"{result['peak_required_staff']:.1f}")
    cols[2].metric("Potential Shortage", f"{result['potential_shortage']:.1f}")
    cols[3].metric("Potential Excess Capacity", f"{result['potential_excess_capacity']:.1f}")

    st.write(f"Peak demand date: **{pd.Timestamp(result['peak_demand_date']).date()}**")
    st.write("Highest-demand dates:")
    st.dataframe(result["high_demand_dates"], use_container_width=True)


def render_claude_section(use_claude: bool, best: dict, stats: dict, forecast_df: pd.DataFrame, staffing_result: dict):
    if not use_claude:
        st.info("Executive summary generation is disabled in the sidebar.")
        return

    if not claude_summary.is_api_key_available():
        st.warning(
            "No ANTHROPIC_API_KEY found in your .env file. Add one to enable "
            "executive summary generation. The rest of the app works normally without it."
        )
        return

    if st.button("Generate Executive Summary"):
        forecast_avg = float(forecast_df["forecast"].mean())
        pct_change = ((forecast_avg - stats["average_volume"]) / stats["average_volume"] * 100) if stats["average_volume"] else 0.0
        peak_row = forecast_df.loc[forecast_df["forecast"].idxmax()]

        risks = []
        if staffing_result["potential_shortage"] > 0:
            risks.append(f"Potential staffing shortage of {staffing_result['potential_shortage']:.1f} FTE at peak demand.")
        if best["metrics"]["mape"] == best["metrics"]["mape"] and best["metrics"]["mape"] > 20:
            risks.append("Model MAPE exceeds 20%, indicating meaningful forecast uncertainty.")
        if not risks:
            risks.append("No major risks identified from the available metrics.")

        prompt = claude_summary.build_compact_summary(
            model_type="ARIMA",
            arima_order=best["order"],
            mae=best["metrics"]["mae"],
            rmse=best["metrics"]["rmse"],
            mape=best["metrics"]["mape"],
            recent_avg=stats["average_volume"],
            forecast_avg=forecast_avg,
            peak_date=str(pd.Timestamp(peak_row["date"]).date()),
            pct_change=pct_change,
            staffing_summary=(
                f"Average required staff: {staffing_result['average_required_staff']:.1f}, "
                f"Peak required staff: {staffing_result['peak_required_staff']:.1f}, "
                f"Available staff: {staffing_result['available_staff']:.1f}"
            ),
            risks="; ".join(risks),
        )
        with st.spinner("Calling Claude..."):
            try:
                summary_text = claude_summary.generate_executive_summary(prompt)
                st.markdown(summary_text)
                st.download_button("Download summary", data=summary_text,
                                    file_name="executive_summary.txt", mime="text/plain")
            except Exception as exc:
                st.error(f"Claude API call failed: {exc}")


def render_limitations():
    st.markdown(
        "- ARIMA assumes patterns from the historical period continue into the future; "
        "it will not anticipate unannounced business changes.\n"
        "- Only four ARIMA parameter combinations are evaluated; a production system "
        "might use auto-ARIMA or additional models (SARIMA, Prophet, ML-based).\n"
        "- Confidence intervals assume roughly normal forecast errors.\n"
        "- Staffing estimates are simplified and do not account for shift scheduling, "
        "shrinkage, part-time staff, or intraday arrival patterns.\n"
        "- This tool is a portfolio/demo project and has not been validated for "
        "production workforce planning decisions."
    )


if __name__ == "__main__":
    main()
