"""
claude_interpreter.py

Uses Claude to translate SARIMA forecast results into plain-language operational
commentary — seasonal patterns, staffing implications, and capacity planning
recommendations that a non-technical ops manager can act on.
"""

from __future__ import annotations

import os
from typing import Optional

import anthropic
import pandas as pd

SYSTEM_PROMPT = """You are a data science advisor helping an operations manager understand \
a time series forecast. The forecast was produced by a SARIMA model validated against a \
held-out test period.

Your job is to translate technical results into actionable operational language.

Rules:
- Only use the data provided. Do not invent numbers or reference external benchmarks.
- Be specific: name months, magnitudes, and trends from the forecast data.
- Write for a non-technical audience (operations managers, workforce planners).
- Keep the total response under 400 words.
- Use Markdown with clear headings."""


def _format_forecast_context(
    model_label: str,
    test_mape: float,
    baseline_mape: float,
    improvement_pct: float,
    forecast_df: pd.DataFrame,
    series_tail: pd.Series,
    horizon: int,
) -> str:
    """Build the context block sent to Claude."""
    peak_month = forecast_df.loc[forecast_df["forecast"].idxmax(), "month"]
    peak_val = forecast_df["forecast"].max()
    trough_month = forecast_df.loc[forecast_df["forecast"].idxmin(), "month"]
    trough_val = forecast_df["forecast"].min()
    avg_forecast = forecast_df["forecast"].mean()

    last_actual = series_tail.iloc[-1]
    first_forecast = forecast_df["forecast"].iloc[0]
    momentum = "above" if first_forecast > last_actual else "below"

    lines = [
        "--- Model Performance ---",
        f"Model: {model_label}",
        f"Out-of-sample MAPE (12-month holdout): {test_mape:.2f}%",
        f"Best baseline MAPE (seasonal naive): {baseline_mape:.2f}%",
        f"Improvement over baseline: {improvement_pct:.1f}%",
        "",
        f"--- {horizon}-Month Forward Forecast ---",
        f"Forecast horizon: {horizon} months",
        f"Average forecasted volume: {avg_forecast:.1f} units/month",
        f"Peak volume: {peak_val:.1f} units in {peak_month.strftime('%B %Y')}",
        f"Trough volume: {trough_val:.1f} units in {trough_month.strftime('%B %Y')}",
        f"Peak-to-trough swing: {peak_val - trough_val:.1f} units ({(peak_val/trough_val - 1)*100:.1f}%)",
        f"First forecasted month is {momentum} the last actual observation ({last_actual:.1f})",
        "",
        "--- Monthly Forecast Detail ---",
    ]
    for _, row in forecast_df.iterrows():
        lines.append(
            f"  {row['month'].strftime('%b %Y')}: {row['forecast']:.1f} "
            f"(95% CI: {row['lower_95']:.1f} – {row['upper_95']:.1f})"
        )
    return "\n".join(lines)


class ForecastInterpreter:
    """Generates operational commentary on a SARIMA forecast using Claude."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)

    def interpret(
        self,
        model_label: str,
        test_mape: float,
        baseline_mape: float,
        improvement_pct: float,
        forecast_df: pd.DataFrame,
        series_tail: pd.Series,
        horizon: int = 12,
        use_case: str = "call centre staffing",
    ) -> str:
        """
        Generate a plain-language operational interpretation of the forecast.

        Args:
            model_label: SARIMA order string, e.g. "SARIMA(0,1,2)(0,1,1,12)".
            test_mape: Out-of-sample MAPE of the SARIMA model.
            baseline_mape: MAPE of the best baseline model.
            improvement_pct: Percentage improvement of SARIMA over baseline.
            forecast_df: DataFrame with columns: month, forecast, lower_95, upper_95.
            series_tail: Last 12 months of the actual series for momentum context.
            horizon: Forecast horizon in months.
            use_case: The operational planning use case (e.g. "call centre staffing").

        Returns:
            Markdown-formatted operational commentary string.
        """
        context = _format_forecast_context(
            model_label, test_mape, baseline_mape, improvement_pct,
            forecast_df, series_tail, horizon
        )

        user_message = (
            f"The following data comes from a SARIMA demand forecast used for {use_case} planning.\n\n"
            f"{context}\n\n"
            "Please provide operational commentary with these sections:\n\n"
            "## Model Reliability\n"
            "How accurate is this forecast and can operations trust it for planning?\n\n"
            "## Seasonal Patterns\n"
            "What seasonal trends does the forecast reveal? When are the peak and trough periods?\n\n"
            "## Staffing and Capacity Implications\n"
            "What should the operations team do differently in the peak vs trough periods?\n\n"
            "## Planning Risks\n"
            "What are the main risks in relying on this forecast? Mention the confidence intervals.\n\n"
            "## Recommended Actions\n"
            "Three bullet-point actions the operations manager should take in the next 30 days."
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text
