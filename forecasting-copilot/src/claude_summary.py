"""Claude (Anthropic API) integration for generating executive summaries.

Only a compact, pre-aggregated summary is ever sent to the model - never the
raw dataset - to keep the prompt small and avoid sending unnecessary detail.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")

SYSTEM_PROMPT = (
    "You are a data science communication assistant helping a workforce "
    "planning team understand a call-volume forecasting model. Write in "
    "clear, concise business language for a non-technical audience. "
    "Only use the numbers provided to you - do not invent facts, statistics, "
    "or context that was not supplied. If information is missing, say so "
    "rather than guessing."
)

USER_PROMPT_TEMPLATE = """Here is a compact summary of a call-volume forecasting model. Use ONLY
this information to write your response - do not invent any additional
facts, numbers, or context.

Model type: {model_type}
Selected ARIMA order (p,d,q): {arima_order}
Mean Absolute Error (MAE): {mae}
Root Mean Squared Error (RMSE): {rmse}
Mean Absolute Percentage Error (MAPE): {mape}%
Recent historical average daily volume: {recent_avg}
Forecasted average daily volume: {forecast_avg}
Peak forecast date: {peak_date}
Percentage change (forecast vs recent history): {pct_change}%
Staffing estimate: {staffing_summary}
Identified risks: {risks}

Write your response with these exact sections, each with a short heading:
1. Executive Summary
2. Forecast Trend
3. Operational Impact
4. Staffing Recommendation
5. Risks
6. Model Limitations

Keep the entire response under 400 words and avoid technical jargon where possible.
"""


def is_api_key_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def build_compact_summary(
    model_type,
    arima_order,
    mae,
    rmse,
    mape,
    recent_avg,
    forecast_avg,
    peak_date,
    pct_change,
    staffing_summary,
    risks,
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        model_type=model_type,
        arima_order=arima_order,
        mae=round(mae, 2),
        rmse=round(rmse, 2),
        mape=round(mape, 2) if mape == mape else "N/A",
        recent_avg=round(recent_avg, 2),
        forecast_avg=round(forecast_avg, 2),
        peak_date=peak_date,
        pct_change=round(pct_change, 2),
        staffing_summary=staffing_summary,
        risks=risks,
    )


def generate_executive_summary(user_prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Call the Anthropic API and return the generated text.

    Raises RuntimeError if no API key is configured. Callers should check
    is_api_key_available() first and show a friendly message instead of
    calling this function when no key is present.
    """
    if not is_api_key_available():
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
