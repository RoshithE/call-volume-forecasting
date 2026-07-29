# Forecasting Copilot

A locally runnable portfolio project that forecasts call/demand volume with
ARIMA, evaluates accuracy, estimates staffing needs, and uses the Claude API
to explain the results in plain business language.

## Business Problem

Operations and workforce planning teams need to anticipate call or demand
volume in order to staff appropriately - understaffing causes long wait
times and burnout, while overstaffing wastes budget. This project shows an
end-to-end approach: clean the historical data, fit and validate a simple
statistical forecasting model, translate the forecast into a staffing plan,
and communicate the results to non-technical stakeholders.

## Project Overview

Forecasting Copilot is a single-page Streamlit application. A user uploads a
CSV of historical daily volume (or generates synthetic sample data), the app
validates and cleans it, explores historical patterns, trains and evaluates
an ARIMA model, produces a forward-looking forecast with confidence
intervals, estimates staffing requirements, and optionally calls the Claude
API to produce an executive summary.

## Architecture

```
CSV upload / synthetic data
        |
Data validation & cleaning (src/data_validation.py)
        |
Exploratory analysis (src/exploratory_analysis.py)
        |
Chronological train/test split + ARIMA model selection (src/forecasting.py)
        |
Accuracy evaluation (src/evaluation.py)
        |
Future forecast with confidence intervals (src/forecasting.py)
        |
Staffing estimate (src/staffing.py)
        |
Executive summary via Claude API (src/claude_summary.py)
        |
Streamlit UI (app.py)
```

## Features

- CSV upload with a built-in synthetic-data generator (18+ months of daily
  data with weekly seasonality, trend, noise, spikes, and low-volume periods)
- Data validation that separates blocking errors (e.g. negative volumes) from
  auto-corrected changes (date parsing, sorting, de-duplication, small-gap
  interpolation), with every change shown to the user
- Exploratory analysis: summary statistics, day-of-week and monthly patterns
- Chronological train/test split (never shuffled) with ARIMA order search
  over four candidate parameter combinations, selected by test RMSE
- MAE / RMSE / MAPE evaluation with plain-English explanations and safe
  handling of zero-volume days
- Forecast for 7/14/30/60/90 days with confidence intervals and a
  downloadable CSV
- Simplified staffing calculator (average and peak required staff, shortage
  and excess capacity)
- Optional Claude-generated executive summary built from a small, compact
  set of numbers - never the raw dataset
- Example SQL (`sql/demand_summary.sql`) showing how source data could be
  aggregated upstream of the app

## Technology Stack

Python 3.11, Pandas, NumPy, Statsmodels, Scikit-learn, Streamlit, Plotly,
Anthropic Python SDK, python-dotenv, Pytest.

## Folder Structure

```
forecasting-copilot/
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   └── sample_demand.csv
├── sql/
│   └── demand_summary.sql
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── synthetic_data.py
│   ├── exploratory_analysis.py
│   ├── forecasting.py
│   ├── evaluation.py
│   ├── staffing.py
│   └── claude_summary.py
└── tests/
    ├── test_data_validation.py
    ├── test_forecasting.py
    └── test_staffing.py
```

## Installation

```bash
cd forecasting-copilot
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## API Key Setup

Copy `.env.example` to `.env` and add your Anthropic API key:

```bash
cp .env.example .env
```

```
ANTHROPIC_API_KEY=your_api_key_here
CLAUDE_MODEL=claude-3-5-sonnet-20241022
```

The app works fully without a key - the executive summary section will
simply show a message explaining that the key is missing.

## Run Instructions

```bash
streamlit run app.py
```

Then, in the sidebar, either upload a CSV or click "Generate sample data" to
explore the app immediately.

## Input Format

A CSV with at minimum a date column and a volume column, for example:

```
date,volume
2025-01-01,340
2025-01-02,375
2025-01-03,361
```

Optional `team` and `category` columns are supported and preserved by the
synthetic-data generator, though the current model trains on total daily
volume.

## Forecast Methodology

The cleaned series is split chronologically into roughly 80% training and
20% testing data - time series data is never randomly shuffled. Four ARIMA
parameter combinations, (1,1,1), (2,1,1), (1,1,2), and (2,1,2), are each
fit on the training data and evaluated on the test data; the combination
with the lowest test RMSE is selected and refit on the full history before
producing the forward-looking forecast with confidence intervals.

## Evaluation Metrics

MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and MAPE (Mean
Absolute Percentage Error) are calculated on the held-out test period, with
MAPE safely handling any zero-volume days. Plain-English explanations for
each metric are shown in the app.

## Screenshots

_Add screenshots of the Streamlit app here once you have run it locally._

## Limitations

- ARIMA assumes historical patterns continue into the future and cannot
  anticipate unannounced business changes.
- Only four ARIMA parameter combinations are evaluated; a production system
  might use auto-ARIMA or additional model families.
- Confidence intervals assume roughly normal forecast errors.
- The staffing calculator is a simplified planning estimate and does not
  account for shift scheduling, shrinkage, part-time staff, or intraday
  arrival patterns.
- This is a portfolio/demo project. It has not been deployed to the cloud
  and has not been used in a production workforce-planning environment.

## Future Improvements

- Add SARIMA or Prophet as alternative model options
- Support multiple series (e.g. per-team forecasts) in a single run
- Add backtesting across multiple rolling train/test windows
- Persist uploaded data and results to a database for historical comparison

## Interview Explanation

"I built Forecasting Copilot to demonstrate an end-to-end forecasting
workflow similar to what I've done analyzing fraud call volume for workforce
planning. The app validates and cleans uploaded data, fits and compares a
small set of ARIMA models using a chronological train/test split, reports
MAE/RMSE/MAPE in plain language, and turns the forecast into a simplified
staffing estimate. I also integrated the Claude API so the numeric output
can be translated into a short executive summary, which mirrors how I'd
communicate model results to non-technical stakeholders in a real role."

## Resume Bullets

- Built a Streamlit forecasting application that cleans and validates
  time-series call/demand data and trains an ARIMA model selected by
  test-set RMSE across multiple candidate parameter combinations.
- Implemented a chronological train/test evaluation pipeline reporting
  MAE, RMSE, and MAPE, with safe handling of zero-volume edge cases and
  automated unit tests covering validation, forecasting, and staffing logic.
- Translated statistical forecasts into a simplified staffing model
  estimating required headcount, potential shortages, and excess capacity
  for workforce planning use cases.
- Integrated the Anthropic Claude API to generate plain-language executive
  summaries from a compact, pre-aggregated set of model metrics rather than
  raw data, demonstrating practical LLM integration for business reporting.
