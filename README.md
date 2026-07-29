# Monthly Demand Forecasting: SARIMA vs. Baselines

Time series forecasting project that replicates the methodology used in production
to forecast call and case volume for staffing and resource planning. Builds two
honest baselines, grid searches a seasonal ARIMA model, validates it against a true
holdout period, and reports the improvement in forecast error (MAPE).

**Why this project exists:** at Barclays I built ARIMA-based time series models to
forecast call and case volume for the Fraud team, enabling proactive staffing and
resource allocation instead of reactive scheduling. The production data isn't mine
to publish, so this project uses a public monthly retail sales series as a proxy
and reproduces the same modeling and validation workflow end to end.

## Results

| Model | Out-of-sample MAPE (12-month holdout) |
|---|---|
| Naive (repeat last value) | 22.27% |
| Seasonal naive (12-month lag) | 10.83% |
| **SARIMA(0,1,2)(0,1,1,12), grid searched and validated** | **8.48%** |

The selected model reduced forecast error by about 62% versus the naive baseline
and about 22% versus the seasonal-naive baseline, on a genuine out-of-sample test
period the model never saw during fitting.

## Methodology

1. **Stationarity check.** Augmented Dickey-Fuller test on the raw series (non-stationary,
   p = 0.66) and the first-differenced series (stationary, p < 0.001), which justifies
   using `d = 1` in the ARIMA order.
2. **Two baselines first.** A naive forecast (repeat the last value) and a seasonal
   naive forecast (repeat the value from 12 months prior). Any model has to beat both
   of these to justify the added complexity in production.
3. **Grid search on AIC, then validate on a real holdout.** Search `(p,d,q)(P,D,Q,12)`
   combinations, rank the top candidates by AIC on the training fit, then evaluate
   the top 5 against a 12-month holdout and select the final model by out-of-sample
   MAPE, not by AIC alone. AIC can favor an overfit model; the holdout check is what
   an actual deployment decision has to survive.
4. **Residual diagnostics** on the selected model to confirm residuals resemble white
   noise before trusting the forward forecast.
5. **Refit on full history** and produce a 12-month forward forecast with 95%
   confidence intervals, the artifact a staffing or capacity-planning team would
   actually use.

## Repository structure

```
call-volume-forecasting/
├── data/
│   └── monthly-car-sales.csv       # Public dataset, used as a demand proxy
├── notebooks/
│   └── forecasting_analysis.ipynb  # Full analysis, executed with real outputs
├── app/
│   └── dashboard.py                # Interactive Streamlit dashboard
├── src/
│   └── forecasting.py              # Core forecasting/evaluation functions
├── build_notebook.py               # Script that generates the notebook
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt

# Explore the full analysis
jupyter notebook notebooks/forecasting_analysis.ipynb

# Or launch the interactive dashboard
streamlit run app/dashboard.py
```

## Stack

Python, pandas, statsmodels (SARIMAX), matplotlib, Streamlit.

## Note on data

The dataset (`monthly-car-sales.csv`) is a public time series (monthly car sales,
1960 to 1968, via the [jbrownlee/Datasets](https://github.com/jbrownlee/Datasets)
repository) used as a stand-in for call/case volume data, which is proprietary to
my employer and not publishable. The modeling approach, validation discipline, and
evaluation metric shown here are the same ones used in that production work.
