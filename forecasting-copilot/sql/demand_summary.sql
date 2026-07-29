-- sql/demand_summary.sql
-- Example query demonstrating how source call/demand volume data could be
-- aggregated and prepared before being loaded into the Forecasting Copilot
-- application. Written in generic ANSI-ish SQL (Redshift/Snowflake/Postgres
-- style window functions); this does not connect to a real database, it is
-- illustrative of the upstream data preparation step.

WITH daily_volume AS (
    SELECT
        team,
        CAST(event_date AS DATE)      AS call_date,
        SUM(call_count)               AS daily_calls
    FROM raw_call_events
    GROUP BY team, CAST(event_date AS DATE)
),

rolling_avg AS (
    SELECT
        team,
        call_date,
        daily_calls,
        AVG(daily_calls) OVER (
            PARTITION BY team
            ORDER BY call_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_7day_avg
    FROM daily_volume
),

peak_days AS (
    SELECT
        team,
        call_date,
        daily_calls,
        RANK() OVER (PARTITION BY team ORDER BY daily_calls DESC) AS volume_rank
    FROM daily_volume
)

SELECT
    r.team,
    r.call_date,
    r.daily_calls,
    ROUND(r.rolling_7day_avg, 1) AS rolling_7day_avg,
    CASE WHEN p.volume_rank <= 5 THEN TRUE ELSE FALSE END AS is_top5_peak_day
FROM rolling_avg r
LEFT JOIN peak_days p
    ON r.team = p.team AND r.call_date = p.call_date
ORDER BY r.team, r.call_date;
