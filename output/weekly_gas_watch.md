# Weekly Gas Price Outlook (as of 2025-08-15)
**Trend**: UP ↑  |  **Expected change**: +6.50% (1-week ahead)

## Model Snapshot
Ridge(alpha=10.00)  |  R²: -2.839  |  MAE: 0.0813  |  RMSE: 0.0918  | Train/Val weeks: 124/32  |  TS-CV R²: -1.145 ± 0.645

## Top Factor Contributions (standardized units)
- storage_dev_5y_pp_ma4: +0.022
- storage_dev_5y_pp: +0.019
- ttf_eur_mwh: +0.015
- wind_index_ma4: -0.011
- storage_fill_pct_wow: +0.010
- brent_usd_bbl: -0.010
- storage_fill_pct_ma4: +0.006
- eua_eur_t_wow: +0.006

*Note:* Contributions are approximations from standardized features and linear coefficients; they provide **directional** insight, not exact price impacts.

---
**Implementation status:** Live storage/wind/FX/HDD; optional Brent via EIA; TTF/EUA/JKM via NDL or CSV if provided.
