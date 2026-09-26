# Outcomes of the pre-registered experiments exp10, exp13 and exp14

This file reports the outcome of every prediction in `PREREG_exp10_exp13_exp14.md`. The pre-registration file itself is left unchanged.

## Timeline (local time, UTC+8)

| Event | Time | Commit / file |
|---|---|---|
| Pre-registration committed on branch `exp/robustness-prereg` | 26 Sep 2026, 08:27:13 | "Pre-register exp10, exp13, exp14: design and predictions P4-P9" |
| Pre-registration pushed to GitHub | 08:27:27 | |
| exp14 started / finished | about 08:30 / 08:35 | `results/logs/exp14.log` |
| exp13 finished | 09:02 | `results/logs/exp13.log` |
| exp10 PharmaSales finished | 10:33 | `results/logs/exp10_pharma.log` |
| exp10 Rossmann finished | 12:22 | `results/logs/exp10_rossmann.log` |
| Results committed and pushed | 19:11 | "Results of pre-registered exp10, exp13, exp14" |

No design parameter was changed after the pre-registration, and no run was repeated. Section 7 of the pre-registration (Deviations) therefore remains empty.

## Outcomes

| Prediction | Statement (short) | Observed | Outcome |
|---|---|---|---|
| P4a | LSTM with a wider grid changes pooled test RMSE by at most ±2% | −0.15% | Confirmed |
| P4b | AR-LRX not worse than wide-grid LSTM by more than 1% | −0.65% daily, −1.38% weekly | Confirmed |
| P5a | AR-LRX not worse than rolling ETS by more than 1% | +4.91% daily, +0.69% weekly | **Failed** (daily) |
| P5b | ETS not more accurate than rolling ARIMA on daily data | ETS −4.27% | **Failed** |
| P6a | Share of closed gates at alpha = 0.50 ≥ clean | 0.48 vs 0.53 | **Failed** |
| P6b | Gated mean change vs S1 ≤ +0.5% and below the ungated hybrid | 0.28 / 0.04 / 0.14% vs 2.55 / 1.07 / 1.83% | Confirmed |
| P7a | Rossmann `resid_val_r2` decreases monotonically with noise | 0.238 → 0.206 → 0.159 → 0.068 | Confirmed |
| P7b | Advantage over S1 [structural] shrinks monotonically | −23.03 → −20.30 → −11.98 → −4.39% | Confirmed |
| P7c | Never worse than S1 by more than 0.5% | always better | Confirmed |
| P7d | At alpha ≤ 0.25, better than XGBoost and LightGBM | −6.08 / −5.42% and −5.45 / −4.65% | Confirmed |
| P8a | PharmaSales shocks, affected rows: AR-LRX better than XGBoost | −4.44% (spike), −0.89% (level) | Confirmed |
| P8b | PharmaSales shocks, affected rows: AR-LRX vs S1 ≤ +2% | +0.49%, +0.33% | Confirmed |
| P9a | Rossmann shocks, affected rows: AR-LRX-Aug better than S1 [structural] | −29.90%, −33.33% | Confirmed |
| P9b | Rossmann shocks, affected rows: AR-LRX-Aug not worse than XGBoost by more than 2% | +3.43%, +8.15% | **Failed** |

Ten of fourteen predictions were confirmed.

## Exploratory observations (not pre-registered)

- **Why ETS wins on daily data.** In 10 of 16 daily configurations ETS selected a seasonal component with period 7, and in 8 of these together with a damped trend. AR-LRX with the calendar-keyed first stage of exp11 remains 2.27% behind ETS on daily data and 1.25% on weekly data (pooled, 10 distinct configurations each).
- **LSTM grid.** The wide-grid LSTM selected a new edge width (32 or 512) in 20 of 32 configurations, with practically no change in accuracy.
- **Clean-target scoring on Rossmann.** When predictions under noise are scored against the unperturbed target, the advantage of AR-LRX-Aug over XGBoost grows with noise: −6.2%, −6.7%, −8.8% and −13.4%.
- **Whole test block under shocks on Rossmann.** AR-LRX-Aug vs XGBoost is −4.89% in the spike scenario and −1.07% in the level-shift scenario; vs LightGBM it is +0.99% in the level-shift scenario.

## Where the results are used

- Dissertation Chapter IV, Section 4.6.2 and Table 4.11: exp13 and exp14.
- Dissertation Chapter V, Section 5.7 and Tables 5.10 to 5.12: exp10 and the prediction outcomes.
