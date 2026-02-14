 1. Reduce turnover (biggest impact)
  - Smooth predictions with an EMA or require a minimum threshold before trading
  - Only flip position when conviction crosses a threshold (e.g., predict > 0.001 to go long, <
   -0.001 to go short, else hold)
  - This alone could turn a gross Sharpe of 0.19 into a net-positive strategy

  2. Then proceed to ensemble (Step 3)
  - The whole point of ensembling is that Ridge and XGBoost make different mistakes
  - Even though XGBoost is weak alone, averaging may stabilize the signal and reduce turnover
  naturally (conflicting signals → no trade)

  3. XGBoost tuning can wait
  - Its zero Sharpe suggests it's learning noise. But in an ensemble, it might still add value
  by dampening Ridge's false signals
  - If anything, make it simpler (fewer features, shallower trees) rather than more complex
