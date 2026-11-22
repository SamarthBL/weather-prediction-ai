# Presentation and Viva Notes

## Suggested 8-slide presentation

1. **Title** - Weather Prediction Using Artificial Intelligence; student, course and institution.
2. **Problem and objectives** - predict next-day maximum temperature and rain/no-rain from current weather readings.
3. **AI approach** - supervised learning; regression for temperature, binary classification for rain; why two outputs need different model types.
4. **Dataset** - features, targets, daily synthetic dataset, deterministic seed; state clearly that synthetic data is for demonstration.
5. **Method** - feature engineering, chronological 80/20 split, Random Forest training; show the pipeline diagram: readings -> features -> trained models -> forecast.
6. **Evaluation** - fill with actual values from `models/metrics.json`; explain MAE/RMSE/R² and precision/recall/F1/AUC; include confusion matrix.
7. **Application demo** - start Flask app, change sliders, show temperature and rain probability, open model metrics.
8. **Conclusion and next steps** - end-to-end AI prototype; real weather data, time-series validation, location-specific features as future work.

## Viva questions and short answers

**Why is this an AI project?** It learns a mapping from labeled weather examples to future outcomes instead of relying on fixed if/else forecast rules.

**Why are there two models?** Temperature is continuous and is treated as regression. Rain/no-rain is categorical and is treated as classification.

**What is Random Forest?** An ensemble that trains multiple decision trees on varied samples/features and combines their predictions. Averaging/class voting can reduce the instability of a single tree.

**Why use a chronological split?** The model should learn from earlier dates and be evaluated on later dates, which better reflects forecasting and avoids training on future records.

**What does MAE mean?** The average absolute difference between predicted and actual next-day maximum temperature, in degrees Celsius; lower is better.

**What does recall mean for rain?** Of the rainy test days, the fraction the classifier correctly identifies. Higher recall catches more rainy days, though precision may fall.

**Can this predict real weather accurately?** Not yet. The bundled data is synthetic and only demonstrates the complete workflow. Real, documented weather records and evaluation are needed before claiming practical accuracy.

**How can it be improved?** Use station-level historical data, add lagged weather and location features, tune with rolling time-series validation, compare with simple baselines, and calibrate rain probabilities.
