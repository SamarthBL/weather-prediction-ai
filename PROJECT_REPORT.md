# Mini Project Report: Weather Prediction Using Artificial Intelligence

**Subject:** Principles of Artificial Intelligence  
**Student:** [Your name]  
**Register number:** [Your register number]  
**Institution:** [College name]  
**Academic year:** [Year]

## Abstract

This mini project applies supervised machine learning to estimate next-day maximum temperature and the probability of rain. It includes a responsive website, a data preparation and training pipeline, and model evaluation. A Random Forest regressor handles the continuous temperature target, while a Random Forest classifier handles the binary rain target. The demonstration uses reproducibly generated synthetic daily observations; therefore its scores demonstrate the evaluation workflow and must not be interpreted as operational weather accuracy.

## 1. Introduction

Weather affects agriculture, travel, energy use and daily planning. Artificial intelligence can learn relationships between historical observations and future outcomes. This project demonstrates the AI problem-solving cycle: represent observations as features, learn from labeled examples, predict unseen outcomes, and evaluate predictions with appropriate measures.

## 2. Problem definition

Given weather readings for day t, estimate maximum temperature on day t+1 and whether measurable rain occurs on day t+1. Temperature is a regression problem. Rain/no-rain is a binary classification problem. The demo rain threshold is 1 mm/day.

## 3. Dataset and features

The included generator creates ten years of daily synthetic observations with seasonal temperature variation, correlated humidity/cloud cover, pressure and wind variation, and probabilistic rainfall. Inputs are maximum and minimum temperature, humidity, pressure, wind speed, cloud cover and rainfall. Feature engineering adds daily temperature range and sine/cosine encodings of day of year. Targets are next-day maximum temperature and next-day rain class. A fixed random seed makes the dataset reproducible.

## 4. Methodology

The data is ordered by date and divided chronologically: the first 80% is used for training and the final 20% for testing. This prevents future examples from informing predictions of earlier test dates. Random Forest models are fitted with fixed hyperparameters and random seed. Temperature evaluation uses mean absolute error (MAE), root mean squared error (RMSE) and R-squared. Rain evaluation uses accuracy, precision, recall, F1 and ROC AUC, along with a confusion matrix. The models and metric JSON are persisted for the Flask application.

## 5. System design

The browser dashboard accepts weather readings and submits JSON to a Flask prediction endpoint. The backend engineers the seasonal and range features, loads the fitted models, and returns temperature and rain probability. A metrics endpoint supplies held-out evaluation values to the dashboard. If opened directly as a local HTML file, the front end clearly identifies its illustrative fallback; use the Flask server to run the trained AI models.

## 6. Results

Run `python app.py` once to train the models. Copy the measured values from `models/metrics.json` into this section after that run. The temperature section reports MAE, RMSE and R-squared. The rain section reports accuracy, precision, recall, F1 and AUC. Include the confusion matrix and feature importances as evidence. The project intentionally does not claim fabricated results before an actual local training run.

## 7. AI concepts demonstrated

- Supervised learning from labeled examples
- Regression and binary classification
- Feature engineering and seasonal representation
- Chronological train/test validation
- Ensemble learning with decision tree forests
- Evaluation using task-appropriate metrics
- Model persistence and inference through a web API

## 8. Limitations and future work

Synthetic observations do not capture the full complexity of real weather and cannot support real-world forecasting. Future work should use a documented station dataset, handle missing values, compare baselines and alternative models, tune hyperparameters using time-series validation, calibrate rain probabilities, and evaluate across locations and seasons. A location-aware forecast would require location and historical weather features.

## 9. Conclusion

The project demonstrates an end-to-end AI workflow for two weather prediction tasks, from dataset generation and feature engineering through model evaluation and web inference. Replacing synthetic data with a documented real dataset is the essential next step for meaningful real-world performance claims.

## References

1. Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5-32.
2. scikit-learn documentation, ensemble methods and model evaluation: https://scikit-learn.org/stable/.
3. Flask documentation: https://flask.palletsprojects.com/.
