# Weather Prediction Using Artificial Intelligence

Mini project for Principles of Artificial Intelligence. This classroom application demonstrates supervised learning for two next-day tasks: maximum temperature regression and rain/no-rain classification.

## Run it

1. Install Python 3.10 or newer.
2. In this folder, create a virtual environment and install packages:
   - Windows: `py -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
   - `pip install -r requirements.txt`
3. Start the app: `python app.py`
4. Open `http://127.0.0.1:5000`.

On the first run, a reproducible 10-year synthetic daily dataset is created, chronological training/testing is performed, and the trained models and their metrics are saved under `models/`. The dataset CSV is saved under `data/`. Run `python train_model.py` to retrain.

## AI method

This is supervised machine learning. Random Forest Regression estimates next day's maximum temperature. Random Forest Classification estimates the probability of rain tomorrow. Inputs are today's maximum/minimum temperature, humidity, pressure, wind, cloud cover and rainfall, plus temperature range and day-of-year season encodings. The split is chronological (first 80% train, final 20% test) to avoid testing on earlier dates than the training examples. Metrics include MAE, RMSE, R-squared, precision, recall, F1, AUC and a confusion matrix.

## Important project limitation

The included dataset is synthetic, generated from seasonal patterns and randomized weather variation. It is suitable for demonstrating the complete AI workflow and reproducible code, but it is not an operational forecast or evidence of real-world accuracy. Replace `data/weather_data.csv` with a quality, documented historical weather dataset (same columns) before making real-world claims. Report results from the generated `models/metrics.json`; do not copy example scores from screenshots or drafts.

## Files

- `app.py`: Flask website and prediction API
- `train_model.py`: feature engineering, model training, evaluation, persistence
- `data_gen.py`: reproducible demo dataset generator
- `static/index.html`: responsive interactive dashboard
- `data/weather_data.csv`: generated on first run
- `models/`: fitted models and actual evaluation metrics, generated on first run
- `PROJECT_REPORT.md`: report draft
