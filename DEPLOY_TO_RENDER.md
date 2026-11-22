# Deploy this project to Render

## What this setup does

`render.yaml` defines a Render Python web service. Render installs `requirements.txt`, trains the sample models during the build, then serves the Flask app with Gunicorn on Render's assigned `$PORT`. The `/healthz` endpoint is used for health checks. Gunicorn is excluded on Windows so the local Windows setup remains installable.

## Steps

1. Create a GitHub repository for this project and upload the contents of `weather-ai-mini-project`. Include `app.py`, `train_model.py`, `data_gen.py`, `requirements.txt`, `render.yaml`, `.python-version` if present, and `static/`. Do not upload `.venv/`, `__pycache__/`, or local logs.
2. Sign in to Render and select **New + -> Blueprint**.
3. Connect the GitHub repository containing the project. Render will find `render.yaml` at the repository root.
4. Review the proposed `weather-prediction-ai` service and select **Apply** to create it.
5. Follow the service's **Events** or **Logs** page while Render installs dependencies, trains the models, and deploys Gunicorn.
6. When the service is live, open the `onrender.com` URL shown in the Render dashboard. The health-check endpoint is `/healthz`.

If creating a Web Service instead of a Blueprint, use **Python 3**, build command `pip install -r requirements.txt && python train_model.py`, and start command `gunicorn app:app --workers 1 --bind 0.0.0.0:$PORT`.

The repository must be connected to Render for deployment. Render's Flask deployment guide describes the Git repository, build command, and start command flow: https://render.com/docs/deploy-flask.

## Data note

The bundled training data is synthetic and intended for a classroom demonstration. The build creates model artifacts from this data; this is not an operational weather forecast.
