"""Synthetic, season-aware weather dataset for a reproducible classroom ML demo."""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42

def make_dataset(days=3650, seed=SEED):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2010-01-01', periods=days, freq='D')
    doy = dates.dayofyear.to_numpy()
    seasonal = np.sin(2*np.pi*(doy-80)/365.25)
    tmax = 27 + 7*seasonal + rng.normal(0, 3, days)
    tmin = tmax - rng.uniform(5, 12, days)
    humidity = np.clip(68 - 13*seasonal + rng.normal(0, 12, days), 25, 100)
    pressure = 1012 - 2.3*seasonal + rng.normal(0, 4, days)
    wind = np.clip(rng.gamma(2.4, 5, days), 0, 45)
    clouds = np.clip((humidity-30)*1.15 + rng.normal(0, 18, days), 0, 100)
    wet_prob = 1/(1+np.exp(-(-3.3 + .035*(humidity-60) + .022*(clouds-50) + .15*(1012-pressure))))
    rain = (rng.random(days) < wet_prob) * rng.gamma(1.8, 4, days)
    df = pd.DataFrame({'date': dates, 'temp_max': tmax, 'temp_min': tmin, 'humidity': humidity,
      'pressure': pressure, 'wind_speed': wind, 'cloud_cover': clouds, 'rainfall': rain})
    # Targets refer to the following day's max temperature and rain occurrence.
    df['next_temp_max'] = df.temp_max.shift(-1)
    df['rain_tomorrow'] = (df.rainfall.shift(-1) >= 1.0).astype(float)
    df.loc[df.index[-1], 'rain_tomorrow'] = np.nan
    return df.dropna().reset_index(drop=True)

if __name__ == '__main__':
    out = Path(__file__).parent/'data'/'weather_data.csv'
    out.parent.mkdir(exist_ok=True)
    make_dataset().to_csv(out, index=False)
    print(f'Wrote {len(make_dataset())} daily records to {out}')
