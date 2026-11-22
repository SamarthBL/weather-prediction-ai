"""Train/evaluate the temperature and rain prediction models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from data_gen import make_dataset

ROOT = Path(__file__).parent
DATA = ROOT/'data'/'weather_data.csv'
MODEL_DIR = ROOT/'models'
FEATURES = ['temp_max','temp_min','humidity','pressure','wind_speed','cloud_cover','rainfall','temp_range','doy_sin','doy_cos']

def load_data():
    if not DATA.exists():
        DATA.parent.mkdir(exist_ok=True)
        make_dataset().to_csv(DATA,index=False)
    df = pd.read_csv(DATA, parse_dates=['date'])
    d = df.date.dt.dayofyear
    df['temp_range'] = df.temp_max-df.temp_min
    df['doy_sin'] = np.sin(2*np.pi*d/365.25)
    df['doy_cos'] = np.cos(2*np.pi*d/365.25)
    return df

def train():
    df=load_data(); cut=int(len(df)*.8); train_df=df.iloc[:cut]; test=df.iloc[cut:]
    Xtr=train_df[FEATURES]; Xte=test[FEATURES]
    reg=RandomForestRegressor(n_estimators=180,max_depth=12,min_samples_leaf=2,random_state=42,n_jobs=-1)
    clf=RandomForestClassifier(n_estimators=180,max_depth=10,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=-1)
    reg.fit(Xtr,train_df.next_temp_max); clf.fit(Xtr,train_df.rain_tomorrow.astype(int))
    yp=reg.predict(Xte); prob=clf.predict_proba(Xte)[:,1]; yc=(prob>=.5).astype(int); truth=test.rain_tomorrow.astype(int)
    metrics={'temperature':{'MAE':float(mean_absolute_error(test.next_temp_max,yp)),'RMSE':float(np.sqrt(mean_squared_error(test.next_temp_max,yp))),'R2':float(r2_score(test.next_temp_max,yp))},
      'rain':{'accuracy':float(accuracy_score(truth,yc)),'precision':float(precision_score(truth,yc,zero_division=0)),'recall':float(recall_score(truth,yc,zero_division=0)),'f1':float(f1_score(truth,yc,zero_division=0)),'auc':float(roc_auc_score(truth,prob))},
      'evaluation':{'train_days':int(len(train_df)),'test_days':int(len(test)),'test_start':str(test.date.iloc[0].date()),'test_end':str(test.date.iloc[-1].date())},
      'confusion_matrix':confusion_matrix(truth,yc).tolist(),
      'importance':dict(sorted(zip(FEATURES,map(float,reg.feature_importances_)),key=lambda x:x[1],reverse=True))}
    MODEL_DIR.mkdir(exist_ok=True)
    import joblib
    joblib.dump({'regressor':reg,'classifier':clf,'features':FEATURES},MODEL_DIR/'weather_models.joblib')
    (MODEL_DIR/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2)); return metrics

if __name__=='__main__': train()
