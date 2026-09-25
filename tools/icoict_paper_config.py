"""ICoICT paper configuration as described in its Section II: lag_1..lag_12, rolling means 3 and 7,
80/20 split, XGBoost grid lr {0.1,0.2}, depth 3, n_estimators 200, subsample {0.8,1.0}, averaging hybrid."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
CATS=['M01AB','M01AE','N02BA','N02BE','N05B','N05C','R03','R06']
PAPER={'M01AB':0.57,'M01AE':0.47,'N02BA':0.68,'N02BE':1.30,'N05B':0.79,'N05C':0.54,'R03':1.79,'R06':0.03}
GRID={'learning_rate':[0.1,0.2],'max_depth':[3],'n_estimators':[200],'subsample':[0.8,1.0]}
r=lambda a,b: float(np.sqrt(mean_squared_error(a,b)))
data=pd.read_csv(str(__import__('pathlib').Path(__file__).resolve().parents[1]/'data/raw/pharma-sales/salesdaily.csv')); rows=[]
for c in CATS:
  for v in ['leaky','shifted']:
    d=pd.DataFrame({'y':data[c].values})
    for L in range(1,13): d[f'lag_{L}']=d.y.shift(L)
    for w in (3,7):
        rm=d.y.rolling(w).mean(); d[f'rm{w}']=rm if v=='leaky' else rm.shift(1)
    d=d.dropna().reset_index(drop=True); n=int(len(d)*.8)
    Xtr,ytr=d.iloc[:n].drop(columns='y').values,d.y.values[:n]; Xte,yte=d.iloc[n:].drop(columns='y').values,d.y.values[n:]
    plr=LinearRegression().fit(Xtr,ytr).predict(Xte)
    gs=GridSearchCV(xgb.XGBRegressor(random_state=42,n_jobs=4),GRID,scoring='neg_mean_squared_error').fit(Xtr,ytr)
    px=xgb.XGBRegressor(random_state=42,n_jobs=4,**gs.best_params_).fit(Xtr,ytr).predict(Xte)
    rows.append(dict(category=c,variant=v,paper_rmse=PAPER[c],rmse_LR=r(yte,plr),rmse_XGB=r(yte,px),rmse_hybrid_avg=r(yte,(plr+px)/2),rmse_naive=r(yte,d.lag_1.values[n:])))
df=pd.DataFrame(rows); df.to_csv(str(__import__('pathlib').Path(__file__).resolve().parents[1]/'results/icoict_paper_config.csv'),index=False)
pd.set_option('display.width',200); print(df.round(4))
