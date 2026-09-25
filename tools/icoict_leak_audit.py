"""Audit of the ICoICT 2025 pipeline (pharma-sales-daily.ipynb / pharma-sales-weekly.ipynb).
Reproduces the notebook feature construction and the averaging hybrid, then removes one
defect at a time:
  A  notebook as written: rolling_mean_k = y.rolling(k).mean()  (window INCLUDES y_t)
  B  same, but rolling mean shifted by one step (window ends at t-1)
Split 80/20 as stated in the ICoICT paper (daily notebook itself used 60/20/20; both reported).
Lag k = argmax ACF(1..26) on the FULL series, as in the notebook (kept in A and B so only the
rolling-mean defect changes)."""
import sys, numpy as np, pandas as pd, xgboost as xgb
from statsmodels.tsa.stattools import acf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
CATS=['M01AB','M01AE','N02BA','N02BE','N05B','N05C','R03','R06']
GRID={'n_estimators':[50,100,200,300,400,500],'max_depth':[3,5],'learning_rate':[0.05,0.1,0.2]}
def rmse(a,b): return float(np.sqrt(mean_squared_error(a,b)))
def run(path, train_frac, label):
    data=pd.read_csv(path); rows=[]
    for c in CATS:
        a=acf(data[c],nlags=26); k=int(np.argmax(a[1:])+1)
        for variant in ['A_leaky','B_shifted']:
            d=pd.DataFrame({'y':data[c].values})
            for L in range(1,k+1): d[f'lag_{L}']=d.y.shift(L)
            rm=d.y.rolling(window=k).mean()
            d[f'rm_{k}']=rm if variant=='A_leaky' else rm.shift(1)
            d=d.dropna().reset_index(drop=True)
            n=int(len(d)*train_frac); tr,te=d.iloc[:n],d.iloc[n:]
            Xtr,ytr=tr.drop(columns='y').values,tr.y.values; Xte,yte=te.drop(columns='y').values,te.y.values
            lr=LinearRegression().fit(Xtr,ytr); plr=lr.predict(Xte)
            gs=GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror',random_state=42,n_jobs=4),GRID,scoring='neg_mean_squared_error').fit(Xtr,ytr)
            m=xgb.XGBRegressor(objective='reg:squarederror',random_state=42,n_jobs=4,**gs.best_params_).fit(Xtr,ytr); px=m.predict(Xte)
            naive=te[f'lag_1'].values
            rows.append(dict(granularity=label,split=f'{int(train_frac*100)}/{100-int(train_frac*100)}',category=c,k=k,variant=variant,
                rmse_LR=rmse(yte,plr),rmse_XGB=rmse(yte,px),rmse_hybrid_avg=rmse(yte,(plr+px)/2),rmse_naive=rmse(yte,naive)))
            print(rows[-1],flush=True)
    return rows
R=[]
base=str(__import__('pathlib').Path(__file__).resolve().parents[1]/'data/raw/pharma-sales')+'/'
R+=run(base+'salesdaily.csv',0.8,'daily')
R+=run(base+'salesweekly.csv',0.8,'weekly')
pd.DataFrame(R).to_csv(str(__import__('pathlib').Path(__file__).resolve().parents[1]/'results/icoict_leak_audit.csv'),index=False)
