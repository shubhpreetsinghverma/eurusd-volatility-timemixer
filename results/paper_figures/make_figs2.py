import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='/mnt/user-data/uploads/fx-volatility-sentiment-timemixer/'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300,'axes.grid':True,'grid.alpha':.25})
d=pd.read_csv(R+'data/eurusd_model_ready.csv',parse_dates=['Date']).set_index('Date')
r=pd.read_csv(R+'results/rolling_predictions.csv',parse_dates=['Date'])
y=r.actual.values; T=len(y)
F={'Constant':r['Train-mean'].values,'Persistence':r.Persistence.values,'GARCH(1,1)':r.GARCH.values,
   'Implied vol. alone':d.loc[r.Date,'IMPL_VOL_1M'].values,'HAR+IV (linear)':r['HAR+IV (lin)'].values,
   'HAR+IV (log)':r['HAR+IV (log)'].values,'TimeMixer ensemble':r['TimeMixer ens5'].values}
rng=np.random.default_rng(42); L=22; B=5000
nb=int(np.ceil(T/L)); starts=np.arange(T-L+1)
idxs=[]
for _ in range(B):
    s=rng.choice(starts,nb); idxs.append((s[:,None]+np.arange(L)).ravel()[:T])
idxs=np.array(idxs)
def rmse(a,f):return np.sqrt(((a-f)**2).mean())
def mae(a,f):return np.abs(a-f).mean()
def qlike(a,f):z=a**2/f**2;return (z-np.log(z)-1).mean()
rows=[]
for n,f in F.items():
    res={}
    for mn,m in [('RMSE',rmse),('MAE',mae),('QLIKE',qlike)]:
        pt=m(y,f); bs=np.array([m(y[i],f[i]) for i in idxs]); lo,hi=np.percentile(bs,[2.5,97.5]); res[mn]=(pt,lo,hi)
    rows.append((n,res))
    print(n,{k:tuple(round(x,3) for x in v) for k,v in res.items()})
# paired CI on RMSE-difference vs HAR log and vs implied vol alone
def diffci(a,b):
    dd=np.array([rmse(y[i],F[a][i])-rmse(y[i],F[b][i]) for i in idxs]);return rmse(y,F[a])-rmse(y,F[b]),*np.percentile(dd,[2.5,97.5])
for a,b in [('TimeMixer ensemble','HAR+IV (log)'),('TimeMixer ensemble','Implied vol. alone'),('HAR+IV (log)','Implied vol. alone'),('GARCH(1,1)','Implied vol. alone')]:
    print('RMSE diff',a,'-',b,[round(x,3) for x in diffci(a,b)])
# forest plot
fig,ax=plt.subplots(figsize=(7.5,3.8))
order=list(F)[::-1]
cols={'Constant':'#7f7f7f','Persistence':'#9467bd','GARCH(1,1)':'#2ca02c','Implied vol. alone':'#d62728','HAR+IV (linear)':'#17becf','HAR+IV (log)':'#1f77b4','TimeMixer ensemble':'#ff7f0e'}
for i,n in enumerate(order):
    pt,lo,hi=dict(rows)[n]['RMSE']; ax.errorbar(pt,i,xerr=[[pt-lo],[hi-pt]],fmt='o',color=cols[n],capsize=3)
    ax.text(hi+.04,i,f'{pt:.2f} [{lo:.2f}, {hi:.2f}]',va='center',fontsize=8.5)
ax.set_yticks(range(len(order)));ax.set_yticklabels(order);ax.set_xlabel('Pooled rolling RMSE with 95% block-bootstrap interval');ax.set_xlim(1.2,3.6);ax.grid(axis='y',alpha=0)
fig.tight_layout();fig.savefig('fig_rmse_ci.png');plt.close()
# seeds
seeds=[c for c in r.columns if c.startswith('tm_seed')]
sr=[rmse(y,r[c].values) for c in seeds]; ens=rmse(y,r['TimeMixer ens5'].values)
print('seed RMSE',[round(x,3) for x in sr],'mean',np.mean(sr),'std',np.std(sr),'ens',ens)
fig,ax=plt.subplots(figsize=(6,3.6))
ax.bar([f'Seed {i}' for i in range(len(sr))],sr,color='#ff7f0e',alpha=.85)
ax.axhline(ens,color='k',ls='--',lw=1.2,label=f'Five-seed ensemble ({ens:.3f})')
hl=rmse(y,F['HAR+IV (log)']);ax.axhline(hl,color='#1f77b4',lw=1.2,label=f'HAR+IV (log) ({hl:.3f})')
for i,v in enumerate(sr): ax.text(i,v+.01,f'{v:.3f}',ha='center',fontsize=8.5)
ax.set_ylim(1.7,max(sr)+.15);ax.set_ylabel('Pooled rolling RMSE');ax.legend(loc='lower right',fontsize=8.5);ax.grid(axis='x',alpha=0)
fig.tight_layout();fig.savefig('fig_seed_contribution.png');plt.close()
