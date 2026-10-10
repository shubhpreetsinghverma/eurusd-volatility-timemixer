"""Figures 1-6 of the paper (data overview, test and rolling forecasts, RMSE by fold, scatter, SHAP and permutation importance).

Reads data/eurusd_model_ready.csv and the forecasts in results/, writes PNGs to results/paper_figures/.
Usage: python scripts/paper_figures.py
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = str(ROOT) + "/"
OUT = ROOT / "results" / "paper_figures"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)
C = {
    "Actual": "black",
    "HAR+IV (log)": "#1f77b4",
    "HAR+IV (lin)": "#17becf",
    "TimeMixer": "#ff7f0e",
    "GARCH": "#2ca02c",
    "Persistence": "#9467bd",
    "Constant": "#7f7f7f",
}
d = pd.read_csv(R + "data/eurusd_model_ready.csv", parse_dates=["Date"])
t = pd.read_csv(R + "results/test_predictions.csv", parse_dates=["Date"])
r = pd.read_csv(R + "results/rolling_predictions.csv", parse_dates=["Date"])


def lab(ax, s):
    ax.text(-0.07, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=11)


fmt = mdates.DateFormatter("%Y-%m")

# Fig 1 data overview
blocks = [
    (0, 868, "Train", "#e8eef7"),
    (890, 1054, "Validation", "#fdf0d5"),
    (1076, 1240, "Test", "#e3f3e3"),
]
fig, ax = plt.subplots(3, 1, figsize=(8, 7.2), sharex=True)
for a in ax:
    for s, e, n, c in blocks:
        a.axvspan(d.Date[s], d.Date[e - 1], color=c, alpha=0.9, lw=0)
ax[0].plot(d.Date, d.PX_LAST, color="k", lw=1)
ax[0].set_ylabel("EUR/USD close")
lab(ax[0], "(a)")
for s, e, n, c in blocks:
    ax[0].text(
        d.Date[(s + e) // 2], ax[0].get_ylim()[1] * 0.995, n, ha="center", va="top", fontsize=8.5
    )
ax[1].plot(d.Date, d.realized_vol_22d, label="Realized vol (22 d)", color="k", lw=1)
ax[1].plot(d.Date, d.IMPL_VOL_1M, label="Implied vol (1 month)", color="#d62728", lw=1)
ax[1].plot(d.Date, d.park_m, label="Parkinson (monthly)", color="#1f77b4", lw=1, alpha=0.8)
ax[1].set_ylabel("Annualised volatility (%)")
ax[1].legend(ncol=3, loc="upper right")
lab(ax[1], "(b)")
ax[2].plot(
    d.Date, d.target_22d.rolling(66).mean(), color="k", lw=1.2, label="Target, 66-day rolling mean"
)
ax[2].plot(
    d.Date,
    d.IMPL_VOL_1M.rolling(66).mean(),
    color="#d62728",
    lw=1.2,
    label="Implied vol, 66-day rolling mean",
)
ax[2].set_ylabel("Level (%)")
ax[2].legend(loc="upper right")
lab(ax[2], "(c)")
ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
fig.savefig(OUT / "fig_data.png")
plt.close()

# Fig 2 single-split test
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
ax[0].plot(t.Date, t.actual, color="k", lw=2, label="Actual (next 22-day realized vol)")
for k, n in [
    ("HAR+IV (log)", "HAR+IV (log)"),
    ("TimeMixer ens10", "TimeMixer"),
    ("GARCH", "GARCH"),
]:
    ax[0].plot(t.Date, t[k], color=C[n], lw=1.3, label=n)
ax[0].set_ylabel("Annualised volatility (%)")
ax[0].legend(ncol=2)
lab(ax[0], "(a)")
for k, n in [
    ("HAR+IV (log)", "HAR+IV (log)"),
    ("TimeMixer ens10", "TimeMixer"),
    ("GARCH", "GARCH"),
]:
    ax[1].plot(t.Date, t[k] - t.actual, color=C[n], lw=1.2, label=n)
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_ylabel("Forecast error (pp)")
lab(ax[1], "(b)")
ax[1].xaxis.set_major_formatter(fmt)
fig.autofmt_xdate(rotation=0, ha="center")
fig.tight_layout()
fig.savefig(OUT / "fig_test_forecasts.png")
plt.close()

# Fig 3 rolling
cols = {"HAR+IV (log)": "HAR+IV (log)", "TimeMixer ens5": "TimeMixer", "GARCH": "GARCH"}
starts = r.groupby("fold").Date.min().tolist()[1:]
fig, ax = plt.subplots(
    2, 1, figsize=(8.5, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
)
ax[0].plot(r.Date, r.actual, color="k", lw=1.8, label="Actual")
for k, n in cols.items():
    ax[0].plot(r.Date, r[k], color=C[n], lw=1.1, label=n)
ax[0].set_ylabel("Annualised volatility (%)")
ax[0].legend(ncol=4, loc="upper right")
lab(ax[0], "(a)")
diff = ((r["TimeMixer ens5"] - r.actual) ** 2 - (r["HAR+IV (log)"] - r.actual) ** 2).cumsum()
ax[1].plot(r.Date, diff, color="#ff7f0e", lw=1.5)
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_ylabel("Cumulative loss\ndifference")
lab(ax[1], "(b)")
ax[1].text(
    0.01,
    0.92,
    "above 0: HAR+IV (log) ahead of TimeMixer",
    transform=ax[1].transAxes,
    va="top",
    fontsize=8.5,
)
for a in ax:
    for s in starts:
        a.axvline(s, color="gray", ls=":", lw=1)
ax[1].xaxis.set_major_formatter(fmt)
fig.tight_layout()
fig.savefig(OUT / "fig_rolling_forecasts.png")
plt.close()
print("cum diff end", diff.iloc[-1])

# Fig 4 by fold
mods = [
    ("Train-mean", "Constant"),
    ("Persistence", "Persistence"),
    ("GARCH", "GARCH"),
    ("HAR+IV (lin)", "HAR+IV (lin)"),
    ("HAR+IV (log)", "HAR+IV (log)"),
    ("TimeMixer ens5", "TimeMixer"),
]
fs = sorted(r.fold.unique())
w = 0.13
fig, ax = plt.subplots(figsize=(8, 4.2))
for i, (k, n) in enumerate(mods):
    v = [np.sqrt(((r[r.fold == f][k] - r[r.fold == f].actual) ** 2).mean()) for f in fs]
    ax.bar(np.arange(4) + (i - 2.5) * w, v, w, label=n, color=C[n])
ax.set_xticks(range(4))
ax.set_xticklabels([f"Fold {i+1}\n{r[r.fold==f].Date.min():%Y-%m-%d}" for i, f in enumerate(fs)])
ax.set_ylabel("RMSE")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
ax.grid(axis="x", alpha=0)
fig.tight_layout()
fig.savefig(OUT / "fig_rolling_by_fold.png")
plt.close()

# Fig 5 scatter
fig, ax = plt.subplots(1, 3, figsize=(9, 3.4), sharex=True, sharey=True)
for a, (k, n) in zip(
    ax, [("HAR+IV (log)", "HAR+IV (log)"), ("TimeMixer ens5", "TimeMixer"), ("GARCH", "GARCH")]
):
    a.scatter(r[k], r.actual, s=6, alpha=0.45, color=C[n])
    lim = [2.5, 14.5]
    a.plot(lim, lim, "k--", lw=0.8)
    c = np.corrcoef(r[k], r.actual)[0, 1]
    a.set_title(f"{n}  (r = {c:.2f})")
    a.set_xlabel("Forecast (%)")
ax[0].set_ylabel("Actual (%)")
fig.tight_layout()
fig.savefig(OUT / "fig_scatter.png")
plt.close()

# Fig 6 SHAP + permutation
s = pd.read_csv(R + "results/shap_importance.csv").iloc[::-1]
nm = {
    "IMPL_VOL_1M": "Implied vol (1M)",
    "park_w": "Parkinson (weekly)",
    "park_d": "Parkinson (daily)",
    "realized_vol_22d": "Realized vol (22 d)",
    "VOL_90D": "Hist. vol (90 d)",
    "VOL_10D": "Hist. vol (10 d)",
    "park_m": "Parkinson (monthly)",
}
perm = {
    "Realized vol (22 d)": (0.146, 0.070),
    "Parkinson (daily)": (0.044, 0.021),
    "Parkinson (weekly)": (0.260, -0.091),
    "Parkinson (monthly)": (0.038, -0.029),
    "Implied vol (1M)": (0.287, -0.072),
    "Hist. vol (10 d)": (0.139, -0.003),
    "Hist. vol (90 d)": (0.001, -0.092),
}
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
ax[0].barh([nm[f] for f in s.feature], s["share (%)"], color="#1f77b4")
for i, v in enumerate(s["share (%)"]):
    ax[0].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8.5)
ax[0].set_xlabel("Share of mean |SHAP| (%)")
ax[0].set_xlim(0, 48)
lab(ax[0], "(a)")
ax[0].grid(axis="y", alpha=0)
names = [nm[f] for f in s.feature]
y = np.arange(7)
ax[1].barh(y + 0.2, [perm[n][0] for n in names], 0.38, label="Validation", color="#ff7f0e")
ax[1].barh(y - 0.2, [perm[n][1] for n in names], 0.38, label="Test", color="#2ca02c")
ax[1].axvline(0, color="k", lw=0.8)
ax[1].set_yticks(y)
ax[1].set_yticklabels([])
ax[1].set_xlabel("Change in RMSE after shuffling")
ax[1].legend(loc="lower right")
lab(ax[1], "(b)")
ax[1].grid(axis="y", alpha=0)
fig.tight_layout()
fig.savefig(OUT / "fig_shap_perm.png")
plt.close()
