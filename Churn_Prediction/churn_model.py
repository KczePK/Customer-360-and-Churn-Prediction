"""
Customer churn prediction: probabilistic classification on monthly RFM snapshots.

Question: of the service customers who bought in the last 12 months, who will NOT buy again in the next 12 months?

Data (demo, see data/): one row per customer per month-end snapshot, with recency / frequency / monetary
features computed only from history up to that month (fact_customer_rfm_monthly in the Power BI model).

Design
- Population : customers active at the snapshot (freq_12m > 0).
- Label      : churn = 1 when the customer has no invoice in the 12 months after the snapshot.
- Snapshots  : quarter ends only (Mar / Jun / Sep / Dec) to limit near-duplicate rows of the same customer.
- Split      : by time, never random  ->  train <= 2022-12  |  calibration 2023  |  test 2024-03 .. 2025-06.
               The last 12 months have no complete label yet: they are only scored.
- Models     : logistic regression (baseline, explainable) and gradient boosting; both calibrated with
               Platt scaling (sigmoid) on the calibration year so that a score of 0.30 means about 30 % churn.
               Sigmoid rather than isotonic: one year of snapshots is too few rows for a stable isotonic fit.
- Selection  : the model with the best PR-AUC on the calibration year (the test period is not used to choose).
- Metrics    : ROC-AUC and PR-AUC (ranking), Brier score and log loss (probability quality), calibration curve,
               lift and capture in the top 10 % (who to contact first).
- Output     : outputs/churn_scores.csv = churn probability and risk band for every active customer at the
               latest snapshot, ready to load into Power BI.

Run:  pip install -r requirements.txt  &&  python churn_model.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).parent
DATA, OUT = HERE / "data", HERE / "outputs"
OUT.mkdir(exist_ok=True)

HORIZON = 12                      # months ahead
QUARTER_ENDS = {3, 6, 9, 12}
TRAIN_END, CALIB_END = "2022-12-01", "2023-12-01"
SEED = 42
RISK_BANDS = [(0.50, "High"), (0.25, "Medium"), (0.0, "Low")]

NUMERIC = ["recency_months", "tenure_months", "freq_12m", "freq_ltd", "active_months_12m", "avg_gap_months",
           "overdue_ratio", "r_score", "f_score", "m_score", "log_mon_12m", "log_mon_ltd", "log_avg_doc_value_12m",
           "yoy_change_signed_log", "first_time_flag"]
CATEGORICAL = ["segment", "customer_type", "industry_section"]


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def load():
    rfm = pd.read_csv(DATA / "rfm_monthly_demo.csv", parse_dates=["snapshot_month"])
    cust = pd.read_csv(DATA / "customers_demo.csv").rename(columns={"customer_id": "enterprise_id"})
    cust["industry_section"] = cust.industry_section.fillna("").str[:1].replace("", "Unknown")
    return rfm.merge(cust, on="enterprise_id", how="left")


def add_label(rfm):
    """churn = 1 if no active month in (t, t + HORIZON]; None when the horizon goes past the last data month."""
    last_month = rfm.snapshot_month.max()
    active = rfm.loc[rfm.active_this_month == 1, ["enterprise_id", "snapshot_month"]]
    by_cust = active.groupby("enterprise_id").snapshot_month.apply(lambda s: np.sort(s.values)).to_dict()

    def churned(row):
        months = by_cust.get(row.enterprise_id, np.array([], dtype="datetime64[ns]"))
        t = np.datetime64(row.snapshot_month)
        end = np.datetime64(row.snapshot_month + pd.DateOffset(months=HORIZON))
        return int(not ((months > t) & (months <= end)).any())

    rfm = rfm.copy()
    rfm["horizon_end"] = rfm.snapshot_month + pd.DateOffset(months=HORIZON)
    rfm["churn"] = rfm.apply(churned, axis=1).astype("Int64")
    rfm.loc[rfm.horizon_end > last_month, "churn"] = pd.NA
    return rfm


def features(df):
    df = df.copy()
    df["log_mon_12m"] = np.log1p(df.mon_12m.clip(lower=0))
    df["log_mon_ltd"] = np.log1p(df.mon_ltd.clip(lower=0))
    df["log_avg_doc_value_12m"] = np.log1p(df.avg_doc_value_12m.clip(lower=0))
    df["yoy_change_signed_log"] = signed_log(df.mon_12m_yoy_change.fillna(0))
    for c in CATEGORICAL:
        df[c] = df[c].fillna("Unknown").astype(str)
    return df


def preprocess(scale):
    num = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
    return ColumnTransformer([("num", Pipeline(num), NUMERIC),
                              ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CATEGORICAL)])


def lift_table(y, p, bins=10):
    t = pd.DataFrame({"y": y, "p": p}).sort_values("p", ascending=False).reset_index(drop=True)
    t["decile"] = pd.qcut(t.index, bins, labels=range(1, bins + 1))
    g = t.groupby("decile", observed=True).agg(customers=("y", "size"), churners=("y", "sum"), mean_p=("p", "mean"))
    g["churn_rate"] = g.churners / g.customers
    g["lift"] = g.churn_rate / t.y.mean()
    g["cum_capture"] = g.churners.cumsum() / t.y.sum()
    return g


def main():
    df = features(add_label(load()))
    df = df[(df.freq_12m > 0) & (df.snapshot_month.dt.month.isin(QUARTER_ENDS))]
    lab = df[df.churn.notna()].copy(); lab["churn"] = lab.churn.astype(int)
    train = lab[lab.snapshot_month <= TRAIN_END]
    calib = lab[(lab.snapshot_month > TRAIN_END) & (lab.snapshot_month <= CALIB_END)]
    test = lab[lab.snapshot_month > CALIB_END]
    latest = df[df.snapshot_month == df.snapshot_month.max()]
    X = lambda d: d[NUMERIC + CATEGORICAL]
    print(f"rows  train {len(train)}  calib {len(calib)}  test {len(test)}  | churn rate train {train.churn.mean():.1%} "
          f"test {test.churn.mean():.1%} | customers {lab.enterprise_id.nunique()}")

    models = {
        "Logistic regression": Pipeline([("prep", preprocess(True)),
                                         ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight=None))]),
        "Gradient boosting": Pipeline([("prep", preprocess(False)),
                                       ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                                              min_samples_leaf=40, l2_regularization=1.0,
                                                                              random_state=SEED))]),
    }
    results, probs, fitted = {}, {}, {}
    for name, pipe in models.items():
        pipe.fit(X(train), train.churn)
        cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="sigmoid").fit(X(calib), calib.churn)
        for tag, m in [("raw", pipe), ("calibrated", cal)]:
            p = m.predict_proba(X(test))[:, 1]
            lt = lift_table(test.churn.values, p)
            results[f"{name} ({tag})"] = {
                "roc_auc": round(roc_auc_score(test.churn, p), 3),
                "pr_auc": round(average_precision_score(test.churn, p), 3),
                "brier": round(brier_score_loss(test.churn, p), 4),
                "log_loss": round(log_loss(test.churn, np.clip(p, 1e-6, 1 - 1e-6)), 4),
                "top10_lift": round(float(lt.lift.iloc[0]), 2),
                "top10_capture": round(float(lt.cum_capture.iloc[0]), 3),
            }
            probs[f"{name} ({tag})"] = p
        fitted[name] = cal
        results[f"{name} (calibrated)"]["calib_year_pr_auc"] = round(average_precision_score(calib.churn, pipe.predict_proba(X(calib))[:, 1]), 3)
    results["_base_rate_test"] = round(float(test.churn.mean()), 4)
    results["_brier_if_always_base_rate"] = round(brier_score_loss(test.churn, np.full(len(test), train.churn.mean())), 4)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(pd.DataFrame({k: v for k, v in results.items() if not k.startswith("_")}).T.to_string())

    best = max(models, key=lambda n: results[f"{n} (calibrated)"]["calib_year_pr_auc"])
    p_best = probs[f"{best} (calibrated)"]
    print("selected model:", best)

    # ---- charts
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for name in models:
        for tag, ls in [("raw", ":"), ("calibrated", "-")]:
            k = f"{name} ({tag})"; p = probs[k]
            fx, fy = roc_curve(test.churn, p)[:2]; ax[0].plot(fx, fy, ls, label=f"{k}  AUC {results[k]['roc_auc']}")
            pr, rc = precision_recall_curve(test.churn, p)[:2]; ax[1].plot(rc, pr, ls, label=f"{k}  AP {results[k]['pr_auc']}")
            fp, mp = calibration_curve(test.churn, p, n_bins=8, strategy="quantile"); ax[2].plot(mp, fp, ls + "o", ms=3, label=k)
    ax[0].plot([0, 1], [0, 1], color="grey", lw=0.8); ax[2].plot([0, 1], [0, 1], color="grey", lw=0.8)
    ax[1].axhline(test.churn.mean(), color="grey", lw=0.8)
    for a, t, xl, yl in [(ax[0], "ROC", "False positive rate", "True positive rate"),
                         (ax[1], "Precision-recall", "Recall", "Precision"),
                         (ax[2], "Calibration (test)", "Predicted churn probability", "Observed churn rate")]:
        a.set_title(t); a.set_xlabel(xl); a.set_ylabel(yl); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "evaluation.png", dpi=130); plt.close(fig)

    lt = lift_table(test.churn.values, p_best); lt.to_csv(OUT / "lift_by_decile.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(lt.index.astype(str), lt.churn_rate, color="#2563EB"); ax.axhline(test.churn.mean(), color="grey", ls="--", label="average")
    ax.set_title(f"Churn rate by risk decile (test, {best})"); ax.set_xlabel("Decile (1 = highest predicted risk)")
    ax.set_ylabel("Observed churn rate"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "lift_by_decile.png", dpi=130); plt.close(fig)

    imp = permutation_importance(fitted[best], X(test), test.churn, scoring="average_precision", n_repeats=10, random_state=SEED)
    fi = pd.Series(imp.importances_mean, index=NUMERIC + CATEGORICAL).sort_values()
    fi.to_csv(OUT / "feature_importance.csv", header=["pr_auc_drop"])
    fig, ax = plt.subplots(figsize=(7, 5)); fi.tail(12).plot.barh(ax=ax, color="#2563EB")
    ax.set_title("Permutation importance (drop in PR-AUC on test)"); fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=130); plt.close(fig)

    # ---- score the latest snapshot
    s = latest[["enterprise_id", "snapshot_month", "segment", "recency_months", "freq_12m", "mon_12m", "overdue_ratio"]].copy()
    s["churn_probability"] = fitted[best].predict_proba(X(latest))[:, 1].round(4)
    s["risk_band"] = s.churn_probability.map(lambda p: next(b for t, b in RISK_BANDS if p >= t))
    s["model"] = best
    s.sort_values("churn_probability", ascending=False).to_csv(OUT / "churn_scores.csv", index=False)
    print(f"scored {len(s)} active customers at {s.snapshot_month.max():%Y-%m}:", s.risk_band.value_counts().to_dict())


if __name__ == "__main__":
    main()
